"""Fenced Two-Phase Atomic Checkpoint & Torn-Shard Recovery Engine.

Enforces monotonically increasing `lease_epoch` fencing tokens in SQLite WAL
(blocking stale/preempted workers from corrupting newer checkpoints), writes
multi-rank checkpoints via two-phase atomic rename (`_COMMITTED_MANIFEST.json`)
with per-shard SHA-256 byte digests, quarantines torn/partial crash shards on disk,
and resumes from the highest verified step with bit-exact parameter parity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FencedWriteRejectedError(RuntimeError):
    """Raised when a stale/preempted worker attempts a checkpoint write with an outdated lease_epoch."""


class FencedCheckpointEngine:
    """Distributed multi-rank checkpoint coordinator with monotonic lease fencing and torn-shard recovery."""

    NUM_RANKS = 4
    PARAM_DIM = 16
    BATCH_TOKENS = 256

    def __init__(self, runtime_dir: Path, db_path: Path) -> None:
        self.runtime_dir = runtime_dir
        self.ckpt_root = runtime_dir / "checkpoints"
        self.db_path = db_path
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_leases (
                    job_id TEXT PRIMARY KEY,
                    active_worker_id TEXT NOT NULL,
                    lease_epoch INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoint_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    lease_epoch INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM checkpoint_leases;")
            conn.execute("DELETE FROM checkpoint_events;")
        if self.ckpt_root.exists():
            shutil.rmtree(self.ckpt_root, ignore_errors=True)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def _log_event(
        self,
        job_id: str,
        step: int,
        worker_id: str,
        lease_epoch: int,
        event_type: str,
        status: str,
        detail: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_events (
                    job_id, step, worker_id, lease_epoch, event_type, status, detail, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, step, worker_id, lease_epoch, event_type, status, detail, _utc_now_iso()),
            )

    def acquire_lease(self, job_id: str, worker_id: str) -> Dict[str, Any]:
        """Atomically increment monotonic lease_epoch in SQLite WAL and assign ownership."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT lease_epoch FROM checkpoint_leases WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            next_epoch = 1 if row is None else int(row["lease_epoch"]) + 1
            now_str = _utc_now_iso()
            conn.execute(
                """
                INSERT INTO checkpoint_leases (job_id, active_worker_id, lease_epoch, updated_at_utc)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    active_worker_id = excluded.active_worker_id,
                    lease_epoch = excluded.lease_epoch,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (job_id, worker_id, next_epoch, now_str),
            )
        self._log_event(
            job_id=job_id,
            step=0,
            worker_id=worker_id,
            lease_epoch=next_epoch,
            event_type="LEASE_ACQUIRED",
            status="ACTIVE",
            detail=f"Worker '{worker_id}' granted monotonic fencing token epoch={next_epoch}.",
        )
        return {"job_id": job_id, "worker_id": worker_id, "lease_epoch": next_epoch}

    def get_active_lease(self, job_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id, active_worker_id, lease_epoch, updated_at_utc FROM checkpoint_leases WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return {"job_id": job_id, "active_worker_id": "none", "lease_epoch": 0, "updated_at_utc": None}
        return dict(row)

    @classmethod
    def _initial_rank_states(cls) -> List[Dict[str, Any]]:
        states: List[Dict[str, Any]] = []
        for r in range(cls.NUM_RANKS):
            w = [round(0.02 * (r + 1) * ((i % 5) - 2), 10) for i in range(cls.PARAM_DIM)]
            m = [0.0 for _ in range(cls.PARAM_DIM)]
            v = [0.0 for _ in range(cls.PARAM_DIM)]
            states.append({"rank": r, "weights": w, "m": m, "v": v})
        return states

    @classmethod
    def _advance_one_step(
        cls,
        step_num: int,
        rank_states: List[Dict[str, Any]],
        lr: float = 0.004,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """Deterministic 4-rank stateful AdamW-style parameter update."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        next_states: List[Dict[str, Any]] = []
        step_loss = 0.0

        for r_state in rank_states:
            r = int(r_state["rank"])
            w = list(r_state["weights"])
            m = list(r_state["m"])
            v = list(r_state["v"])
            for i in range(cls.PARAM_DIM):
                # Deterministic synthetic token feature & target tied strictly to dataloader step_num
                x = math.sin(step_num * 0.37 + r * 0.53 + i * 0.19)
                target = math.cos(step_num * 0.23 - r * 0.41 + i * 0.11) * 0.5
                pred = w[i] * x
                diff = pred - target
                step_loss += diff * diff
                grad = 2.0 * diff * x + 0.001 * w[i]
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
                m_hat = m[i] / (1.0 - (beta1 ** step_num))
                v_hat = v[i] / (1.0 - (beta2 ** step_num))
                w[i] = w[i] - lr * (m_hat / (math.sqrt(v_hat) + eps))
            next_states.append({"rank": r, "weights": w, "m": m, "v": v})

        mean_loss = round(step_loss / (cls.NUM_RANKS * cls.PARAM_DIM), 6)
        cursor = {
            "completed_step": step_num,
            "token_offset": step_num * cls.BATCH_TOKENS * cls.NUM_RANKS,
            "shard_index": (step_num - 1) // 3,
        }
        return next_states, cursor, mean_loss

    def commit_checkpoint(
        self,
        job_id: str,
        step: int,
        worker_id: str,
        lease_epoch: int,
        rank_states: List[Dict[str, Any]],
        dataloader_cursor: Dict[str, Any],
        loss: float,
    ) -> Dict[str, Any]:
        """Enforce monotonic lease_epoch fence, write per-rank shards, and atomically commit manifest."""
        t0 = time.perf_counter()
        active = self.get_active_lease(job_id)
        active_epoch = int(active["lease_epoch"])

        if lease_epoch < active_epoch or active_epoch == 0:
            msg = (
                f"FENCING_TOKEN_STALE: worker '{worker_id}' presented epoch={lease_epoch} "
                f"< active epoch={active_epoch} owned by '{active['active_worker_id']}'."
            )
            self._log_event(
                job_id=job_id,
                step=step,
                worker_id=worker_id,
                lease_epoch=lease_epoch,
                event_type="ZOMBIE_WRITE_BLOCKED",
                status="FENCED_REJECTED",
                detail=msg,
            )
            raise FencedWriteRejectedError(msg)

        step_dir = self.ckpt_root / job_id / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        shard_meta: Dict[str, Dict[str, Any]] = {}
        total_bytes = 0
        for r_state in rank_states:
            r = int(r_state["rank"])
            shard_name = f"rank_{r}.shard"
            shard_path = step_dir / shard_name
            shard_payload = _canonical_json(
                {
                    "job_id": job_id,
                    "step": step,
                    "lease_epoch": lease_epoch,
                    "rank": r,
                    "state": r_state,
                }
            ).encode("utf-8")
            tmp_shard = step_dir / f".{shard_name}.tmp"
            tmp_shard.write_bytes(shard_payload)
            os.replace(tmp_shard, shard_path)
            digest = _sha256_bytes(shard_payload)
            shard_meta[shard_name] = {
                "rank": r,
                "byte_size": len(shard_payload),
                "sha256": digest,
            }
            total_bytes += len(shard_payload)

        # Re-check fencing token right before atomic two-phase manifest commit
        active_after = self.get_active_lease(job_id)
        if lease_epoch < int(active_after["lease_epoch"]):
            shutil.rmtree(step_dir, ignore_errors=True)
            msg = f"FENCING_TOKEN_PREEMPTED_MID_WRITE: epoch={lease_epoch} < {active_after['lease_epoch']}."
            self._log_event(
                job_id=job_id,
                step=step,
                worker_id=worker_id,
                lease_epoch=lease_epoch,
                event_type="ZOMBIE_WRITE_BLOCKED",
                status="FENCED_REJECTED",
                detail=msg,
            )
            raise FencedWriteRejectedError(msg)

        manifest_body = {
            "job_id": job_id,
            "step": step,
            "worker_id": worker_id,
            "lease_epoch": lease_epoch,
            "loss": loss,
            "dataloader_cursor": dataloader_cursor,
            "num_ranks": len(rank_states),
            "shards": shard_meta,
            "committed_at_utc": _utc_now_iso(),
        }
        manifest_bytes = _canonical_json(manifest_body).encode("utf-8")
        manifest_sha = _sha256_bytes(manifest_bytes)
        manifest_body["manifest_sha256"] = manifest_sha
        final_manifest_bytes = json.dumps(manifest_body, indent=2, sort_keys=True).encode("utf-8")

        tmp_manifest = step_dir / "_MANIFEST.json.tmp"
        committed_manifest = step_dir / "_COMMITTED_MANIFEST.json"
        tmp_manifest.write_bytes(final_manifest_bytes)
        os.replace(tmp_manifest, committed_manifest)

        commit_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self._log_event(
            job_id=job_id,
            step=step,
            worker_id=worker_id,
            lease_epoch=lease_epoch,
            event_type="CHECKPOINT_COMMITTED",
            status="COMMITTED",
            detail=f"Committed 4-rank shards ({total_bytes} B) at step={step}, cursor_tokens={dataloader_cursor['token_offset']}.",
        )
        return {
            "job_id": job_id,
            "step": step,
            "lease_epoch": lease_epoch,
            "commit_ms": commit_ms,
            "total_bytes": total_bytes,
            "manifest_sha256": manifest_sha,
            "step_dir": str(step_dir),
        }

    def inject_disk_fault(self, job_id: str, step: int, fault_mode: str = "torn_rank_shard") -> Dict[str, Any]:
        """Inject a real torn shard or missing commit marker onto disk at `step_{step:04d}`."""
        step_dir = self.ckpt_root / job_id / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if fault_mode == "uncommitted_manifest_crash":
            manifest_file = step_dir / "_COMMITTED_MANIFEST.json"
            if manifest_file.exists():
                manifest_file.unlink()
            # Also drop rank_3.shard to mimic partial flush crash
            r3 = step_dir / "rank_3.shard"
            if r3.exists():
                r3.unlink()
            detail = f"Removed _COMMITTED_MANIFEST.json and rank_3.shard in step_{step:04d} (simulated mid-flush crash)."
        else:
            target_shard = step_dir / "rank_2.shard"
            if target_shard.is_file():
                orig = target_shard.read_bytes()
                torn = orig[: max(16, len(orig) // 2)] + b"\x00\xffTORN_CRASH_TAIL"
            else:
                torn = b'{"corrupt_partial_shard": true'
            target_shard.write_bytes(torn)
            detail = f"Truncated and corrupted byte stream of step_{step:04d}/rank_2.shard ({len(torn)} B torn write)."

        self._log_event(
            job_id=job_id,
            step=step,
            worker_id="chaos-injector",
            lease_epoch=self.get_active_lease(job_id)["lease_epoch"],
            event_type="FAULT_INJECTED",
            status="CORRUPTED_ON_DISK",
            detail=detail,
        )
        return {"job_id": job_id, "step": step, "fault_mode": fault_mode, "detail": detail}

    def inspect_and_quarantine(self, job_id: str) -> Dict[str, Any]:
        """Verify all checkpoint steps from newest to oldest, quarantining torn/incomplete steps."""
        t0 = time.perf_counter()
        job_dir = self.ckpt_root / job_id
        quarantine_root = job_dir / "quarantine"
        if not job_dir.is_dir():
            return {
                "highest_verified_step": 0,
                "verified_manifest": None,
                "verified_rank_states": None,
                "quarantined_steps": [],
                "verify_ms": 0.0,
            }

        step_dirs = sorted(
            [p for p in job_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda p: p.name,
            reverse=True,
        )

        quarantined: List[Dict[str, Any]] = []
        highest_step = 0
        verified_manifest: Optional[Dict[str, Any]] = None
        verified_states: Optional[List[Dict[str, Any]]] = None

        for sdir in step_dirs:
            try:
                step_num = int(sdir.name.split("_")[1])
            except Exception:
                continue

            manifest_path = sdir / "_COMMITTED_MANIFEST.json"
            corrupt_reason: Optional[str] = None
            loaded_states: List[Dict[str, Any]] = []
            manifest_data: Optional[Dict[str, Any]] = None

            if not manifest_path.is_file():
                corrupt_reason = "MISSING_COMMITTED_MANIFEST"
            else:
                try:
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    shards_spec = manifest_data.get("shards") or {}
                    if len(shards_spec) != self.NUM_RANKS:
                        corrupt_reason = f"INCOMPLETE_SHARD_COUNT({len(shards_spec)}/{self.NUM_RANKS})"
                    else:
                        for r in range(self.NUM_RANKS):
                            sname = f"rank_{r}.shard"
                            spath = sdir / sname
                            if not spath.is_file():
                                corrupt_reason = f"MISSING_SHARD({sname})"
                                break
                            raw = spath.read_bytes()
                            actual_sha = _sha256_bytes(raw)
                            expected_sha = shards_spec[sname]["sha256"]
                            expected_len = int(shards_spec[sname]["byte_size"])
                            if len(raw) != expected_len or actual_sha != expected_sha:
                                corrupt_reason = (
                                    f"TORN_SHARD_SHA256_MISMATCH({sname}: "
                                    f"actual={actual_sha[:12]}.. vs expected={expected_sha[:12]}..)"
                                )
                                break
                            parsed_shard = json.loads(raw.decode("utf-8"))
                            loaded_states.append(parsed_shard["state"])
                except Exception as exc:
                    corrupt_reason = f"CORRUPT_MANIFEST_JSON({exc})"

            if corrupt_reason is not None:
                quarantine_root.mkdir(parents=True, exist_ok=True)
                q_dest = quarantine_root / f"{sdir.name}_{int(time.time() * 1000) % 100000}"
                shutil.move(str(sdir), str(q_dest))
                quarantined.append(
                    {
                        "step": step_num,
                        "reason": corrupt_reason,
                        "quarantined_to": str(q_dest),
                    }
                )
                self._log_event(
                    job_id=job_id,
                    step=step_num,
                    worker_id="recovery-verifier",
                    lease_epoch=self.get_active_lease(job_id)["lease_epoch"],
                    event_type="SHARD_QUARANTINED",
                    status="QUARANTINED",
                    detail=f"Quarantined {sdir.name} -> {corrupt_reason}",
                )
            elif verified_manifest is None and manifest_data is not None:
                highest_step = step_num
                verified_manifest = manifest_data
                verified_states = sorted(loaded_states, key=lambda x: int(x["rank"]))

        verify_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "highest_verified_step": highest_step,
            "verified_manifest": verified_manifest,
            "verified_rank_states": verified_states,
            "quarantined_steps": quarantined,
            "verify_ms": verify_ms,
        }

    def run_uninterrupted_reference(self, total_steps: int = 12) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Compute clean uninterrupted reference trajectory across 1..total_steps."""
        states = self._initial_rank_states()
        losses: List[float] = []
        for s in range(1, total_steps + 1):
            states, _, loss = self._advance_one_step(s, states)
            losses.append(loss)
        return states, losses

    def run_fault_and_recovery_drill(
        self,
        job_id: str = "hyperpod-llama-sft-job",
        fault_mode: str = "torn_rank_shard",
        total_steps: int = 12,
        crash_at_step: int = 9,
        checkpoint_every: int = 3,
    ) -> Dict[str, Any]:
        """Execute a full checkpointed run, inject a real disk/zombie fault, fence stale worker, and verify bit-exact recovery."""
        t_all = time.perf_counter()
        # Clean previous state for this job_id so drill is self-contained and repeatable
        job_dir = self.ckpt_root / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        with self._connect() as conn:
            conn.execute("DELETE FROM checkpoint_leases WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM checkpoint_events WHERE job_id = ?", (job_id,))

        # 1. Worker-1 acquires epoch=1 and trains up to crash_at_step
        lease_1 = self.acquire_lease(job_id=job_id, worker_id="worker-node-01")
        epoch_1 = lease_1["lease_epoch"]
        states = self._initial_rank_states()
        loss_history: List[Dict[str, Any]] = []
        commits_phase1: List[Dict[str, Any]] = []

        for s in range(1, crash_at_step + 1):
            states, cursor, loss = self._advance_one_step(s, states)
            loss_history.append({"step": s, "worker": "worker-node-01", "epoch": epoch_1, "loss": loss})
            if s % checkpoint_every == 0:
                c_info = self.commit_checkpoint(
                    job_id=job_id,
                    step=s,
                    worker_id="worker-node-01",
                    lease_epoch=epoch_1,
                    rank_states=states,
                    dataloader_cursor=cursor,
                    loss=loss,
                )
                commits_phase1.append(c_info)

        # 2. Inject requested fault (torn shard, missing manifest, or zombie split-brain attempt)
        fault_report: Optional[Dict[str, Any]] = None
        if fault_mode in ("torn_rank_shard", "uncommitted_manifest_crash"):
            fault_report = self.inject_disk_fault(job_id=job_id, step=crash_at_step, fault_mode=fault_mode)

        # 3. Recovery Worker-2 acquires monotonic epoch=2 (immediately fencing Worker-1!)
        lease_2 = self.acquire_lease(job_id=job_id, worker_id="worker-node-02-recovery")
        epoch_2 = lease_2["lease_epoch"]

        # 4. Preempted zombie Worker-1 (still holding stale epoch=1) attempts split-brain write -> MUST BE BLOCKED!
        zombie_blocked = False
        zombie_error_msg = ""
        try:
            corrupt_states = self._initial_rank_states()
            self.commit_checkpoint(
                job_id=job_id,
                step=crash_at_step,
                worker_id="worker-node-01",
                lease_epoch=epoch_1,
                rank_states=corrupt_states,
                dataloader_cursor={"completed_step": crash_at_step, "token_offset": 0, "shard_index": 0},
                loss=99.99,
            )
        except FencedWriteRejectedError as exc:
            zombie_blocked = True
            zombie_error_msg = str(exc)

        # 5. Inspect disk checkpoints, quarantine torn shards, and restore highest verified step
        t_rec = time.perf_counter()
        inspection = self.inspect_and_quarantine(job_id=job_id)
        restored_step = int(inspection["highest_verified_step"])
        if inspection["verified_rank_states"] is not None:
            resumed_states = inspection["verified_rank_states"]
            restored_cursor = inspection["verified_manifest"]["dataloader_cursor"]
        else:
            resumed_states = self._initial_rank_states()
            restored_cursor = {"completed_step": 0, "token_offset": 0, "shard_index": 0}

        # 6. Resume deterministically from restored_step + 1 .. total_steps under epoch=2
        commits_phase2: List[Dict[str, Any]] = []
        for s in range(restored_step + 1, total_steps + 1):
            resumed_states, cursor, loss = self._advance_one_step(s, resumed_states)
            loss_history.append({"step": s, "worker": "worker-node-02-recovery", "epoch": epoch_2, "loss": loss})
            if s % checkpoint_every == 0 or s == total_steps:
                c_info = self.commit_checkpoint(
                    job_id=job_id,
                    step=s,
                    worker_id="worker-node-02-recovery",
                    lease_epoch=epoch_2,
                    rank_states=resumed_states,
                    dataloader_cursor=cursor,
                    loss=loss,
                )
                commits_phase2.append(c_info)

        recovery_ms = round((time.perf_counter() - t_rec) * 1000.0, 3)

        # 7. Prove bit-exact parameter parity against an uninterrupted 1..total_steps reference run
        ref_states, ref_losses = self.run_uninterrupted_reference(total_steps=total_steps)
        max_abs_diff = 0.0
        for r in range(self.NUM_RANKS):
            for w_rec, w_ref in zip(resumed_states[r]["weights"], ref_states[r]["weights"]):
                max_abs_diff = max(max_abs_diff, abs(float(w_rec) - float(w_ref)))

        total_drill_ms = round((time.perf_counter() - t_all) * 1000.0, 3)
        return {
            "job_id": job_id,
            "fault_mode": fault_mode,
            "total_steps": total_steps,
            "crash_at_step": crash_at_step,
            "restored_from_step": restored_step,
            "restored_dataloader_cursor": restored_cursor,
            "stale_worker_epoch": epoch_1,
            "active_recovery_epoch": epoch_2,
            "zombie_write_blocked": zombie_blocked,
            "zombie_rejection_reason": zombie_error_msg,
            "quarantined_steps": inspection["quarantined_steps"],
            "shard_verify_ms": inspection["verify_ms"],
            "recovery_and_resume_ms": recovery_ms,
            "total_drill_ms": total_drill_ms,
            "bit_exact_parity": max_abs_diff == 0.0,
            "max_abs_weight_diff": max_abs_diff,
            "final_loss": ref_losses[-1] if ref_losses else 0.0,
            "commits_before_crash": len(commits_phase1),
            "commits_after_recovery": len(commits_phase2),
            "loss_timeline": loss_history,
            "events": self.list_events(job_id=job_id, limit=25),
        }

    def list_events(self, job_id: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if job_id:
                rows = conn.execute(
                    "SELECT * FROM checkpoint_events WHERE job_id = ? ORDER BY event_id DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM checkpoint_events ORDER BY event_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]
