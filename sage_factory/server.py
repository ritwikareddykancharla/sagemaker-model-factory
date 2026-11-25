"""Local HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for SageFactory."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .eval_registry import CANDIDATE_PROFILES, BedrockEvalRegistry
from .fenced_checkpoint import FencedCheckpointEngine
from .pipeline_dag import PRESET_RECIPES, PipelineRecipe, compile_pipeline_dag, recipe_to_python_sdk
from .step_cache import StepCacheEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


class SageFactoryControlPlane:
    """Coordinates DAG compilation, SHA-256 step cache, fenced checkpoint recovery, and Bedrock model registry."""

    def __init__(self, runtime_dir: Optional[Path] = None) -> None:
        base_dir = runtime_dir or Path(os.environ.get("SAGEFACTORY_RUNTIME_DIR", "/tmp/sagefactory-runtime"))
        base_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = base_dir
        self.db_path = base_dir / "sagefactory_state.db"

        self.cache_engine = StepCacheEngine(runtime_dir=base_dir, db_path=self.db_path)
        self.checkpoint_engine = FencedCheckpointEngine(runtime_dir=base_dir, db_path=self.db_path)
        self.eval_registry = BedrockEvalRegistry(db_path=self.db_path)
        self._warm_seed_state()

    def _warm_seed_state(self) -> None:
        """Seed initial cached pipeline run, checkpoint recovery state, and baseline registry entry."""
        if not self.cache_engine.list_cache_entries(limit=1):
            default_recipe = PRESET_RECIPES["nova-sft-rag-prod"]
            run_out = self.cache_engine.execute_pipeline(default_recipe)
            ds_digest = run_out["steps"][0]["output_digest"][:24]
            self.eval_registry.seed_baseline_entry_if_empty(
                lineage_context={
                    "dataset_sha256": ds_digest,
                    "pipeline_merkle_root": run_out["pipeline_merkle_root"][:24],
                    "checkpoint_epoch_step": "epoch=2 / step=0012 (4-rank verified)",
                }
            )
        else:
            self.eval_registry.seed_baseline_entry_if_empty()

        if not self.checkpoint_engine.list_events(limit=1):
            self.checkpoint_engine.run_fault_and_recovery_drill(
                job_id="hyperpod-nova-sft-job",
                fault_mode="torn_rank_shard",
                total_steps=12,
                crash_at_step=9,
                checkpoint_every=3,
            )

    def load_benchmarks(self) -> Dict[str, Any]:
        if BENCHMARK_JSON_PATH.is_file():
            try:
                return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"service": "sagemaker-model-factory", "headline_kpis": {}}

    def get_state_snapshot(self) -> Dict[str, Any]:
        presets_meta = []
        for pid, rec in PRESET_RECIPES.items():
            presets_meta.append(
                {
                    "pipeline_id": pid,
                    "title": rec.title,
                    "description": rec.description,
                    "model_family": rec.model_family,
                    "is_fault_probe": pid.startswith("bug-"),
                    "canonical_dag": rec.to_dict(),
                    "python_sdk_code": recipe_to_python_sdk(rec),
                }
            )
        return {
            "service": "sagemaker-model-factory",
            "presets": presets_meta,
            "candidate_profiles": list(CANDIDATE_PROFILES.values()),
            "cache_entries": self.cache_engine.list_cache_entries(limit=24),
            "recent_lineage": self.cache_engine.list_recent_lineage(limit=24),
            "checkpoint_events": self.checkpoint_engine.list_events(limit=20),
            "registry_versions": self.eval_registry.list_versions(limit=16),
            "benchmarks": self.load_benchmarks(),
        }

    def execute_pipeline_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        preset_id = str(body.get("preset_id") or "").strip()
        spec_payload: Any = body.get("pipeline_spec")

        if spec_payload:
            recipe_obj = (
                compile_pipeline_dag(spec_payload)
                if isinstance(spec_payload, (str, dict))
                else compile_pipeline_dag( PRESET_RECIPES["nova-sft-rag-prod"] )
            )
            if not recipe_obj["valid"]:
                return {"executed": False, "compile_result": recipe_obj}
            recipe = PipelineRecipe.from_dict(recipe_obj["canonical_dag"])
        else:
            recipe = PipelineRecipe.from_dict(
                PRESET_RECIPES.get(preset_id, PRESET_RECIPES["nova-sft-rag-prod"]).to_dict()
            )

        # Apply optional interactive hyperparameter overrides for live step-cache hit/miss testing
        overrides = body.get("overrides") or {}
        for s in recipe.steps:
            if s.step_kind == "DataPrepStep" and "dataset_version" in overrides:
                s.hyperparams["dataset_version"] = str(overrides["dataset_version"])
            if s.step_kind == "FineTuneStep" and "learning_rate" in overrides:
                s.hyperparams["learning_rate"] = float(overrides["learning_rate"])
            if s.step_kind == "BedrockEvalStep" and "rubric_version" in overrides:
                s.hyperparams["rubric_version"] = str(overrides["rubric_version"])
            if s.step_kind == "ConditionStep" and "min_quality_score" in overrides:
                val = float(overrides["min_quality_score"])
                s.hyperparams["min_quality_score"] = val
                if s.condition_rules is not None:
                    s.condition_rules["min_quality_score"] = val

        use_live_bedrock = bool(body.get("use_live_bedrock", False))

        def _live_bedrock_step(step_spec: Any, _up: Dict[str, Any]) -> Dict[str, Any]:
            cand_id = str(step_spec.hyperparams.get("candidate_profile", "cand-nova-rag-sft-v2"))
            ev = self.eval_registry.evaluate_and_register(
                candidate_id=cand_id,
                min_quality_score=float(overrides.get("min_quality_score", 0.82)),
                target_stage="Canary_10Pct",
            )
            return {
                "quality_score": ev["quality_score"],
                "baseline_score": ev["baseline_score"],
                "p95_latency_ms": ev["p95_latency_ms"],
                "safety_passed": ev["safety_passed"],
                "bedrock_model": ev["bedrock_model"],
                "registry_version_id": ev["version_id"],
                "total_tokens": ev["total_tokens"],
            }

        force_steps = body.get("force_refresh_steps") or []
        return self.cache_engine.execute_pipeline(
            recipe_input=recipe,
            force_refresh_steps=force_steps,
            bedrock_eval_callback=_live_bedrock_step if use_live_bedrock else None,
        )

    def reset(self) -> None:
        self.cache_engine.reset()
        self.checkpoint_engine.reset()
        self.eval_registry.reset()
        self._warm_seed_state()


CONTROL_PLANE = SageFactoryControlPlane()


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Shared API router for local HTTP server and AWS Lambda behind CloudFront `/api/*`."""
    try:
        if method == "GET" and path == "/api/health":
            cache_entries = CONTROL_PLANE.cache_engine.list_cache_entries(limit=50)
            versions = CONTROL_PLANE.eval_registry.list_versions(limit=50)
            return 200, {
                "status": "ok",
                "service": "sagemaker-model-factory",
                "delivery": os.environ.get("SAGEFACTORY_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "bedrock_model": os.environ.get("SAGEFACTORY_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "preset_recipes_count": len(PRESET_RECIPES),
                "cached_steps_count": len(cache_entries),
                "registered_models_count": len(versions),
            }

        if method == "GET" and path == "/api/state":
            snap = CONTROL_PLANE.get_state_snapshot()
            snap["request_id"] = request_id
            return 200, snap

        if method == "GET" and path == "/api/benchmarks":
            bench = CONTROL_PLANE.load_benchmarks()
            bench["request_id"] = request_id
            return 200, bench

        if method == "POST" and path == "/api/pipeline/compile":
            preset_id = str(body.get("preset_id") or "").strip()
            spec = body.get("pipeline_spec")
            if spec:
                compiled = compile_pipeline_dag(spec)
            elif preset_id and preset_id in PRESET_RECIPES:
                compiled = compile_pipeline_dag(PRESET_RECIPES[preset_id])
            else:
                compiled = compile_pipeline_dag(PRESET_RECIPES["nova-sft-rag-prod"])
            compiled["request_id"] = request_id
            return 200, compiled

        if method == "POST" and path == "/api/pipeline/execute":
            out = CONTROL_PLANE.execute_pipeline_request(body)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/checkpoint/recover":
            job_id = str(body.get("job_id") or "hyperpod-nova-sft-job").strip()
            fault_mode = str(body.get("fault_mode") or "torn_rank_shard").strip()
            total_steps = max(6, min(24, int(body.get("total_steps", 12))))
            crash_at_step = max(3, min(total_steps - 1, int(body.get("crash_at_step", 9))))
            out = CONTROL_PLANE.checkpoint_engine.run_fault_and_recovery_drill(
                job_id=job_id,
                fault_mode=fault_mode,
                total_steps=total_steps,
                crash_at_step=crash_at_step,
                checkpoint_every=3,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/registry/evaluate":
            candidate_id = str(body.get("candidate_id") or "cand-nova-rag-sft-v2").strip()
            min_q = float(body.get("min_quality_score", 0.84))
            max_lat = float(body.get("max_p95_latency_ms", 2600.0))
            target_stage = body.get("target_stage")
            custom_prompt = body.get("custom_system_prompt")

            recent_lineage = CONTROL_PLANE.cache_engine.list_recent_lineage(limit=6)
            lineage_ctx: Dict[str, Any] = {}
            if recent_lineage:
                lineage_ctx["dataset_sha256"] = recent_lineage[-1]["output_digest"][:24]
                lineage_ctx["pipeline_merkle_root"] = recent_lineage[0]["merkle_key"][:24]
                lineage_ctx["checkpoint_epoch_step"] = "epoch=2 / step=0012 (4-rank verified)"

            out = CONTROL_PLANE.eval_registry.evaluate_and_register(
                candidate_id=candidate_id,
                min_quality_score=min_q,
                max_p95_latency_ms=max_lat,
                target_stage=str(target_stage) if target_stage else None,
                custom_system_prompt=str(custom_prompt) if custom_prompt else None,
                lineage_context=lineage_ctx,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            CONTROL_PLANE.reset()
            return 200, {
                "status": "reset_complete",
                "service": "sagemaker-model-factory",
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalSageFactoryHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalSageFactoryHandler)
    print(f"SageFactory MLOps Control Plane running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
