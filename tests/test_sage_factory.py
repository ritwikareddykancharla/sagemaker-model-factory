"""Unit test suite for SageFactory MLOps Pipeline DAG, Step Cache, Fenced Checkpoints & Eval Registry."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from sage_factory.eval_registry import BedrockEvalRegistry
from sage_factory.fenced_checkpoint import FencedCheckpointEngine, FencedWriteRejectedError
from sage_factory.pipeline_dag import (
    ArtifactContract,
    ComputeResources,
    DataPrepStep,
    FineTuneStep,
    PRESET_RECIPES,
    PipelineRecipe,
    compile_pipeline_dag,
    parse_python_sdk_snippet,
    pipeline_step,
    recipe_to_python_sdk,
)
from sage_factory.server import SageFactoryControlPlane, lambda_handler, route_api_request
from sage_factory.step_cache import StepCacheEngine, compute_step_merkle_key, normalize_code_ast


class TestSageFactory(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_offline = os.environ.get("SAGEFACTORY_OFFLINE_TEST")
        os.environ["SAGEFACTORY_OFFLINE_TEST"] = "1"
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="sagefactory-test-"))
        self.db_path = self.tmp_dir / "test_state.db"

    def tearDown(self) -> None:
        if self._prev_offline is None:
            os.environ.pop("SAGEFACTORY_OFFLINE_TEST", None)
        else:
            os.environ["SAGEFACTORY_OFFLINE_TEST"] = self._prev_offline

    def test_01_valid_diamond_dag_waves_and_hyperpod_yaml(self) -> None:
        res = compile_pipeline_dag(PRESET_RECIPES["nova-sft-rag-prod"])
        self.assertTrue(res["valid"])
        self.assertEqual(res["step_count"], 6)
        self.assertEqual(len(res["waves"]), 5)
        # Wave 2 runs latency_profile and bedrock_eval concurrently
        self.assertEqual(res["waves"][2]["concurrency"], 2)
        self.assertIn("kind: PyTorchJob", res["hyperpod_manifest_yaml"])
        self.assertIn("FI_PROVIDER", res["hyperpod_manifest_yaml"])

    def test_02_verl_grpo_lowers_to_eks_rayjob(self) -> None:
        res = compile_pipeline_dag(PRESET_RECIPES["verl-grpo-reasoning"])
        self.assertTrue(res["valid"])
        self.assertIn("kind: RayJob", res["hyperpod_manifest_yaml"])
        self.assertIn("verl.trainer.main_grpo", res["hyperpod_manifest_yaml"])

    def test_03_bidirectional_python_sdk_ast_roundtrip(self) -> None:
        orig = PRESET_RECIPES["nova-sft-rag-prod"]
        sdk_code = recipe_to_python_sdk(orig)
        parsed = parse_python_sdk_snippet(sdk_code)
        self.assertEqual(parsed.pipeline_id, orig.pipeline_id)
        self.assertEqual(len(parsed.steps), len(orig.steps))
        compiled = compile_pipeline_dag(sdk_code)
        self.assertTrue(compiled["valid"])
        self.assertEqual(compiled["dag_sha256"], compile_pipeline_dag(orig)["dag_sha256"])

    def test_04_decorator_pipeline_step_builds_typed_spec(self) -> None:
        @pipeline_step(
            step_id="custom_clean",
            step_kind="DataPrepStep",
            outputs={"out": ArtifactContract("out", "dataset", "jsonl", "bfloat16", ["prompt", "response"])},
            hyperparams={"min_tokens": 12},
        )
        def clean_fn(rows: list) -> list:
            """Docstring ignored by AST normalizer."""
            return [r for r in rows if r]

        rec = PipelineRecipe("dec-pipe", "Decorator Test", "Typed decorator", "nova-lite", [clean_fn])
        compiled = compile_pipeline_dag(rec)
        self.assertTrue(compiled["valid"])
        self.assertEqual(compiled["step_count"], 1)

    def test_05_compile_catches_cycle_missing_port_schema_and_topology_bugs(self) -> None:
        c_res = compile_pipeline_dag(PRESET_RECIPES["bug-cyclic-feedback"])
        self.assertFalse(c_res["valid"])
        self.assertTrue(any(e["rule_id"] == "DAG-CYCLE-01" for e in c_res["errors"]))

        m_res = compile_pipeline_dag(PRESET_RECIPES["bug-missing-artifact"])
        self.assertFalse(m_res["valid"])
        self.assertTrue(any(e["rule_id"] == "DAG-ARTIFACT-02" for e in m_res["errors"]))

        s_res = compile_pipeline_dag(PRESET_RECIPES["bug-schema-dtype-mismatch"])
        self.assertFalse(s_res["valid"])
        self.assertTrue(any(e["rule_id"] == "DAG-SCHEMA-03" for e in s_res["errors"]))

        t_res = compile_pipeline_dag(PRESET_RECIPES["bug-topology-hbm-overflow"])
        self.assertFalse(t_res["valid"])
        self.assertTrue(any(e["rule_id"] == "DAG-TOPOLOGY-04" for e in t_res["errors"]))

    def test_06_ast_normalization_ignores_comments_and_docstrings(self) -> None:
        code_a = 'def fn(x):\n    """Some docstring."""\n    # comment\n    return x + 1\n'
        code_b = "def fn(x):\n    return x + 1"
        code_c = "def fn(x):\n    return x + 2"
        self.assertEqual(normalize_code_ast(code_a), normalize_code_ast(code_b))
        self.assertNotEqual(normalize_code_ast(code_a), normalize_code_ast(code_c))

        k1 = compute_step_merkle_key("DataPrepStep", code_a, {"b": 2, "a": 1}, {"in": "abc"})
        k2 = compute_step_merkle_key("DataPrepStep", code_b, {"a": 1, "b": 2}, {"in": "abc"})
        self.assertEqual(k1["merkle_key"], k2["merkle_key"])

    def test_07_step_cache_hits_upstream_and_cascades_invalidation_on_data_change(self) -> None:
        engine = StepCacheEngine(runtime_dir=self.tmp_dir, db_path=self.db_path)
        recipe = PipelineRecipe.from_dict(PRESET_RECIPES["nova-sft-rag-prod"].to_dict())

        # Cold run: 0 hits, 6 misses
        run1 = engine.execute_pipeline(recipe)
        self.assertEqual(run1["cache_hits"], 0)
        self.assertEqual(run1["cache_misses"], 6)

        # Mutate only downstream ConditionStep threshold: 4 upstream steps MUST hit cache in < 1 ms
        recipe_tweaked = PipelineRecipe.from_dict(recipe.to_dict())
        recipe_tweaked.steps[4].hyperparams["min_quality_score"] = 0.88
        recipe_tweaked.steps[4].condition_rules["min_quality_score"] = 0.88
        run2 = engine.execute_pipeline(recipe_tweaked)
        self.assertEqual(run2["cache_hits"], 4)
        self.assertEqual(run2["cache_misses"], 2)
        self.assertLess(run2["steps"][0]["duration_ms"], 5.0)

        # Mutate root DataPrepStep dataset_version: cascades invalidation across all 6 steps
        recipe_new_data = PipelineRecipe.from_dict(recipe_tweaked.to_dict())
        recipe_new_data.steps[0].hyperparams["dataset_version"] = "v3.0-fresh"
        run3 = engine.execute_pipeline(recipe_new_data)
        self.assertEqual(run3["cache_hits"], 0)
        self.assertEqual(run3["cache_misses"], 6)

    def test_08_fenced_checkpoint_blocks_stale_epoch_zombie_worker(self) -> None:
        ckpt = FencedCheckpointEngine(runtime_dir=self.tmp_dir, db_path=self.db_path)
        l1 = ckpt.acquire_lease("job-fence", "worker-1")
        self.assertEqual(l1["lease_epoch"], 1)
        states = ckpt._initial_rank_states()
        states, cursor, loss = ckpt._advance_one_step(1, states)
        ckpt.commit_checkpoint("job-fence", 1, "worker-1", 1, states, cursor, loss)

        # Worker-2 preempts and acquires epoch=2
        l2 = ckpt.acquire_lease("job-fence", "worker-2")
        self.assertEqual(l2["lease_epoch"], 2)

        # Delayed Worker-1 tries to write with stale epoch=1 -> must raise FencedWriteRejectedError
        with self.assertRaises(FencedWriteRejectedError):
            ckpt.commit_checkpoint("job-fence", 2, "worker-1", 1, states, cursor, loss)

    def test_09_torn_shard_quarantine_and_bit_exact_resume_parity(self) -> None:
        ckpt = FencedCheckpointEngine(runtime_dir=self.tmp_dir, db_path=self.db_path)
        drill = ckpt.run_fault_and_recovery_drill(
            job_id="job-torn",
            fault_mode="torn_rank_shard",
            total_steps=12,
            crash_at_step=9,
            checkpoint_every=3,
        )
        self.assertTrue(drill["zombie_write_blocked"])
        self.assertEqual(drill["restored_from_step"], 6)
        self.assertEqual(len(drill["quarantined_steps"]), 1)
        self.assertEqual(drill["quarantined_steps"][0]["step"], 9)
        self.assertTrue(drill["bit_exact_parity"])
        self.assertEqual(drill["max_abs_weight_diff"], 0.0)

    def test_10_uncommitted_manifest_crash_recovery(self) -> None:
        ckpt = FencedCheckpointEngine(runtime_dir=self.tmp_dir, db_path=self.db_path)
        drill = ckpt.run_fault_and_recovery_drill(
            job_id="job-uncommitted",
            fault_mode="uncommitted_manifest_crash",
            total_steps=12,
            crash_at_step=9,
            checkpoint_every=3,
        )
        self.assertEqual(drill["restored_from_step"], 6)
        self.assertIn("MISSING_COMMITTED_MANIFEST", drill["quarantined_steps"][0]["reason"])
        self.assertTrue(drill["bit_exact_parity"])

    def test_11_eval_registry_promotes_calibrated_and_rejects_regression(self) -> None:
        reg = BedrockEvalRegistry(db_path=self.db_path)
        good = reg.evaluate_and_register(candidate_id="cand-nova-rag-sft-v2", target_stage="Production")
        self.assertTrue(good["condition_gate_passed"])
        self.assertEqual(good["deployment_stage"], "Production")
        self.assertEqual(good["traffic_weight_pct"], 100)

        bad = reg.evaluate_and_register(candidate_id="cand-nova-uncalibrated-v3", target_stage="Production")
        self.assertFalse(bad["condition_gate_passed"])
        self.assertEqual(bad["deployment_stage"], "Rejected")
        self.assertEqual(bad["traffic_weight_pct"], 0)

    def test_12_lambda_handler_and_api_routes(self) -> None:
        status, health = route_api_request("GET", "/api/health", {}, {})
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

        resp = lambda_handler(
            {
                "rawPath": "/api/pipeline/compile",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"preset_id": "nova-sft-rag-prod"}),
                "isBase64Encoded": False,
            }
        )
        self.assertEqual(resp["statusCode"], 200)
        payload = json.loads(resp["body"])
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["step_count"], 6)


if __name__ == "__main__":
    unittest.main()
