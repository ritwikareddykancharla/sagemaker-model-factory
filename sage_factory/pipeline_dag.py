"""Bidirectional Typed Python Pipeline SDK & Multi-Backend Manifest Compiler.

Compiles declarative Python SDK recipes and canonical JSON DAGs via Kahn's
topological wave algorithm, detects cycles, missing upstream artifacts, cross-step
schema/dtype mismatches, and accelerator world-size/HBM violations at compile time,
and lowers valid workflows to Serverless Wave Plans and Kubernetes/HyperPod
PyTorchJob & RayJob YAML specifications.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArtifactContract:
    name: str
    artifact_type: str  # "dataset", "weights", "profile", "eval_report", "model_package"
    storage_format: str  # "parquet", "jsonl", "safetensors", "json"
    dtype: str  # "bfloat16", "float32", "int64", "json"
    required_columns: List[str] = field(default_factory=list)
    vocab_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "storage_format": self.storage_format,
            "dtype": self.dtype,
            "required_columns": list(self.required_columns),
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactContract":
        return cls(
            name=str(data.get("name", "output")),
            artifact_type=str(data.get("artifact_type", "dataset")),
            storage_format=str(data.get("storage_format", "jsonl")),
            dtype=str(data.get("dtype", "json")),
            required_columns=[str(c) for c in (data.get("required_columns") or [])],
            vocab_size=int(data["vocab_size"]) if data.get("vocab_size") is not None else None,
        )


@dataclass(frozen=True)
class ComputeResources:
    backend: str = "serverless_lambda"  # "hyperpod_pytorch", "eks_ray_verl", "serverless_lambda", "bedrock_eval"
    instance_type: str = "lambda.python3.12"
    num_accelerators: int = 0
    tp_degree: int = 1
    pp_degree: int = 1
    dp_degree: int = 1
    per_rank_mem_gb: float = 0.0
    instance_hbm_gb: float = 80.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputeResources":
        if not data:
            return cls()
        return cls(
            backend=str(data.get("backend", "serverless_lambda")),
            instance_type=str(data.get("instance_type", "lambda.python3.12")),
            num_accelerators=int(data.get("num_accelerators", 0)),
            tp_degree=int(data.get("tp_degree", 1)),
            pp_degree=int(data.get("pp_degree", 1)),
            dp_degree=int(data.get("dp_degree", 1)),
            per_rank_mem_gb=float(data.get("per_rank_mem_gb", 0.0)),
            instance_hbm_gb=float(data.get("instance_hbm_gb", 80.0)),
        )


@dataclass
class PipelineStepSpec:
    step_id: str
    step_kind: str
    depends_on: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)  # local_port -> "upstream_step.output_name"
    expected_inputs: Dict[str, ArtifactContract] = field(default_factory=dict)
    outputs: Dict[str, ArtifactContract] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    code_body: str = ""
    compute: ComputeResources = field(default_factory=ComputeResources)
    condition_rules: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_kind": self.step_kind,
            "depends_on": list(self.depends_on),
            "inputs": dict(self.inputs),
            "expected_inputs": {k: v.to_dict() for k, v in self.expected_inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "hyperparams": dict(self.hyperparams),
            "code_body": self.code_body,
            "compute": self.compute.to_dict(),
            "condition_rules": dict(self.condition_rules) if self.condition_rules else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStepSpec":
        expected_in = {
            str(k): ArtifactContract.from_dict(v)
            for k, v in (data.get("expected_inputs") or {}).items()
        }
        outs = {
            str(k): ArtifactContract.from_dict(v)
            for k, v in (data.get("outputs") or {}).items()
        }
        return cls(
            step_id=str(data["step_id"]),
            step_kind=str(data.get("step_kind", "DataPrepStep")),
            depends_on=[str(d) for d in (data.get("depends_on") or [])],
            inputs={str(k): str(v) for k, v in (data.get("inputs") or {}).items()},
            expected_inputs=expected_in,
            outputs=outs,
            hyperparams=dict(data.get("hyperparams") or {}),
            code_body=str(data.get("code_body") or DEFAULT_STEP_CODE.get(str(data.get("step_kind", "")), "def run(ctx): return ctx")),
            compute=ComputeResources.from_dict(data.get("compute") or {}),
            condition_rules=dict(data["condition_rules"]) if data.get("condition_rules") else None,
        )


@dataclass
class PipelineRecipe:
    pipeline_id: str
    title: str
    description: str
    model_family: str
    steps: List[PipelineStepSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "title": self.title,
            "description": self.description,
            "model_family": self.model_family,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineRecipe":
        return cls(
            pipeline_id=str(data.get("pipeline_id", "custom-pipeline")),
            title=str(data.get("title", "Custom Model Factory Pipeline")),
            description=str(data.get("description", "Declarative MLOps lifecycle pipeline.")),
            model_family=str(data.get("model_family", "us.amazon.nova-lite-v1:0")),
            steps=[PipelineStepSpec.from_dict(s) for s in (data.get("steps") or [])],
        )


DEFAULT_STEP_CODE: Dict[str, str] = {
    "DataPrepStep": (
        "def prepare_corpus(records, dedup_window=64, min_tokens=8):\n"
        "    seen = set()\n"
        "    cleaned = []\n"
        "    for row in records:\n"
        "        key = row['prompt'].strip().lower()[:dedup_window]\n"
        "        if key not in seen and len(row['response'].split()) >= min_tokens:\n"
        "            seen.add(key)\n"
        "            cleaned.append(row)\n"
        "    return cleaned\n"
    ),
    "FineTuneStep": (
        "def train_sft_adapter(shards, learning_rate=2e-4, epochs=3, lora_rank=16):\n"
        "    weights = [0.0] * lora_rank\n"
        "    for epoch in range(epochs):\n"
        "        for idx, sample in enumerate(shards):\n"
        "            grad = (len(sample['response']) - len(sample['prompt'])) * 0.001\n"
        "            weights[idx % lora_rank] -= learning_rate * grad\n"
        "    return weights\n"
    ),
    "LatencyProfileStep": (
        "def profile_adapter_latency(weights, batch_size=8, sequence_len=1024):\n"
        "    norm = sum(abs(w) for w in weights)\n"
        "    est_p95_ms = round(180.0 + batch_size * 9.5 + norm * 0.05, 2)\n"
        "    return {'estimated_p95_ms': est_p95_ms, 'sequence_len': sequence_len}\n"
    ),
    "BedrockEvalStep": (
        "def evaluate_on_bedrock(weights, rubric_version='v2.1', temperature=0.1):\n"
        "    return {'model': 'us.amazon.nova-lite-v1:0', 'rubric': rubric_version, 'temp': temperature}\n"
    ),
    "ConditionStep": (
        "def check_promotion_gate(eval_report, min_quality=0.82, max_p95_ms=2400.0):\n"
        "    return eval_report['quality_score'] >= min_quality and eval_report['p95_latency_ms'] <= max_p95_ms\n"
    ),
    "RegisterModelStep": (
        "def register_canary_package(eval_report, stage='Canary_10Pct', traffic_pct=10):\n"
        "    return {'deployment_stage': stage, 'traffic_pct': traffic_pct}\n"
    ),
}


def pipeline_step(
    step_id: str,
    step_kind: str,
    depends_on: Optional[List[str]] = None,
    inputs: Optional[Dict[str, str]] = None,
    expected_inputs: Optional[Dict[str, ArtifactContract]] = None,
    outputs: Optional[Dict[str, ArtifactContract]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    compute: Optional[ComputeResources] = None,
    condition_rules: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], PipelineStepSpec]:
    """Typed Python SDK decorator turning a Python function into a validated PipelineStepSpec."""

    def _decorator(fn: Callable[..., Any]) -> PipelineStepSpec:
        try:
            src = inspect.getsource(fn)
        except OSError:
            src = DEFAULT_STEP_CODE.get(step_kind, f"def {fn.__name__}(ctx): return ctx")
        return PipelineStepSpec(
            step_id=step_id,
            step_kind=step_kind,
            depends_on=list(depends_on or []),
            inputs=dict(inputs or {}),
            expected_inputs=dict(expected_inputs or {}),
            outputs=dict(outputs or {}),
            hyperparams=dict(hyperparams or {}),
            code_body=src,
            compute=compute or ComputeResources(),
            condition_rules=dict(condition_rules) if condition_rules else None,
        )

    return _decorator


# Convenience typed SDK constructors matching SageMaker Pipelines ergonomics
def DataPrepStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "DataPrepStep", **kwargs})


def FineTuneStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "FineTuneStep", **kwargs})


def LatencyProfileStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "LatencyProfileStep", **kwargs})


def BedrockEvalStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "BedrockEvalStep", **kwargs})


def ConditionStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "ConditionStep", **kwargs})


def RegisterModelStep(step_id: str, **kwargs: Any) -> PipelineStepSpec:
    return PipelineStepSpec.from_dict({"step_id": step_id, "step_kind": "RegisterModelStep", **kwargs})


def recipe_to_python_sdk(recipe: PipelineRecipe) -> str:
    """Render a canonical PipelineRecipe into clean, editable Python SDK code."""
    lines: List[str] = [
        "from sage_factory.pipeline_dag import (",
        "    PipelineRecipe, DataPrepStep, FineTuneStep,",
        "    LatencyProfileStep, BedrockEvalStep, ConditionStep, RegisterModelStep,",
        ")",
        "",
        f"pipeline = PipelineRecipe(",
        f"    pipeline_id={recipe.pipeline_id!r},",
        f"    title={recipe.title!r},",
        f"    description={recipe.description!r},",
        f"    model_family={recipe.model_family!r},",
        "    steps=[",
    ]
    for s in recipe.steps:
        ctor = s.step_kind if s.step_kind in {
            "DataPrepStep",
            "FineTuneStep",
            "LatencyProfileStep",
            "BedrockEvalStep",
            "ConditionStep",
            "RegisterModelStep",
        } else "DataPrepStep"
        step_dict = s.to_dict()
        lines.append(f"        {ctor}(")
        lines.append(f"            step_id={s.step_id!r},")
        if s.depends_on:
            lines.append(f"            depends_on={s.depends_on!r},")
        if s.inputs:
            lines.append(f"            inputs={s.inputs!r},")
        if step_dict["expected_inputs"]:
            lines.append(f"            expected_inputs={step_dict['expected_inputs']!r},")
        if step_dict["outputs"]:
            lines.append(f"            outputs={step_dict['outputs']!r},")
        if s.hyperparams:
            lines.append(f"            hyperparams={s.hyperparams!r},")
        if s.compute.backend != "serverless_lambda" or s.compute.num_accelerators > 0:
            lines.append(f"            compute={step_dict['compute']!r},")
        if s.condition_rules:
            lines.append(f"            condition_rules={s.condition_rules!r},")
        lines.append("        ),")
    lines.append("    ],")
    lines.append(")")
    return "\n".join(lines) + "\n"


def parse_python_sdk_snippet(sdk_source: str) -> PipelineRecipe:
    """Safely parse a declarative Python SDK snippet via Python AST without executing arbitrary code."""
    tree = ast.parse(sdk_source, mode="exec")
    allowed_step_ctors = {
        "DataPrepStep",
        "FineTuneStep",
        "LatencyProfileStep",
        "BedrockEvalStep",
        "ConditionStep",
        "RegisterModelStep",
        "PipelineStepSpec",
    }

    def _eval_call_kwargs(call_node: ast.Call) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if call_node.args:
            out["step_id"] = ast.literal_eval(call_node.args[0])
        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            out[kw.arg] = ast.literal_eval(kw.value)
        return out

    recipe_call: Optional[ast.Call] = None
    step_vars: Dict[str, Dict[str, Any]] = {}

    for node in tree.body:
        value_node: Optional[ast.AST] = None
        target_name: Optional[str] = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.Expr):
            value_node = node.value

        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name):
            fn_name = value_node.func.id
            if fn_name in allowed_step_ctors and target_name:
                kw = _eval_call_kwargs(value_node)
                if fn_name != "PipelineStepSpec":
                    kw["step_kind"] = fn_name
                step_vars[target_name] = kw
            elif fn_name == "PipelineRecipe":
                recipe_call = value_node

    if recipe_call is None:
        raise ValueError("Python SDK snippet must define a PipelineRecipe(...) call.")

    recipe_data: Dict[str, Any] = {"steps": []}
    for kw in recipe_call.keywords:
        if kw.arg == "steps" and isinstance(kw.value, ast.List):
            parsed_steps: List[Dict[str, Any]] = []
            for elt in kw.value.elts:
                if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name) and elt.func.id in allowed_step_ctors:
                    s_kw = _eval_call_kwargs(elt)
                    if elt.func.id != "PipelineStepSpec":
                        s_kw["step_kind"] = elt.func.id
                    parsed_steps.append(s_kw)
                elif isinstance(elt, ast.Name) and elt.id in step_vars:
                    parsed_steps.append(step_vars[elt.id])
                else:
                    raise ValueError("Unsupported element inside PipelineRecipe(steps=[...]).")
            recipe_data["steps"] = parsed_steps
        elif kw.arg is not None:
            recipe_data[kw.arg] = ast.literal_eval(kw.value)

    return PipelineRecipe.from_dict(recipe_data)


def _render_hyperpod_yaml(recipe: PipelineRecipe, distributed_steps: List[PipelineStepSpec]) -> str:
    """Lower distributed training / RL steps into HyperPod PyTorchJob or EKS RayJob YAML manifests."""
    if not distributed_steps:
        return (
            "# Serverless-only pipeline: no distributed GPU/Trainium cluster manifest required.\n"
            f"# Pipeline: {recipe.pipeline_id}\n"
        )

    blocks: List[str] = []
    for step in distributed_steps:
        comp = step.compute
        if comp.backend == "eks_ray_verl":
            workers = max(1, comp.num_accelerators // 8)
            gpus_per_pod = min(8, max(1, comp.num_accelerators))
            blocks.append(
                "\n".join(
                    [
                        "apiVersion: ray.io/v1",
                        "kind: RayJob",
                        "metadata:",
                        f"  name: {recipe.pipeline_id}-{step.step_id}".lower().replace("_", "-"),
                        "  namespace: sagemaker-model-factory",
                        "  labels:",
                        f"    sagefactory.aws/pipeline: {recipe.pipeline_id}",
                        f"    sagefactory.aws/step: {step.step_id}",
                        "    sagefactory.aws/framework: verl-grpo",
                        "spec:",
                        f"  entrypoint: python3 -m verl.trainer.main_grpo --tp={comp.tp_degree} --pp={comp.pp_degree} --dp={comp.dp_degree}",
                        "  shutdownAfterJobFinishes: true",
                        "  rayClusterSpec:",
                        "    rayVersion: '2.38.0'",
                        "    headGroupSpec:",
                        "      rayStartParams:",
                        "        dashboard-host: '0.0.0.0'",
                        "      template:",
                        "        spec:",
                        "          containers:",
                        "            - name: ray-head",
                        "              image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.4.0-gpu-py311",
                        "    workerGroupSpecs:",
                        "      - groupName: hyperpod-gpu-group",
                        f"        replicas: {workers}",
                        "        template:",
                        "          metadata:",
                        "            annotations:",
                        f"              instance.sagefactory.aws/type: {comp.instance_type}",
                        "          spec:",
                        "            containers:",
                        "              - name: verl-worker",
                        "                resources:",
                        "                  limits:",
                        f"                    nvidia.com/gpu: {gpus_per_pod}",
                        f"                    vpc.amazonaws.com/efa: {max(1, gpus_per_pod // 2)}",
                    ]
                )
            )
        else:
            nodes = max(1, comp.num_accelerators // 8) if comp.num_accelerators >= 8 else 1
            nproc = 8 if comp.num_accelerators >= 8 else max(1, comp.num_accelerators)
            lr = step.hyperparams.get("learning_rate", 2e-4)
            epochs = step.hyperparams.get("epochs", 3)
            blocks.append(
                "\n".join(
                    [
                        "apiVersion: kubeflow.org/v1",
                        "kind: PyTorchJob",
                        "metadata:",
                        f"  name: {recipe.pipeline_id}-{step.step_id}".lower().replace("_", "-"),
                        "  namespace: sagemaker-hyperpod",
                        "  labels:",
                        f"    sagefactory.aws/pipeline: {recipe.pipeline_id}",
                        f"    sagefactory.aws/step: {step.step_id}",
                        f"    sagefactory.aws/instance-type: {comp.instance_type}",
                        "spec:",
                        "  elasticPolicy:",
                        f"    minReplicas: {nodes}",
                        f"    maxReplicas: {nodes}",
                        f"    nprocPerNode: {nproc}",
                        "  pytorchReplicaSpecs:",
                        "    Worker:",
                        f"      replicas: {nodes}",
                        "      restartPolicy: OnFailure",
                        "      template:",
                        "        spec:",
                        "          containers:",
                        "            - name: pytorch-fsdp",
                        "              image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.4.0-gpu-py311",
                        "              args:",
                        "                - torchrun",
                        f"                - --nproc_per_node={nproc}",
                        f"                - --nnodes={nodes}",
                        "                - /opt/ml/code/train_fsdp.py",
                        f"                - --tp-degree={comp.tp_degree}",
                        f"                - --pp-degree={comp.pp_degree}",
                        f"                - --dp-degree={comp.dp_degree}",
                        f"                - --lr={lr}",
                        f"                - --epochs={epochs}",
                        "                - --checkpoint-dir=/fsx/checkpoints",
                        "              env:",
                        "                - name: FI_PROVIDER",
                        "                  value: efa",
                        "                - name: NCCL_PROTO",
                        "                  value: Simple",
                        "                - name: TORCH_NCCL_ASYNC_ERROR_HANDLING",
                        "                  value: '1'",
                        "              resources:",
                        "                limits:",
                        f"                  nvidia.com/gpu: {nproc}",
                        f"                  vpc.amazonaws.com/efa: {max(1, nproc // 2)}",
                    ]
                )
            )
    return "\n---\n".join(blocks) + "\n"


def compile_pipeline_dag(
    recipe_input: PipelineRecipe | Dict[str, Any] | str,
) -> Dict[str, Any]:
    """Validate cross-step contracts, run Kahn's topological wave sort, and lower to target manifests."""
    t0 = time.perf_counter()
    errors: List[Dict[str, Any]] = []

    try:
        if isinstance(recipe_input, str):
            stripped = recipe_input.strip()
            if stripped.startswith("{"):
                recipe = PipelineRecipe.from_dict(json.loads(stripped))
            else:
                recipe = parse_python_sdk_snippet(stripped)
        elif isinstance(recipe_input, dict):
            recipe = PipelineRecipe.from_dict(recipe_input)
        else:
            recipe = recipe_input
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "valid": False,
            "compile_ms": elapsed_ms,
            "errors": [
                {
                    "rule_id": "DAG-SYNTAX-00",
                    "step_id": "parser",
                    "message": f"Failed to parse pipeline specification: {exc}",
                }
            ],
            "waves": [],
            "step_count": 0,
            "edge_count": 0,
        }

    step_map: Dict[str, PipelineStepSpec] = {}
    for s in recipe.steps:
        if s.step_id in step_map:
            errors.append(
                {
                    "rule_id": "DAG-DUP-00",
                    "step_id": s.step_id,
                    "message": f"Duplicate step_id '{s.step_id}' declared in pipeline '{recipe.pipeline_id}'.",
                }
            )
        step_map[s.step_id] = s

    # Build directed graph & verify upstream step / artifact port existence
    adj: Dict[str, List[str]] = {sid: [] for sid in step_map}
    in_degree: Dict[str, int] = {sid: 0 for sid in step_map}
    edges: List[Dict[str, str]] = []

    for s in recipe.steps:
        declared_upstreams = set(s.depends_on)
        for port_name, ref in s.inputs.items():
            if "." not in ref:
                errors.append(
                    {
                        "rule_id": "DAG-ARTIFACT-02",
                        "step_id": s.step_id,
                        "message": f"Input port '{port_name}' reference '{ref}' must use 'upstream_step.output_name' format.",
                    }
                )
                continue
            up_step_id, up_out_name = ref.split(".", 1)
            declared_upstreams.add(up_step_id)
            if up_step_id not in step_map:
                errors.append(
                    {
                        "rule_id": "DAG-ARTIFACT-02",
                        "step_id": s.step_id,
                        "message": f"Input port '{port_name}' references missing upstream step '{up_step_id}'.",
                    }
                )
            elif up_out_name not in step_map[up_step_id].outputs:
                avail = sorted(step_map[up_step_id].outputs.keys())
                errors.append(
                    {
                        "rule_id": "DAG-ARTIFACT-02",
                        "step_id": s.step_id,
                        "message": (
                            f"Input port '{port_name}' references missing artifact '{ref}'; "
                            f"step '{up_step_id}' only produces {avail}."
                        ),
                    }
                )

        for up_id in sorted(declared_upstreams):
            if up_id not in step_map:
                if not any(
                    e["step_id"] == s.step_id and f"'{up_id}'" in e["message"]
                    for e in errors
                ):
                    errors.append(
                        {
                            "rule_id": "DAG-ARTIFACT-02",
                            "step_id": s.step_id,
                            "message": f"Step '{s.step_id}' depends on undefined step '{up_id}'.",
                        }
                    )
                continue
            adj[up_id].append(s.step_id)
            in_degree[s.step_id] += 1
            edges.append({"from": up_id, "to": s.step_id})

    # Kahn's topological wave algorithm
    waves: List[Dict[str, Any]] = []
    ready = deque(sorted([sid for sid, deg in in_degree.items() if deg == 0]))
    visited_order: List[str] = []
    working_deg = dict(in_degree)

    wave_idx = 0
    while ready:
        current_wave_ids = list(ready)
        ready.clear()
        waves.append(
            {
                "wave_index": wave_idx,
                "concurrency": len(current_wave_ids),
                "steps": [
                    {
                        "step_id": sid,
                        "step_kind": step_map[sid].step_kind,
                        "backend": step_map[sid].compute.backend,
                        "instance_type": step_map[sid].compute.instance_type,
                    }
                    for sid in current_wave_ids
                ],
            }
        )
        visited_order.extend(current_wave_ids)
        next_batch: List[str] = []
        for sid in current_wave_ids:
            for nxt in adj[sid]:
                working_deg[nxt] -= 1
                if working_deg[nxt] == 0:
                    next_batch.append(nxt)
        for nxt in sorted(next_batch):
            ready.append(nxt)
        wave_idx += 1

    if len(visited_order) < len(step_map):
        cyclic_nodes = sorted([sid for sid, deg in working_deg.items() if deg > 0])
        errors.append(
            {
                "rule_id": "DAG-CYCLE-01",
                "step_id": cyclic_nodes[0] if cyclic_nodes else "dag",
                "message": f"Topological cycle detected across steps {cyclic_nodes}; DAG must be acyclic.",
            }
        )

    # Cross-step schema / dtype / column / vocab contract checks
    for s in recipe.steps:
        for port_name, ref in s.inputs.items():
            if "." not in ref or port_name not in s.expected_inputs:
                continue
            up_step_id, up_out_name = ref.split(".", 1)
            if up_step_id not in step_map or up_out_name not in step_map[up_step_id].outputs:
                continue
            produced = step_map[up_step_id].outputs[up_out_name]
            expected = s.expected_inputs[port_name]

            if produced.storage_format != expected.storage_format:
                errors.append(
                    {
                        "rule_id": "DAG-SCHEMA-03",
                        "step_id": s.step_id,
                        "message": (
                            f"Storage format mismatch on '{port_name}' ({ref}): upstream produces "
                            f"'{produced.storage_format}' but '{s.step_id}' expects '{expected.storage_format}'."
                        ),
                    }
                )
            if produced.dtype != expected.dtype:
                errors.append(
                    {
                        "rule_id": "DAG-SCHEMA-03",
                        "step_id": s.step_id,
                        "message": (
                            f"Tensor/data dtype mismatch on '{port_name}' ({ref}): upstream produces "
                            f"'{produced.dtype}' but '{s.step_id}' expects '{expected.dtype}'."
                        ),
                    }
                )
            missing_cols = [c for c in expected.required_columns if c not in produced.required_columns]
            if missing_cols:
                errors.append(
                    {
                        "rule_id": "DAG-SCHEMA-03",
                        "step_id": s.step_id,
                        "message": (
                            f"Missing required schema columns {missing_cols} on '{port_name}' ({ref}); "
                            f"upstream only provides {produced.required_columns}."
                        ),
                    }
                )
            if (
                expected.vocab_size is not None
                and produced.vocab_size is not None
                and expected.vocab_size != produced.vocab_size
            ):
                errors.append(
                    {
                        "rule_id": "DAG-SCHEMA-03",
                        "step_id": s.step_id,
                        "message": (
                            f"Tokenizer vocab_size mismatch on '{port_name}': upstream vocab={produced.vocab_size} "
                            f"!= expected vocab={expected.vocab_size}."
                        ),
                    }
                )

        # Distributed accelerator world-size & HBM memory check
        comp = s.compute
        if comp.num_accelerators > 0:
            world_product = comp.tp_degree * comp.pp_degree * comp.dp_degree
            if world_product != comp.num_accelerators:
                errors.append(
                    {
                        "rule_id": "DAG-TOPOLOGY-04",
                        "step_id": s.step_id,
                        "message": (
                            f"Invalid 3D parallelism topology: TP({comp.tp_degree}) * PP({comp.pp_degree}) * "
                            f"DP({comp.dp_degree}) = {world_product} != num_accelerators({comp.num_accelerators})."
                        ),
                    }
                )
            if comp.per_rank_mem_gb > comp.instance_hbm_gb:
                errors.append(
                    {
                        "rule_id": "DAG-TOPOLOGY-04",
                        "step_id": s.step_id,
                        "message": (
                            f"Per-rank memory {comp.per_rank_mem_gb:.1f} GB exceeds {comp.instance_type} "
                            f"HBM limit of {comp.instance_hbm_gb:.1f} GB."
                        ),
                    }
                )

    canonical_json = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    dag_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    valid = len(errors) == 0

    distributed_steps = [s for s in recipe.steps if s.compute.num_accelerators > 0]
    hyperpod_yaml = _render_hyperpod_yaml(recipe, distributed_steps) if valid else ""
    python_sdk_str = recipe_to_python_sdk(recipe)

    serverless_plan = {
        "pipeline_id": recipe.pipeline_id,
        "dag_sha256": dag_digest,
        "execution_mode": "topological_wave_orchestrator",
        "total_waves": len(waves) if valid else 0,
        "max_wave_concurrency": max((w["concurrency"] for w in waves), default=0) if valid else 0,
        "waves": waves if valid else [],
    }

    compile_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "valid": valid,
        "compile_ms": compile_ms,
        "pipeline_id": recipe.pipeline_id,
        "title": recipe.title,
        "model_family": recipe.model_family,
        "dag_sha256": dag_digest,
        "step_count": len(recipe.steps),
        "edge_count": len(edges),
        "edges": edges,
        "waves": waves if valid else [],
        "errors": errors,
        "serverless_plan": serverless_plan,
        "hyperpod_manifest_yaml": hyperpod_yaml,
        "python_sdk_code": python_sdk_str,
        "canonical_dag": recipe.to_dict(),
    }


def build_preset_recipes() -> Dict[str, PipelineRecipe]:
    """Return production reference recipes and static-contract red-team probes."""
    nova_sft = PipelineRecipe(
        pipeline_id="nova-sft-rag-prod",
        title="Enterprise Grounded QA SFT + Parallel Eval & Canary Gate",
        description="6-step diamond DAG: DataPrep -> FSDP LoRA FineTune -> Parallel [BedrockEval, LatencyProfile] -> ConditionGate -> Canary Model Registry.",
        model_family="us.amazon.nova-lite-v1:0",
        steps=[
            DataPrepStep(
                step_id="data_prep",
                outputs={
                    "clean_shards": {
                        "name": "clean_shards",
                        "artifact_type": "dataset",
                        "storage_format": "jsonl",
                        "dtype": "bfloat16",
                        "required_columns": ["prompt", "response", "domain", "safety_tag"],
                        "vocab_size": 128256,
                    }
                },
                hyperparams={"dedup_window": 64, "min_tokens": 8, "dataset_version": "v2.4"},
                compute={"backend": "serverless_lambda", "instance_type": "lambda.python3.12"},
            ),
            FineTuneStep(
                step_id="fsdp_finetune",
                depends_on=["data_prep"],
                inputs={"train_data": "data_prep.clean_shards"},
                expected_inputs={
                    "train_data": {
                        "name": "train_data",
                        "artifact_type": "dataset",
                        "storage_format": "jsonl",
                        "dtype": "bfloat16",
                        "required_columns": ["prompt", "response", "domain"],
                        "vocab_size": 128256,
                    }
                },
                outputs={
                    "adapter_weights": {
                        "name": "adapter_weights",
                        "artifact_type": "weights",
                        "storage_format": "safetensors",
                        "dtype": "bfloat16",
                        "required_columns": [],
                        "vocab_size": 128256,
                    }
                },
                hyperparams={"learning_rate": 0.0002, "epochs": 3, "lora_rank": 16, "seed": 42},
                compute={
                    "backend": "hyperpod_pytorch",
                    "instance_type": "ml.p5.48xlarge",
                    "num_accelerators": 8,
                    "tp_degree": 2,
                    "pp_degree": 1,
                    "dp_degree": 4,
                    "per_rank_mem_gb": 54.0,
                    "instance_hbm_gb": 80.0,
                },
            ),
            LatencyProfileStep(
                step_id="latency_profile",
                depends_on=["fsdp_finetune"],
                inputs={"weights": "fsdp_finetune.adapter_weights"},
                expected_inputs={
                    "weights": {
                        "name": "weights",
                        "artifact_type": "weights",
                        "storage_format": "safetensors",
                        "dtype": "bfloat16",
                        "required_columns": [],
                    }
                },
                outputs={
                    "perf_profile": {
                        "name": "perf_profile",
                        "artifact_type": "profile",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["estimated_p95_ms"],
                    }
                },
                hyperparams={"batch_size": 8, "sequence_len": 1024},
                compute={"backend": "serverless_lambda", "instance_type": "lambda.python3.12"},
            ),
            BedrockEvalStep(
                step_id="bedrock_eval",
                depends_on=["fsdp_finetune"],
                inputs={"weights": "fsdp_finetune.adapter_weights"},
                expected_inputs={
                    "weights": {
                        "name": "weights",
                        "artifact_type": "weights",
                        "storage_format": "safetensors",
                        "dtype": "bfloat16",
                        "required_columns": [],
                    }
                },
                outputs={
                    "eval_scorecard": {
                        "name": "eval_scorecard",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["quality_score", "p95_latency_ms", "safety_passed"],
                    }
                },
                hyperparams={"rubric_version": "v2.1", "temperature": 0.1, "candidate_profile": "cand-nova-rag-sft-v2"},
                compute={"backend": "bedrock_eval", "instance_type": "us.amazon.nova-lite-v1:0"},
            ),
            ConditionStep(
                step_id="quality_gate",
                depends_on=["bedrock_eval", "latency_profile"],
                inputs={
                    "scorecard": "bedrock_eval.eval_scorecard",
                    "perf": "latency_profile.perf_profile",
                },
                expected_inputs={
                    "scorecard": {
                        "name": "scorecard",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["quality_score", "p95_latency_ms", "safety_passed"],
                    },
                    "perf": {
                        "name": "perf",
                        "artifact_type": "profile",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["estimated_p95_ms"],
                    },
                },
                outputs={
                    "gate_decision": {
                        "name": "gate_decision",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["approved"],
                    }
                },
                hyperparams={"min_quality_score": 0.82, "max_p95_latency_ms": 2500.0, "require_safety_pass": True},
                condition_rules={"min_quality_score": 0.82, "max_p95_latency_ms": 2500.0, "require_safety_pass": True},
            ),
            RegisterModelStep(
                step_id="register_model",
                depends_on=["quality_gate"],
                inputs={"gate": "quality_gate.gate_decision"},
                expected_inputs={
                    "gate": {
                        "name": "gate",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["approved"],
                    }
                },
                outputs={
                    "model_package": {
                        "name": "model_package",
                        "artifact_type": "model_package",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["model_version_id", "deployment_stage"],
                    }
                },
                hyperparams={"target_stage": "Canary_10Pct", "auto_promote_production": True},
            ),
        ],
    )

    verl_grpo = PipelineRecipe(
        pipeline_id="verl-grpo-reasoning",
        title="RLVR Reasoning Pipeline (verl GRPO on EKS RayJob)",
        description="5-step RLVR workflow lowering GRPO policy optimization to Kubernetes RayJob with verifiable reward rubrics.",
        model_family="us.amazon.nova-lite-v1:0",
        steps=[
            DataPrepStep(
                step_id="rlvr_prep",
                outputs={
                    "rl_prompts": {
                        "name": "rl_prompts",
                        "artifact_type": "dataset",
                        "storage_format": "parquet",
                        "dtype": "bfloat16",
                        "required_columns": ["prompt", "response", "reward_rubric"],
                        "vocab_size": 128256,
                    }
                },
                hyperparams={"dedup_window": 96, "min_tokens": 10, "dataset_version": "rlvr-v1.2"},
            ),
            FineTuneStep(
                step_id="verl_grpo_train",
                depends_on=["rlvr_prep"],
                inputs={"prompts": "rlvr_prep.rl_prompts"},
                expected_inputs={
                    "prompts": {
                        "name": "prompts",
                        "artifact_type": "dataset",
                        "storage_format": "parquet",
                        "dtype": "bfloat16",
                        "required_columns": ["prompt", "reward_rubric"],
                        "vocab_size": 128256,
                    }
                },
                outputs={
                    "policy_ckpt": {
                        "name": "policy_ckpt",
                        "artifact_type": "weights",
                        "storage_format": "safetensors",
                        "dtype": "bfloat16",
                        "required_columns": [],
                    }
                },
                hyperparams={"learning_rate": 0.0001, "epochs": 2, "lora_rank": 32, "group_size": 8},
                compute={
                    "backend": "eks_ray_verl",
                    "instance_type": "ml.p5.48xlarge",
                    "num_accelerators": 16,
                    "tp_degree": 4,
                    "pp_degree": 1,
                    "dp_degree": 4,
                    "per_rank_mem_gb": 62.5,
                    "instance_hbm_gb": 80.0,
                },
            ),
            BedrockEvalStep(
                step_id="bedrock_eval",
                depends_on=["verl_grpo_train"],
                inputs={"weights": "verl_grpo_train.policy_ckpt"},
                expected_inputs={
                    "weights": {
                        "name": "weights",
                        "artifact_type": "weights",
                        "storage_format": "safetensors",
                        "dtype": "bfloat16",
                        "required_columns": [],
                    }
                },
                outputs={
                    "eval_scorecard": {
                        "name": "eval_scorecard",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["quality_score", "p95_latency_ms", "safety_passed"],
                    }
                },
                hyperparams={"rubric_version": "rlvr-v1", "temperature": 0.1, "candidate_profile": "cand-nova-rag-sft-v2"},
                compute={"backend": "bedrock_eval", "instance_type": "us.amazon.nova-lite-v1:0"},
            ),
            ConditionStep(
                step_id="quality_gate",
                depends_on=["bedrock_eval"],
                inputs={"scorecard": "bedrock_eval.eval_scorecard"},
                expected_inputs={
                    "scorecard": {
                        "name": "scorecard",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["quality_score", "p95_latency_ms", "safety_passed"],
                    }
                },
                outputs={
                    "gate_decision": {
                        "name": "gate_decision",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["approved"],
                    }
                },
                hyperparams={"min_quality_score": 0.85, "max_p95_latency_ms": 2500.0, "require_safety_pass": True},
                condition_rules={"min_quality_score": 0.85, "max_p95_latency_ms": 2500.0, "require_safety_pass": True},
            ),
            RegisterModelStep(
                step_id="register_model",
                depends_on=["quality_gate"],
                inputs={"gate": "quality_gate.gate_decision"},
                expected_inputs={
                    "gate": {
                        "name": "gate",
                        "artifact_type": "eval_report",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["approved"],
                    }
                },
                outputs={
                    "model_package": {
                        "name": "model_package",
                        "artifact_type": "model_package",
                        "storage_format": "json",
                        "dtype": "json",
                        "required_columns": ["model_version_id", "deployment_stage"],
                    }
                },
                hyperparams={"target_stage": "Canary_10Pct", "auto_promote_production": False},
            ),
        ],
    )

    # Broken Recipe 1: Dependency Cycle
    bug_cycle = PipelineRecipe.from_dict(nova_sft.to_dict())
    bug_cycle.pipeline_id = "bug-cyclic-feedback"
    bug_cycle.title = "Contract Fault: Feedback Cycle (register_model -> fsdp_finetune)"
    bug_cycle.steps[1].depends_on.append("register_model")

    # Broken Recipe 2: Missing Upstream Artifact
    bug_missing = PipelineRecipe.from_dict(nova_sft.to_dict())
    bug_missing.pipeline_id = "bug-missing-artifact"
    bug_missing.title = "Contract Fault: Missing Upstream Artifact Port"
    bug_missing.steps[1].inputs["train_data"] = "data_prep.nonexistent_parquet_v9"

    # Broken Recipe 3: Cross-Step Schema & Dtype Mismatch
    bug_schema = PipelineRecipe.from_dict(nova_sft.to_dict())
    bug_schema.pipeline_id = "bug-schema-dtype-mismatch"
    bug_schema.title = "Contract Fault: CSV/float32 vs JSONL/bfloat16 Schema Mismatch"
    bug_schema.steps[0].outputs["clean_shards"] = ArtifactContract(
        name="clean_shards",
        artifact_type="dataset",
        storage_format="csv",
        dtype="float32",
        required_columns=["prompt"],
        vocab_size=32000,
    )

    # Broken Recipe 4: Invalid 3D Parallelism Topology & HBM Overflow
    bug_topology = PipelineRecipe.from_dict(nova_sft.to_dict())
    bug_topology.pipeline_id = "bug-topology-hbm-overflow"
    bug_topology.title = "Contract Fault: TP*PP*DP World-Size Mismatch & HBM Overflow"
    bug_topology.steps[1].compute = ComputeResources(
        backend="hyperpod_pytorch",
        instance_type="ml.p5.48xlarge",
        num_accelerators=8,
        tp_degree=4,
        pp_degree=2,
        dp_degree=2,
        per_rank_mem_gb=94.5,
        instance_hbm_gb=80.0,
    )

    return {
        nova_sft.pipeline_id: nova_sft,
        verl_grpo.pipeline_id: verl_grpo,
        bug_cycle.pipeline_id: bug_cycle,
        bug_missing.pipeline_id: bug_missing,
        bug_schema.pipeline_id: bug_schema,
        bug_topology.pipeline_id: bug_topology,
    }


PRESET_RECIPES: Dict[str, PipelineRecipe] = build_preset_recipes()
