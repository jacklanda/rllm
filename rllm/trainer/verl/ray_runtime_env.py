import json
import os
from pathlib import Path

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # Keep full launcher visibility in Ray workers. Verl/rLLM binds each
        # worker with torch.cuda.set_device(local_rank). Letting Ray narrow
        # CUDA_VISIBLE_DEVICES for fractional-GPU actors makes every colocated
        # FSDP rank see cuda:0 and NCCL reports duplicate physical GPUs.
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
    "worker_process_setup_hook": "rllm.experimental.verl.patch.apply_all_verl_patches",
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    "DOCKER_",
    "RAY_",
    "RLLM_",
]

DEFAULT_EXCLUDE_VARS = {
    "CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RLLM_EXCLUDE",
}


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = [name.strip() for name in str(os.environ.get("RLLM_EXCLUDE")).split(",") if name.strip()]
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    exclude_vars = set(DEFAULT_EXCLUDE_VARS)
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            prefix = name.replace("*", "_")
            if prefix in forward_prefix:
                forward_prefix.remove(prefix)
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_prefix) and k not in exclude_vars}
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        forwarded.setdefault("RLLM_CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    return forwarded


def _get_repo_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def _get_r2egym_src() -> str | None:
    configured = os.environ.get("R2EGYM_PATH") or os.environ.get("R2E_GYM_PATH")
    candidates = []
    if configured:
        candidates.extend([Path(configured), Path(configured) / "src"])
    repo_root = Path(_get_repo_root())
    candidates.extend(
        [
            repo_root.parent / "R2E-Gym" / "src",
            Path("/share/nlp/liuyang/workspace/gem/R2E-Gym/src"),
        ]
    )
    for path in candidates:
        if (path / "r2egym").is_dir():
            return str(path)
    return None


def _prepend_pythonpath(env: dict[str, str], path: str) -> None:
    entries = [entry for entry in env.get("PYTHONPATH", os.environ.get("PYTHONPATH", "")).split(os.pathsep) if entry]
    if path in entries:
        entries.remove(path)
    env["PYTHONPATH"] = os.pathsep.join([path, *entries])


def get_ppo_ray_runtime_env():
    env = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    env.update(_get_forwarded_env_vars())
    _prepend_pythonpath(env, _get_repo_root())
    r2egym_src = _get_r2egym_src()
    if r2egym_src:
        _prepend_pythonpath(env, r2egym_src)

    job_runtime_env = {}
    job_config_str = os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR)
    if job_config_str:
        try:
            job_runtime_env = json.loads(job_config_str).get("runtime_env", {}) or {}
        except json.JSONDecodeError:
            job_runtime_env = {}

    for key in (job_runtime_env.get("env_vars") or {}):
        env.pop(key, None)

    runtime_env = {"env_vars": env}
    if "worker_process_setup_hook" not in job_runtime_env:
        runtime_env["worker_process_setup_hook"] = PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"]
    if "working_dir" not in job_runtime_env:
        runtime_env["working_dir"] = None
    return runtime_env
