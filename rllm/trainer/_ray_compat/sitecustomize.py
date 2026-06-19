"""Compatibility patches for Python subprocesses spawned by Ray.

Ray 2.55 starts a dashboard process even when `include_dashboard=False`. In
some environments, `opentelemetry-exporter-prometheus` imports a newer SDK
environment-variable constant that is absent from the installed SDK package.
Patch that constant early so dashboard module discovery does not abort Ray
startup.
"""

import warnings


def _suppress_verl_vllm_noise() -> None:
    """Suppress selected noisy warnings in early Ray child-process startup."""

    try:
        from rllm.trainer.verl.warning_filters import apply_verl_vllm_noise_filters
    except Exception:
        return

    apply_verl_vllm_noise_filters()


def _suppress_vllm_fla_short_sequence_format_warning() -> None:
    """Suppress vLLM FLA's false-positive layout warning on short chunks."""

    warnings.filterwarnings(
        "ignore",
        message=(
            r"Input tensor shape suggests potential format mismatch: "
            r"seq_len \(\d+\) < num_heads \(\d+\)\. "
            r"This may indicate the inputs were passed in head-first format "
            r"\[B, H, T, \.\.\.\].*"
        ),
        category=UserWarning,
    )


def _suppress_torch_inductor_online_softmax_warning() -> None:
    """Suppress a PyTorch Inductor performance warning emitted from worker codegen."""

    warnings.filterwarnings(
        "ignore",
        message=(
            r"Online softmax is disabled on the fly since Inductor decides to\n"
            r"split the reduction\..*"
        ),
        category=UserWarning,
        module=r"torch\._inductor\.lowering",
    )


def _patch_opentelemetry_prometheus_env_var() -> None:
    try:
        from opentelemetry.sdk import environment_variables
    except Exception:
        return

    name = "OTEL_PYTHON_EXPERIMENTAL_DISABLE_PROMETHEUS_UNIT_NORMALIZATION"
    if not hasattr(environment_variables, name):
        setattr(environment_variables, name, name)


def _patch_transformers_use_return_dict() -> None:
    try:
        from transformers.configuration_utils import PreTrainedConfig
    except Exception:
        return

    use_return_dict = getattr(PreTrainedConfig, "use_return_dict", None)
    if not isinstance(use_return_dict, property):
        return

    def _get_return_dict(self):
        return self.return_dict

    def _set_return_dict(self, value):
        self.return_dict = value

    PreTrainedConfig.use_return_dict = property(_get_return_dict, _set_return_dict)


_suppress_verl_vllm_noise()
_suppress_vllm_fla_short_sequence_format_warning()
_suppress_torch_inductor_online_softmax_warning()
_patch_opentelemetry_prometheus_env_var()
_patch_transformers_use_return_dict()
