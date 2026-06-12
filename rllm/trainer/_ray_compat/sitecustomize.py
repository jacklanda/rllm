"""Compatibility patches for Python subprocesses spawned by Ray.

Ray 2.55 starts a dashboard process even when `include_dashboard=False`. In
some environments, `opentelemetry-exporter-prometheus` imports a newer SDK
environment-variable constant that is absent from the installed SDK package.
Patch that constant early so dashboard module discovery does not abort Ray
startup.
"""

import warnings


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


def _patch_opentelemetry_prometheus_env_var() -> None:
    try:
        from opentelemetry.sdk import environment_variables
    except Exception:
        return

    name = "OTEL_PYTHON_EXPERIMENTAL_DISABLE_PROMETHEUS_UNIT_NORMALIZATION"
    if not hasattr(environment_variables, name):
        setattr(environment_variables, name, name)


_suppress_vllm_fla_short_sequence_format_warning()
_patch_opentelemetry_prometheus_env_var()
