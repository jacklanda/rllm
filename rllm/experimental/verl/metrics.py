"""Compatibility helpers around Verl's debug metric utilities."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_EMPTY_REDUCTION_ERROR_SNIPPETS = (
    "Expected reduction dim to be specified",
    "input.numel() == 0",
)

SEARCH_AGENT_TURN_RAW_KEYS = (
    "tool_call_counts",
    "all_call_tool_counts",
    "all_call_tool_success_counts",
)
SEARCH_AGENT_ABNORMAL_RAW_KEYS = (
    "searched_query_count",
    "too_many_tool_call_count",
    "tool_parser_error_count",
    "response_truncated_count",
    "too_many_turn_count",
    "too_long_seq_truncated_count",
    "duplicate_search_result_count",
)
SEARCH_AGENT_RAW_METRIC_KEYS = SEARCH_AGENT_TURN_RAW_KEYS + SEARCH_AGENT_ABNORMAL_RAW_KEYS

_SEARCH_AGENT_ALIAS_KEYS = {
    "tool_call_counts": ("tool_call_turns", "tool_turns", "num_tool_turns"),
    "all_call_tool_counts": ("total_tool_calls", "tool_calls"),
    "all_call_tool_success_counts": ("total_tool_success", "successful_tool_calls"),
    "searched_query_count": ("duplicate_query_count", "duplicate_search_detected"),
    "too_many_tool_call_count": ("excessive_parallel_calls",),
    "tool_parser_error_count": ("total_parse_tool_args_error", "parse_tool_args_error"),
    "response_truncated_count": ("response_truncated",),
    "too_many_turn_count": ("too_many_turns",),
    "too_long_seq_truncated_count": ("overlong", "response_clipped"),
    "duplicate_search_result_count": ("duplicate_search_result_detected",),
}


def _as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool | np.bool_):
        return float(value)
    if isinstance(value, int | float | np.number):
        return float(value)
    return default


def _first_number(metadata: dict[str, Any], key: str, aliases: tuple[str, ...] = ()) -> float | None:
    for candidate in (key, *aliases):
        if candidate in metadata:
            return _as_number(metadata[candidate])
    return None


def canonicalize_search_agent_metric_metadata(metadata: dict[str, Any] | None) -> dict[str, int]:
    """Map rollout metadata into SearchAgent-Zero's raw metric counter names.

    If a trajectory already provides the reference raw keys, they are preserved.
    A few rLLM recipe aliases are accepted so existing rollouts can produce the
    same recorded metrics without changing their public metadata shape.
    """
    if not metadata:
        return {}

    has_search_metric = any(key in metadata for key in SEARCH_AGENT_RAW_METRIC_KEYS)
    has_search_metric = has_search_metric or any(alias in metadata for aliases in _SEARCH_AGENT_ALIAS_KEYS.values() for alias in aliases)
    if not has_search_metric:
        return {}

    values: dict[str, int] = {}
    for key in SEARCH_AGENT_RAW_METRIC_KEYS:
        value = _first_number(metadata, key, _SEARCH_AGENT_ALIAS_KEYS.get(key, ()))
        values[key] = int(value or 0)

    if "all_call_tool_success_counts" not in metadata:
        total = values["all_call_tool_counts"]
        parser_errors = values["tool_parser_error_count"]
        return_errors = int(_as_number(metadata.get("total_tool_return_error", metadata.get("tool_return_error", 0))))
        values["all_call_tool_success_counts"] = max(total - parser_errors - return_errors, 0)

    return values


def _metric_values(batch: Any, key: str) -> np.ndarray | None:
    if not hasattr(batch, "non_tensor_batch") or batch.non_tensor_batch is None:
        return None
    if key not in batch.non_tensor_batch:
        return None

    values = np.asarray(batch.non_tensor_batch[key], dtype=np.float64)
    if values.size == 0:
        return None

    ignore = batch.non_tensor_batch.get("ignore_in_loss")
    if ignore is not None and len(ignore) == len(values):
        values = values[~np.asarray(ignore, dtype=bool)]
    return values if values.size > 0 else None


def compute_search_agent_metrics(batch: Any) -> dict[str, float]:
    """Reduce SearchAgent-Zero raw counters from a DataProto into log metrics."""
    metrics: dict[str, float] = {}

    tool_call_counts = _metric_values(batch, "tool_call_counts")
    if tool_call_counts is not None:
        metrics["turn/tool_call_turn/min"] = float(tool_call_counts.min())
        metrics["turn/tool_call_turn/max"] = float(tool_call_counts.max())
        metrics["turn/tool_call_turn/mean"] = float(tool_call_counts.mean())

    all_call_tool_counts = _metric_values(batch, "all_call_tool_counts")
    tool_success_counts = _metric_values(batch, "all_call_tool_success_counts")
    if all_call_tool_counts is not None and tool_success_counts is not None:
        metrics["turn/all_call_tool_counts/min"] = float(all_call_tool_counts.min())
        metrics["turn/all_call_tool_counts/max"] = float(all_call_tool_counts.max())
        metrics["turn/all_call_tool_counts/mean"] = float(all_call_tool_counts.mean())

        metrics["turn/tool_call_success_counts/min"] = float(tool_success_counts.min())
        metrics["turn/tool_call_success_counts/max"] = float(tool_success_counts.max())
        metrics["turn/tool_call_success_counts/mean"] = float(tool_success_counts.mean())

        total_calls = all_call_tool_counts.sum()
        metrics["turn/tool_call_success_rate/mean"] = float(tool_success_counts.sum() / total_calls) if total_calls > 0 else 1.0

    for key in SEARCH_AGENT_ABNORMAL_RAW_KEYS:
        values = _metric_values(batch, key)
        if values is not None:
            metrics[f"abnormal_trajectory/{key}_percentage"] = float(values.sum() / len(values))

    return metrics


def _load_verl_calculate_debug_metrics() -> Callable[[Any], dict[str, float]]:
    """Load Verl's debug metrics helper lazily to preserve optional dependency boundaries."""
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics


def _default_debug_metrics() -> dict[str, float]:
    """Mirror the newer upstream fallback for all-zero valid-token masks."""
    return {
        "training/rollout_probs_diff_valid": 0,
        "training/rollout_probs_diff_max": float("nan"),
        "training/rollout_probs_diff_mean": float("nan"),
        "training/rollout_probs_diff_std": float("nan"),
        "training/rollout_actor_probs_pearson_corr": float("nan"),
    }


def _is_legacy_empty_mask_error(exc: RuntimeError) -> bool:
    """Detect the empty-mask reduction failure from older Verl helper versions."""
    message = str(exc)
    return all(snippet in message for snippet in _EMPTY_REDUCTION_ERROR_SNIPPETS)


def _normalize_degenerate_std(metrics: dict[str, float]) -> dict[str, float]:
    """Clamp the single-token std case to zero without masking broader metric failures."""
    std = metrics.get("training/rollout_probs_diff_std", float("nan"))
    if metrics.get("training/rollout_probs_diff_valid") != 1 or not math.isnan(std):
        return metrics

    if math.isnan(metrics.get("training/rollout_probs_diff_max", float("nan"))):
        return metrics
    if math.isnan(metrics.get("training/rollout_probs_diff_mean", float("nan"))):
        return metrics

    metrics["training/rollout_probs_diff_std"] = 0.0
    return metrics


def calculate_debug_metrics_compat(data: Any) -> dict[str, float]:
    """Delegate to Verl's helper while backfilling legacy empty-mask behavior."""
    try:
        metrics = _load_verl_calculate_debug_metrics()(data)
    except RuntimeError as exc:
        if not _is_legacy_empty_mask_error(exc):
            raise
        logger.warning("Verl debug metrics hit an empty valid-token mask, returning default metrics")
        return _default_debug_metrics()

    return _normalize_degenerate_std(metrics)
