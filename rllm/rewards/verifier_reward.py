"""Verifier-based reward for MCP/General Agent tasks.

Evaluates agent responses by executing a verifier's ``verification_code``
that ships with each task. The verifier defines a ``verify(tools, answer)``
function which returns a score dict.

Ported from wutong1's ``examples/general_agent/reward_verifier.py``.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)

_MCP_REGISTRATION_LOGGERS = (
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp",
    "mcp.server.fastmcp.tools",
    "mcp.server.fastmcp.tools.tool_manager",
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@contextmanager
def _silence_mcp_registration_loggers():
    """Mute FastMCP registration warnings while importing task tools."""
    previous = {}
    for name in _MCP_REGISTRATION_LOGGERS:
        log = logging.getLogger(name)
        previous[name] = (log.level, log.propagate, log.disabled)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
        log.disabled = True
    try:
        yield
    finally:
        for name, state in previous.items():
            log = logging.getLogger(name)
            log.level, log.propagate, log.disabled = state


# ---------------------------------------------------------------------------
# Failure-taxonomy helpers (fix #9)
# ---------------------------------------------------------------------------

_STRUCTURAL_MSG_FRAGMENTS = (
    "not a list",
    "must be a list",
    "is not a dict",
    "must be a dict",
    "missing required top-level",
    "missing required key",
    "missing required section",
    "payload",
    "schema",
    "json",
    "invalid format",
)

_CONTENT_MSG_FRAGMENTS = (
    "could not verify",
    "insufficient verification",
    "entries matched",
    "does not match",
    "does not contain",
    "mismatch",
)


def _looks_like_structural_msg(msg: str) -> bool:
    low = (msg or "").lower()
    return any(frag in low for frag in _STRUCTURAL_MSG_FRAGMENTS)


def _looks_like_content_msg(msg: str) -> bool:
    low = (msg or "").lower()
    return any(frag in low for frag in _CONTENT_MSG_FRAGMENTS)


def _coerce_payload_shape(payload: Any) -> Any:
    """Adapt a dict payload to a list when the verifier demands a list.

    The three ``submit_result_difficulty_{1,2,3}`` tools declare mutually
    incompatible outer shapes (array vs. object).  When the model picks
    the object shape but the verifier expects an array, the inner list
    is usually present under a single wrapper key.  Recover it.

    Coercion strategies (tried in order):
    1. Single-key dict whose value is a list → return that list.
    2. Dict with a known wrapper key (result/items/data/...) → return its list.
    3. Dict that is itself a plausible list *element* → wrap as [payload].
    """
    if isinstance(payload, dict):
        # Strategy 1: single-key wrapper
        if len(payload) == 1:
            only_val = next(iter(payload.values()))
            if isinstance(only_val, list):
                return only_val
        # Strategy 2: known wrapper keys
        for key in ("result", "results", "items", "data", "answer", "pairings"):
            v = payload.get(key)
            if isinstance(v, list):
                return v
        # Strategy 3: bare single-element — the agent submitted one dict instead
        # of a list of dicts.  Wrap it unless it looks like a meta/schema wrapper.
        _META_KEYS = {"type", "schema", "format", "description", "$schema", "definitions"}
        if not (payload.keys() & _META_KEYS):
            return [payload]
    # Handle double-stringified JSON: the payload is still a string after the
    # first json.loads pass (e.g. finish tool result was double-escaped).
    if isinstance(payload, str):
        try:
            inner = json.loads(payload)
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict):
                return _coerce_payload_shape(inner)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Tool-call reward helpers
# ---------------------------------------------------------------------------


def _get_tool_call_reward(task_info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute step-penalty and tool-call bonus/penalty.

    Returns (raw_tool_call_reward, stats_dict).

    The raw value is NOT added unconditionally: the caller applies the
    step penalty only when ``base_reward > 0`` and clamps so a correct
    rollout cannot become negative.  Trajectory-file evidence: 95 MCP
    rollouts passed the verifier but the unconditional penalty pushed
    their final reward below the base, 6 % of them past zero.
    """
    stats = task_info.get("tool_call_stats", {}) if isinstance(task_info, dict) else {}
    submit_called = bool(stats.get("submit_called"))
    non_submit_calls = int(stats.get("non_submit_tool_calls", 0)) if isinstance(stats, dict) else 0
    step_count = int(stats.get("step_count", 0)) if isinstance(stats, dict) else 0

    min_good_steps = 4
    max_good_steps = 8

    step_penalty = 0.0
    if submit_called:
        if step_count <= 2:
            step_penalty -= 0.5
        elif step_count < min_good_steps:
            step_penalty -= 0.25 * (min_good_steps - step_count)
        elif step_count > max_good_steps:
            step_penalty -= 0.2 * (step_count - max_good_steps)

    # Fix #3: activate tool_call_bonus for genuine exploration. The eval
    # dump showed agents that explored 6+ tools had a 67.6 % correct rate
    # versus 52.1 % for agents that gave up under 6 calls — yet the bonus
    # channel fired 0/1024 times, so exploration was punished (via step
    # penalty) without ever being rewarded. We credit each *distinct
    # successful* non-submit tool name once, capped at +0.15. The caller
    # further gates this by requiring base_reward > 0, so the bonus can
    # only amplify correct rollouts and cannot be farmed by failure.
    distinct_successful = int(stats.get("distinct_successful_tools", 0)) if isinstance(stats, dict) else 0
    tool_call_bonus = min(0.15, 0.03 * distinct_successful)
    no_tool_use = bool(submit_called and non_submit_calls == 0)

    stats = dict(stats) if isinstance(stats, dict) else {}
    stats["step_penalty"] = float(step_penalty)
    stats["tool_call_bonus"] = float(tool_call_bonus)
    stats["no_tool_use"] = no_tool_use
    stats["total_tool_call_reward"] = float(step_penalty + tool_call_bonus)

    return step_penalty + tool_call_bonus, stats


# ---------------------------------------------------------------------------
# Tools context helpers
# ---------------------------------------------------------------------------


def _wrap_tool_for_verifier(fn):
    """Wrap a raw tool function so it returns ``{"result": <value>}`` format.

    MCP task verifiers universally expect tool functions to return a dict with
    a ``"result"`` key wrapping the actual data.  When tools are loaded as raw
    Python functions (instead of called via the MCP protocol), their return
    values are unwrapped (e.g. a plain ``list``).  This wrapper bridges that
    gap so the verifier's ``isinstance(result, dict) and "result" in result``
    checks succeed.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        # If the function already returns the expected format, pass through
        if isinstance(result, dict) and "result" in result:
            return result
        return {"result": result}

    return wrapped


def _load_tools_from_tools_py(tools_py: str) -> dict[str, Any]:
    """Dynamically load public callables from a tools.py file.

    Each callable is wrapped with :func:`_wrap_tool_for_verifier` so that its
    return value matches the ``{"result": <data>}`` format that MCP task
    verifiers expect.
    """
    tools_path = Path(tools_py)
    if not tools_path.exists():
        return {}
    module_name = f"_reward_tools_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    tools_dir = str(tools_path.parent)
    _inserted = tools_dir not in _sys.path
    if _inserted:
        _sys.path.insert(0, tools_dir)
    try:
        with _silence_mcp_registration_loggers():
            spec.loader.exec_module(module)
    finally:
        if _inserted and tools_dir in _sys.path:
            _sys.path.remove(tools_dir)

    tools: dict[str, Any] = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isclass(value):
            continue
        if callable(value):
            tools[name] = _wrap_tool_for_verifier(value)
    return tools


def _build_tools_context(task_info: dict[str, Any]) -> dict[str, Any]:
    """Build a tools dict for the verifier from task_info."""
    tools_context = task_info.get("tools") or {}
    if tools_context:
        return tools_context
    tools_py = task_info.get("tools_py", "")
    if tools_py:
        return _load_tools_from_tools_py(tools_py)
    return {}


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------


def verifier_reward_fn(task_info: dict[str, Any], action: str) -> RewardOutput:
    """Compute reward for general agent / MCP tasks using verifier code.

    Args:
        task_info: Task dict containing ``verifier`` (with ``verification_code``),
                   and optionally ``tools_py``, ``data_root``, ``difficulty``, and
                   ``tool_call_stats``.
        action: The answer string submitted by the agent.

    Returns:
        RewardOutput with reward score and metadata.
    """
    verifier = task_info.get("verifier", {})
    empty_reward_metadata = {
        "reward/base_reward": 0.0,
        "reward/tool_call_total": -0.5,
        "reward/step_penalty": 0.0,
        "reward/tool_call_bonus": 0.0,
    }
    verification_code = ""
    if isinstance(verifier, dict):
        verification_code = verifier.get("verification_code", "") or verifier.get("code", "")

    if not verification_code:
        logger.warning("[REWARD] Missing verification_code in task_info")
        return RewardOutput(reward=0.0, metadata={"error": "missing_verification_code", **empty_reward_metadata})

    exec_globals: dict[str, Any] = {}
    try:
        exec(verification_code, exec_globals)
    except Exception as exc:
        logger.error("[REWARD] Failed to exec verification_code: %s", exc)
        return RewardOutput(reward=0.0, metadata={"error": f"exec_failed: {exc}", **empty_reward_metadata})

    verify_fn = exec_globals.get("verify")
    if not callable(verify_fn):
        logger.error("[REWARD] verify function not found in verification_code")
        return RewardOutput(reward=0.0, metadata={"error": "verify_not_found", **empty_reward_metadata})

    tools_context = _build_tools_context(task_info)

    # Parse action if it's a JSON string
    parsed_action = action
    if isinstance(action, str):
        try:
            parsed_action = json.loads(action)
        except (json.JSONDecodeError, ValueError):
            parsed_action = action

    # Run the verifier
    try:
        verifier_result = verify_fn(tools_context, parsed_action)
    except Exception as exc:
        logger.error("[REWARD] verify_fn raised exception: %s", exc)
        return RewardOutput(reward=0.0, metadata={"error": f"verify_failed: {exc}", **empty_reward_metadata})

    # --- Submit-contract coercion retry (fix #6) --------------------------
    # 131/156 MCP negatives trace to the submit tool declaring
    # ``result: object`` while the verifier demands a list (or vice versa).
    # When the first pass reports a list/shape mismatch and the payload is
    # a single-list-valued dict, retry with the inner list.
    coercion_retry = False
    if isinstance(verifier_result, dict) and verifier_result.get("passed") is False and _looks_like_structural_msg(verifier_result.get("message", "")):
        coerced = _coerce_payload_shape(parsed_action)
        if coerced is not None and coerced is not parsed_action:
            try:
                retry_result = verify_fn(tools_context, coerced)
                if isinstance(retry_result, dict) and retry_result.get("passed") is True:
                    verifier_result = retry_result
                    coercion_retry = True
            except Exception:
                pass

    # Parse reward from verifier result
    base_reward = 0.0
    is_correct: bool | None = None
    if isinstance(verifier_result, dict):
        if isinstance(verifier_result.get("score"), (int, float)):
            base_reward = float(verifier_result["score"])
        elif verifier_result.get("passed") is True:
            base_reward = 1.0
        elif verifier_result.get("passed") is False:
            base_reward = 0.0
        if "passed" in verifier_result:
            is_correct = bool(verifier_result.get("passed"))
    elif isinstance(verifier_result, bool):
        is_correct = verifier_result
        base_reward = 1.0 if verifier_result else 0.0
    elif isinstance(verifier_result, (int, float)):
        base_reward = float(verifier_result)
        is_correct = base_reward > 0.0
    else:
        logger.warning("[REWARD] Verifier result is not a dict: %s", type(verifier_result))

    # Tool-call reward (step penalties etc.)
    tool_call_reward_raw, tool_call_stats = _get_tool_call_reward(task_info)
    step_penalty = float(tool_call_stats.get("step_penalty", 0.0))
    tool_call_bonus = float(tool_call_stats.get("tool_call_bonus", 0.0))

    # Fix #10: step penalty applies ONLY when the verifier gave a positive
    # base_reward, and the total is clamped to [0.0, 1.0] so a correct
    # rollout can never become negative and the tool_call_bonus cannot push
    # a perfect rollout above the verifier's ceiling.  In the training dump
    # 95 passed-but-devalued trajectories (9 % of accept_traj) had base=1.0
    # and final<1.0; 6 of those went below zero and trained against the truth.
    disable_step_penalty = _env_flag("RLLM_MCP_DISABLE_STEP_PENALTY")
    if base_reward > 0.0:
        applied_penalty = 0.0 if disable_step_penalty else step_penalty
        applied_bonus = tool_call_bonus
        total_reward = min(1.0, max(0.0, base_reward + applied_penalty + applied_bonus))
    else:
        applied_penalty = 0.0
        applied_bonus = 0.0
        total_reward = base_reward  # verifier-negative: leave untouched

    # Structural vs content failure classification (fix #9).
    structural_fail = 0
    content_fail = 0
    if is_correct is False and isinstance(verifier_result, dict):
        msg = verifier_result.get("message", "")
        if _looks_like_structural_msg(msg):
            structural_fail = 1
        elif _looks_like_content_msg(msg):
            content_fail = 1
        else:
            content_fail = 1  # default: unclassified failures count as content

    metadata: dict[str, Any] = {
        "verifier_output": verifier_result,
        "reward/base_reward": base_reward,
        "reward/tool_call_total": applied_penalty + applied_bonus,
        "reward/step_penalty": applied_penalty,
        "reward/tool_call_bonus": applied_bonus,
        "reward/raw_step_penalty": step_penalty,
        "reward/step_penalty_disabled": int(disable_step_penalty),
        "tool_call_reward": applied_penalty + applied_bonus,
        "tool_call_stats": tool_call_stats,
        "reward/total_clipped": base_reward > 0.0 and total_reward == 0.0 and (base_reward + step_penalty + tool_call_bonus) < 0.0,
        "verifier/structural_fail": structural_fail,
        "verifier/content_fail": content_fail,
        "verifier/coercion_retry": int(coercion_retry),
    }
    return RewardOutput(reward=total_reward, metadata=metadata, is_correct=is_correct)
