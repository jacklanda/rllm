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
import uuid
from pathlib import Path
from typing import Any

from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call reward helpers
# ---------------------------------------------------------------------------

def _get_tool_call_reward(task_info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute step-penalty and tool-call bonus/penalty.

    Returns (total_tool_call_reward, stats_dict).

    The penalty is meant to encourage multi-step tool usage *without*
    completely overwhelming a positive base_reward.  Current scale:

    * No non-submit tool calls → -0.5 (was -1.0)
    * step_count ≤ 2           → -0.5 (was -2.0)
    * 3 ≤ step_count < 4       → mild ramp
    * 4..8                     → 0 (sweet spot)
    * >8                       → small penalty for over-stepping
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

    tool_call_bonus = 0.0
    if submit_called and non_submit_calls == 0:
        tool_call_bonus -= 0.5

    stats = dict(stats) if isinstance(stats, dict) else {}
    stats["step_penalty"] = float(step_penalty)
    stats["tool_call_bonus"] = float(tool_call_bonus)
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
    spec.loader.exec_module(module)

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
    tool_call_reward, tool_call_stats = _get_tool_call_reward(task_info)
    total_reward = base_reward + tool_call_reward

    step_penalty = float(tool_call_stats.get("step_penalty", 0.0))
    tool_call_bonus = float(tool_call_stats.get("tool_call_bonus", 0.0))

    metadata: dict[str, Any] = {
        "verifier_output": verifier_result,
        "reward/base_reward": base_reward,
        "reward/tool_call_total": tool_call_reward,
        "reward/step_penalty": step_penalty,
        "reward/tool_call_bonus": tool_call_bonus,
        "tool_call_reward": tool_call_reward,
        "tool_call_stats": tool_call_stats,
    }
    return RewardOutput(reward=total_reward, metadata=metadata, is_correct=is_correct)
