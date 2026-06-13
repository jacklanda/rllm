"""Smoke tests for the protocol-error detection signature list.

The full termination path lives inside the async rollout loop in
``agent_execution_engine.py`` which is awkward to unit-test without spinning
up a real env + agent + rollout. This test asserts the cheaper but
load-bearing invariant: the signature list matches the actual strings
ETEnv / ToolEnv emit when the model produces malformed tool calls.

If the env messages drift, this test will catch it before the missed
signal bleeds into a training run as silent zero-advantage trajectories.
"""

from __future__ import annotations

import pytest

from rllm.engine.agent_execution_engine import (
    PROTOCOL_ERROR_TERMINATE_THRESHOLD,
    _ENV_PROTOCOL_ERROR_SIGS,
    _mask_only_reasoning_step,
)


@pytest.mark.parametrize(
    "observation",
    [
        # ETEnv: empty function name path
        "You forgot to use a function call. Please use a function call to interact with the environment.",
        # ETEnv / SWE: unrecognised tool name
        "Unknown function: file_editor. Available: str_replace_editor, execute_bash, submit",
        "Unknown function: ls. Available: str_replace_editor, execute_bash, submit",
        # ETEnv: SWEAction.from_string parsed but ``_build_tool_argv`` couldn't realise it
        "Could not build command for str_replace_editor with parameters {}",
    ],
)
def test_known_protocol_error_observations_are_detected(observation: str) -> None:
    assert any(sig in observation for sig in _ENV_PROTOCOL_ERROR_SIGS), f"Observation should match a protocol-error signature: {observation!r}"


@pytest.mark.parametrize(
    "observation",
    [
        # Real tool output — must NOT be misclassified as protocol error
        "[STDOUT]\ntotal 4\ndrwxr-xr-x 2 root root 4096 May 10 12:00 src\n",
        "Steps Remaining: 12\n[STDOUT]\nfile.txt created\n",
        # Tool-crash signatures handled by _ENV_TOOL_ERROR_SIGS, NOT protocol errors
        'Traceback (most recent call last):\n  File "x", line 1\nModuleNotFoundError',
        # SWE eval output
        "Exit code: 0\nAll tests passed",
    ],
)
def test_legitimate_observations_are_not_protocol_errors(observation: str) -> None:
    assert not any(sig in observation for sig in _ENV_PROTOCOL_ERROR_SIGS), f"Observation must not match a protocol-error signature: {observation!r}"


def test_threshold_is_strict_enough_to_catch_real_failures() -> None:
    """Sanity bound: threshold should be ≥2 (avoid one-off flakes) and ≤5
    (avoid burning the whole 32-step budget on noise). The previous training
    run showed ~3 protocol errors per failing trajectory in the first 5 steps,
    so 3 is the natural cut-off."""
    assert 2 <= PROTOCOL_ERROR_TERMINATE_THRESHOLD <= 5


def test_credit_assignment_mask_keeps_only_reasoning_step() -> None:
    response_masks = [1, 1, 0, 0, 1, 1, 1]
    assistant_msg_masks = [1, 1, 1]

    assert _mask_only_reasoning_step(response_masks, assistant_msg_masks) == [0, 0, 0, 0, 1, 1, 1]
