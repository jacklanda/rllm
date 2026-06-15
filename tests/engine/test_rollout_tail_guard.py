import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rllm.engine.agent_execution_engine import RolloutTailGuardState, _config_bool


def test_config_bool_handles_string_false():
    assert _config_bool("False") is False
    assert _config_bool("true") is True
    assert _config_bool(None, default=True) is True


def test_rollout_tail_guard_enters_tail_phase_after_threshold():
    guard = RolloutTailGuardState(total=5, enabled=True)

    assert guard.is_tail_phase() is False

    guard.record_result({"termination_reason": "ENV_DONE"})
    guard.record_result({"termination_reason": "TRUNCATION"})

    assert guard.is_tail_phase() is False

    guard.record_result({"termination_reason": "ENV_DONE"})

    assert guard.is_tail_phase() is False

    guard.record_result({"termination_reason": "ENV_DONE"})

    assert guard.is_tail_phase() is True
    assert guard.reason_counts["ENV_DONE"] == 3
    assert guard.reason_counts["TRUNCATION"] == 1


def test_rollout_tail_guard_disabled_never_enters_tail_phase():
    guard = RolloutTailGuardState(total=1, enabled=False)
    guard.record_result({"termination_reason": "TRUNCATION"})

    assert guard.is_tail_phase() is False


def test_rollout_tail_guard_detects_overlong_burst_only_after_tail_phase():
    guard = RolloutTailGuardState(total=10, enabled=True)

    for _ in range(7):
        guard.record_result({"termination_reason": "MAX_PROMPT_LENGTH_EXCEEDED"})

    assert guard.is_tail_phase() is False
    assert guard.has_overlong_burst() is False

    guard.record_result({"termination_reason": "ENV_DONE"})

    assert guard.is_tail_phase() is True
    assert guard.has_overlong_burst() is True

    metrics = guard.metrics(elapsed_s=10.0)
    assert metrics["tail_guard_observed_overlong"] == 7.0
    assert metrics["tail_guard_overlong_burst"] == 1.0
    assert metrics["tail_guard_overlong_burst_elapsed_seconds"] >= 0.0
    assert metrics["tail_guard_recent_overlong_ratio"] == 0.875
