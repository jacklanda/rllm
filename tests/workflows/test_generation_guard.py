from rllm.environments.fused.fused import FusedEnv
from rllm.workflows.workflow import Workflow


def test_generation_guard_detects_tool_schema_phrase_loop():
    text = ("The tool call is a function that takes a JSON object as input. " * 10).strip()

    bad, reason = Workflow.detect_abnormal_generation(text)

    assert bad is True
    assert "tool call" in reason


def test_generation_guard_detects_repeated_symbol_loop():
    text = "normal preface " + ("}" * 100)

    bad, reason = Workflow.detect_abnormal_generation(text)

    assert bad is True
    assert "symbol" in reason


def test_generation_guard_allows_ordinary_long_text():
    text = " ".join(f"evidence_{i}" for i in range(300))

    bad, reason = Workflow.detect_abnormal_generation(text)

    assert bad is False
    assert reason == ""


def test_fused_env_runaway_detector_uses_same_phrase_loop_signal():
    text = ("The user message is a question about the answer. " * 10).strip()

    bad, reason = FusedEnv._detect_runaway(text)

    assert bad is True
    assert "user message" in reason
