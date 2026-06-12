from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from rllm.harnesses.react import ReActHarness
from rllm.types import Task


def test_react_harness_default_prompt_uses_standard_react_format():
    harness = ReActHarness()

    assert "Follow the ReAct format strictly:" in harness.system_prompt
    assert "Thought: reason about what you know" in harness.system_prompt
    assert "finish tool" in harness.system_prompt
    assert "\\boxed{}" in harness.system_prompt


def test_react_harness_records_qwen_tool_calls_without_executable_tools(monkeypatch):
    class _FakeCompletions:
        def create(self, **kwargs):
            msg = SimpleNamespace(
                content='<think>Need weather.</think>\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    openai_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr("openai.OpenAI", openai_cls)

    task = Task(
        id="t",
        instruction="What is the weather?",
        metadata={"tools_json": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]},
        dataset_dir=Path("."),
    )
    config = SimpleNamespace(base_url="http://gateway", model="model", session_uid="t:0")

    episode = ReActHarness().run(task, config)

    assert episode.artifacts["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert episode.trajectories[0].steps[0].action == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
