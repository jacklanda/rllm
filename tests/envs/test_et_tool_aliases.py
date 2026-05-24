"""Test the ``file_editor → str_replace_editor`` alias in ETEnv.

Why: 4B-Thinking emits ``<function=file_editor>`` ~7% of the time even
though the ET prompt only documents ``str_replace_editor``. The two tool
scripts are interchangeable for the view/create/str_replace/insert
subset ET tasks use, so the env aliases rather than rejecting.

These tests don't spin up Docker — they exercise the dispatch logic
directly via ``_build_tool_argv`` plus a stubbed ``step()`` entry point.
"""

from __future__ import annotations

import pytest

pytest.importorskip("r2egym")

from r2egym.agenthub.action import Action as SWEAction

from rllm.environments.endless_terminals.et_env import ETEnv


def test_argv_builder_does_not_know_file_editor():
    """Sanity: argv builder only knows the canonical name. Alias must happen
    in dispatch, not in argv building."""
    assert ETEnv._build_tool_argv("file_editor", {"command": "view", "path": "/x"}) is None
    assert ETEnv._build_tool_argv("str_replace_editor", {"command": "view", "path": "/x"}) == [
        "str_replace_editor",
        "view",
        "--path",
        "/x",
    ]


def test_alias_dispatch_does_not_return_unknown_function(monkeypatch):
    """When the model emits ``file_editor``, ETEnv.step must not return the
    'Unknown function' observation. We stub the container exec layer so the
    test runs without Docker."""

    env = ETEnv.__new__(ETEnv)
    env.task_id = "t-alias"
    env.total_steps = 0
    env.entry = {}
    env.step_timeout = 1
    env.verbose = False
    env.container = object()  # truthy sentinel; the real exec is stubbed below.

    captured = {}

    def fake_exec(container, argv, workdir=None, timeout=None):
        captured["argv"] = argv
        return (0, "ok")

    monkeypatch.setattr("rllm.environments.endless_terminals.et_env._exec", fake_exec)

    action = SWEAction(function_name="file_editor", parameters={"command": "view", "path": "/home/user"})
    obs, reward, done, info = env.step(action)

    assert "Unknown function" not in obs, f"file_editor should be aliased, got: {obs!r}"
    assert captured.get("argv") == ["str_replace_editor", "view", "--path", "/home/user"], f"alias should resolve to str_replace_editor argv, got: {captured.get('argv')!r}"
    assert reward == 0.0
    assert done is False


def test_supported_functions_unchanged():
    """The tuple stays canonical-only so adding new aliases stays explicit."""
    assert ETEnv.SUPPORTED_FUNCTIONS == ("execute_bash", "str_replace_editor", "submit", "finish")
    assert "file_editor" not in ETEnv.SUPPORTED_FUNCTIONS
