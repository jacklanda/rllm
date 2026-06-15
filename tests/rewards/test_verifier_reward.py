import logging

from rllm.rewards.verifier_reward import _load_tools_from_tools_py, verifier_reward_fn


def test_load_tools_from_tools_py_silences_mcp_registration_warning(tmp_path, caplog):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "import logging\n"
        "logging.getLogger('mcp.server.fastmcp.tools.tool_manager').warning('Tool already exists: duplicate_tool')\n"
        "def duplicate_tool():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        tools = _load_tools_from_tools_py(str(tools_py))

    assert "duplicate_tool" in tools
    assert tools["duplicate_tool"]() == {"result": "ok"}
    assert "Tool already exists" not in caplog.text


def test_mcp_step_penalty_can_be_disabled_for_offline_rs(monkeypatch):
    task_info = {
        "verifier": {"verification_code": "def verify(tools, answer):\n    return {'passed': True}\n"},
        "tool_call_stats": {
            "submit_called": True,
            "non_submit_tool_calls": 19,
            "step_count": 20,
            "distinct_successful_tools": 8,
        },
    }

    monkeypatch.delenv("RLLM_MCP_DISABLE_STEP_PENALTY", raising=False)
    default_result = verifier_reward_fn(task_info=task_info, action="{}")
    assert default_result.reward == 0.0
    assert default_result.metadata["reward/raw_step_penalty"] == -2.4000000000000004
    assert default_result.metadata["reward/step_penalty_disabled"] == 0

    monkeypatch.setenv("RLLM_MCP_DISABLE_STEP_PENALTY", "True")
    rs_result = verifier_reward_fn(task_info=task_info, action="{}")
    assert rs_result.reward == 1.0
    assert rs_result.metadata["reward/raw_step_penalty"] == -2.4000000000000004
    assert rs_result.metadata["reward/step_penalty"] == 0.0
    assert rs_result.metadata["reward/step_penalty_disabled"] == 1
