from pathlib import Path

import pytest
from transformers import AutoTokenizer

from rllm.agents.fused_agent import FusedAgent
from rllm.agents.system_prompts import (
    COT_SYSTEM_PROMPT,
    COT_USER_PROMPT,
    FUSED_AGENT_SYSTEM_PROMPT,
    FUSED_MCP_SYSTEM_PROMPT,
    FUSED_SEARCH_SYSTEM_PROMPT,
    FUSED_UNIFIED_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT,
)
from rllm.parser import ChatTemplateParser
from rllm.parser.tool_parser import Qwen3CoderToolParser, QwenToolParser


def test_fused_agent_unified_gem_harness_uses_unified_prompt():
    agent = FusedAgent(harness="unified_gem")

    assert agent.harness == "unified_gem"
    assert FUSED_UNIFIED_SYSTEM_PROMPT.splitlines()[0] in agent.system_prompt


def test_fused_agent_gem_harness_uses_task_specific_prompt():
    agent = FusedAgent(harness="gem")

    assert agent.harness == "gem"
    assert isinstance(agent.tool_parser, QwenToolParser)
    assert FUSED_AGENT_SYSTEM_PROMPT.splitlines()[0] in agent.system_prompt
    assert FUSED_UNIFIED_SYSTEM_PROMPT.splitlines()[0] not in agent.system_prompt


def test_fused_agent_qwen35_model_uses_qwen3_coder_tool_format():
    agent = FusedAgent(harness="gem", model_name="/share/nlp/share/plm/Qwen3.5-4B")

    assert isinstance(agent.tool_parser, Qwen3CoderToolParser)
    assert "<function=FUNCTION_NAME>" in agent.system_prompt
    assert "<parameter=PARAMETER_NAME>" in agent.system_prompt
    assert '{"name": <function-name>' not in agent.system_prompt


def test_fused_agent_qwen35_gem_harnesses_use_xml_tool_prompts_for_all_task_types():
    model_path = "/share/nlp/share/plm/Qwen3.5-4B"
    mcp_tool = {
        "type": "function",
        "function": {
            "name": "lookup_record",
            "description": "Look up a record.",
            "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]},
        },
    }

    for harness in ("gem", "unified_gem"):
        for info in (
            {"task_type": "web search"},
            {"task_type": "mcp", "tools_json": [mcp_tool]},
            {"task_type": "et"},
        ):
            agent = FusedAgent(harness=harness, model_name=model_path)
            agent.update_from_env("Do the task.", 0.0, False, info)

            assert isinstance(agent.tool_parser, Qwen3CoderToolParser)
            assert "<function=FUNCTION_NAME>" in agent.messages[0]["content"]
            assert "<parameter=PARAMETER_NAME>" in agent.messages[0]["content"]
            assert '{"name": <function-name>' not in agent.messages[0]["content"]


def test_train_fused_agent_script_passes_qwen35_harness_model_and_thinking_config():
    script = Path(__file__).resolve().parents[2] / "experiments/fused/train_fused_agent.sh"
    content = script.read_text()

    assert "--harness NAME" in content
    assert "--model PATH" in content
    assert "--disable-thinking BOOL" in content
    assert "rllm.disable_thinking=${disable_thinking}" in content
    assert "+rllm.agent.agent_args.harness=${harness}" in content
    assert "+rllm.agent.agent_args.model_name=${model_path}" in content
    assert "+rllm.env.env_args.harness=${harness}" in content


def test_fused_evals_script_passes_harness_to_env_and_agent():
    script = Path(__file__).resolve().parents[2] / "experiments/fused/evals.sh"
    content = script.read_text()

    assert "+rllm.agent.agent_args.harness=${harness}" in content
    assert "+rllm.env.env_args.harness=${harness}" in content


def test_fused_react_qwen35_prompt_enables_thinking():
    model_path = "/share/nlp/share/plm/Qwen3.5-4B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")

    agent = FusedAgent(harness="react", model_name=model_path)
    agent.update_from_env("Find the answer.", 0.0, False, {"task_type": "web search", "max_steps": 3})

    parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=False)
    prompt = parser.parse(agent.chat_completions, add_generation_prompt=True, is_first_msg=True)

    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert parser.enable_thinking is True


def test_fused_agent_react_harness_uses_react_prompt_and_user_template():
    agent = FusedAgent(harness="react")
    agent.update_from_env("Find the answer.", 0.0, False, {"task_type": "web search", "max_steps": 3})

    assert agent.harness == "react"
    assert REACT_SYSTEM_PROMPT.splitlines()[0] in agent.messages[0]["content"]
    assert agent.user_prompt_template == REACT_USER_PROMPT
    assert "Use the ReAct loop" in agent.messages[-1]["content"]


def test_fused_agent_cot_harness_uses_no_tool_prompt_or_actions():
    agent = FusedAgent(harness="cot")
    agent.update_from_env("What is 2+2?", 0.0, False, {"task_type": "web search", "max_steps": 3})

    assert agent.harness == "cot"
    assert agent.messages[0]["content"] == COT_SYSTEM_PROMPT
    assert agent.user_prompt_template == COT_USER_PROMPT
    assert "# Tools" not in agent.messages[0]["content"]
    assert agent.messages[-1]["content"].startswith("What is 2+2?")

    actions = agent.update_from_model('<tool_call>{"name": "web_search", "arguments": {"query": "2+2"}}</tool_call>')
    assert actions[0].action == ""
    assert agent.trajectory.steps[-1].info["cot_suppressed_tool_action"] is True


def test_fused_agent_cot_harness_keeps_plain_text_answer_action():
    agent = FusedAgent(harness="cot")
    agent.update_from_env("What is 2+2?", 0.0, False, {"task_type": "web search", "max_steps": 3})

    response = "The answer is 4."
    actions = agent.update_from_model(response)

    assert actions[0].action == response
    assert agent.trajectory.steps[-1].action == response
    assert agent.trajectory.steps[-1].info["cot_implicit_answer"] is True


def test_fused_agent_cot_harness_keeps_answer_tags_as_plain_text():
    agent = FusedAgent(harness="cot")
    agent.update_from_env("Pick A or B.", 0.0, False, {"task_type": "web search", "max_steps": 3})

    response = "Reasoning...\n<answer>B</answer>\n\\boxed{B}"
    actions = agent.update_from_model(response)

    assert actions[0].action == response
    assert "<function=finish>" not in actions[0].action
    assert agent.trajectory.steps[-1].info["cot_implicit_answer"] is True


def test_fused_agent_bare_harness_uses_user_prompt_without_system_prompt():
    agent = FusedAgent(harness="bare")
    agent.update_from_env("What is 2+2?", 0.0, False, {"task_type": "web search", "max_steps": 3})

    assert agent.harness == "bare"
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[0]["content"].startswith("What is 2+2?")
    assert all(message["role"] != "system" for message in agent.messages)
    assert agent.user_prompt_template == COT_USER_PROMPT

    actions = agent.update_from_model('<tool_call>{"name": "web_search", "arguments": {"query": "2+2"}}</tool_call>')
    assert actions[0].action == ""
    assert agent.trajectory.steps[-1].info["bare_suppressed_tool_action"] is True


def test_fused_agent_gem_selects_task_specific_prompts_by_task_type():
    search_agent = FusedAgent(harness="gem")
    search_agent.update_from_env("Who?", 0.0, False, {"task_type": "web search"})
    assert FUSED_SEARCH_SYSTEM_PROMPT.splitlines()[0] in search_agent.messages[0]["content"]

    mcp_agent = FusedAgent(harness="gem")
    mcp_agent.update_from_env("Fetch data.", 0.0, False, {"task_type": "mcp", "tools_json": []})
    assert FUSED_MCP_SYSTEM_PROMPT.splitlines()[0] in mcp_agent.messages[0]["content"]


def test_fused_agent_rejects_unknown_harness():
    with pytest.raises(ValueError, match="Invalid fused harness"):
        FusedAgent(harness="unknown")
