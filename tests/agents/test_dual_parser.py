"""Tests for the dual SWE-XML / Qwen ``<tool_call>`` parser used by ETAgent."""

from __future__ import annotations

import pytest

pytest.importorskip("r2egym")

from rllm.agents._dual_parser import parse_dual_response


def test_pure_swe_xml():
    resp = "I should view the file.\n" "<function=execute_bash>\n" "<parameter=cmd>ls /home/user</parameter>\n" "</function>"
    thought, action = parse_dual_response(resp)
    assert thought == "I should view the file."
    assert action.function_name == "execute_bash"
    assert action.parameters == {"cmd": "ls /home/user"}


def test_pure_tool_call_json_dict_arguments():
    resp = "<think>plan</think>\n" '<tool_call>\n{"name": "str_replace_editor", ' '"arguments": {"command": "view", "path": "/home/user/x"}}\n</tool_call>'
    thought, action = parse_dual_response(resp)
    assert "<think>plan</think>" in thought
    assert action.function_name == "str_replace_editor"
    assert action.parameters == {"command": "view", "path": "/home/user/x"}


def test_tool_call_arguments_as_json_string():
    """Some Qwen variants emit ``arguments`` as a JSON string, not a dict."""
    resp = '<tool_call>{"name": "execute_bash", ' '"arguments": "{\\"cmd\\": \\"echo hi\\"}"}</tool_call>'
    _, action = parse_dual_response(resp)
    assert action.function_name == "execute_bash"
    assert action.parameters == {"cmd": "echo hi"}


def test_tool_call_with_non_string_scalar_arguments():
    resp = '<tool_call>{"name": "execute_bash", ' '"arguments": {"cmd": "ls", "timeout": 30, "concise": true}}</tool_call>'
    _, action = parse_dual_response(resp)
    assert action.function_name == "execute_bash"
    assert action.parameters["cmd"] == "ls"
    assert action.parameters["timeout"] == "30"
    assert action.parameters["concise"] == "True"


def test_xml_takes_precedence_over_tool_call_when_both_present():
    """If a model accidentally emits both, XML wins (it is the documented protocol)."""
    resp = "<function=execute_bash><parameter=cmd>echo xml</parameter></function>" '<tool_call>{"name": "execute_bash", "arguments": {"cmd": "echo json"}}</tool_call>'
    _, action = parse_dual_response(resp)
    assert action.parameters == {"cmd": "echo xml"}


def test_truncated_tool_call_recovers_function_name():
    """max_tokens cut mid-JSON: best-effort recovery of name + partial args."""
    resp = '<tool_call>{"name": "execute_bash", "arguments": {"cmd": "sleep 1'
    _, action = parse_dual_response(resp)
    assert action.function_name == "execute_bash"


def test_no_tool_call_returns_empty_action():
    resp = "I will think about this and respond in plain text."
    thought, action = parse_dual_response(resp)
    assert thought == resp
    assert action.function_name == ""
    assert action.parameters == {}


def test_truncated_xml_is_repaired():
    """SWE-XML truncated by max_tokens: function name must be recovered.

    Parameter values cannot be recovered (``Action.from_string`` requires
    ``</parameter>``) — this matches existing ``parse_xml_response`` behaviour.
    """
    resp = "<function=execute_bash>\n<parameter=cmd>echo hello"
    _, action = parse_dual_response(resp)
    assert action.function_name == "execute_bash"


def test_alternate_keys_function_and_parameters():
    """Some models emit ``function``/``parameters`` instead of ``name``/``arguments``."""
    resp = '<tool_call>{"function": "submit", "parameters": {}}</tool_call>'
    _, action = parse_dual_response(resp)
    assert action.function_name == "submit"
    assert action.parameters == {}
