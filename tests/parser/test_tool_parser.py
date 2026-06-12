import json

import pytest

from rllm.parser import Qwen3CoderToolParser, QwenToolParser, R1ToolParser, get_tool_parser
from rllm.parser.chat_template_parser import QwenChatTemplateParser
from rllm.parser.tool_parser import ToolParser
from rllm.tools.tool_base import ToolCall


class TestQwenToolParser:
    @pytest.fixture
    def parser(self):
        return QwenToolParser()

    def test_empty_response(self, parser):
        """Test parsing empty response."""
        result = parser.parse("")
        assert len(result) == 0

    def test_no_tool_calls(self, parser):
        """Test response with no tool calls."""
        response = "This is a normal response without any tool calls."
        result = parser.parse(response)
        assert len(result) == 0

    def test_single_valid_tool_call(self, parser):
        """Test parsing a single valid tool call."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York", "unit": "celsius"}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].name == "search_weather"
        assert result[0].arguments == {"location": "New York", "unit": "celsius"}

    def test_multiple_valid_tool_calls(self, parser):
        """Test parsing multiple valid tool calls."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York"}}</tool_call>
        <tool_call>{"name": "search_restaurants", "arguments": {"location": "New York", "cuisine": "Italian"}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 2
        assert result[0].name == "search_weather"
        assert result[1].name == "search_restaurants"

    def test_invalid_json_tool_call(self, parser):
        """Test parsing tool call with invalid JSON."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {invalid json}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_tool_call_end(self, parser):
        """Test parsing tool call with missing end tag."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York"}}
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "search_weather"
        assert result[0].arguments == {"location": "New York"}

    def test_get_tool_prompt(self, parser):
        """Test tool prompt generation."""
        tools_schema = """
        {
            "name": "search_weather",
            "description": "Search for weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
        """
        prompt = parser.get_tool_prompt(tools_schema)
        assert "<tools>" in prompt
        assert tools_schema in prompt
        assert "<tool_call>" in prompt

    def test_qwen3_thinking_tool_call(self, parser):
        """Qwen3 thinking output keeps tool calls after </think>."""
        response = """
        <think>
        I need to search for evidence.
        </think>
        <tool_call>
        {"name": "web_search", "arguments": {"query": "Qwen3 tool calling"}}
        </tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "web_search"
        assert result[0].arguments == {"query": "Qwen3 tool calling"}

    def test_qwen35_bare_json_after_thinking(self, parser):
        """Qwen3.5-style completions may emit bare JSON after </think>."""
        response = """
        <think>Need the final tool call.</think>
        {"name": "finish", "arguments": {"command": "submit", "result": "Paris"}}
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "finish"
        assert result[0].arguments == {"command": "submit", "result": "Paris"}

    def test_boxed_fallback_uses_last_boxed_after_thinking(self, parser):
        """Intermediate boxed values should not override the final answer."""
        response = r"""
        <think>I compute an intermediate value \boxed{6.3 \times 10^{-7}}.</think>
        The matching option is \boxed{A}.
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "finish"
        assert result[0].arguments == {"command": "submit", "result": "A"}

    def test_answer_tag_fallback_takes_precedence_over_boxed(self, parser):
        """Task-specific <answer> tags outrank generic boxed output."""
        response = r"""
        <think>I first considered \boxed{Calvin Coolidge}.</think>
        Final: \boxed{James Madison}
        <answer>james madison</answer>
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "finish"
        assert result[0].arguments == {"command": "submit", "result": "james madison"}

    def test_boxed_fallback_handles_nested_last_boxed(self, parser):
        response = r"<think>calc \boxed{0.64}</think> Therefore \boxed{\text{C}}"
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].arguments["result"] == r"\text{C}"

    def test_chat_template_formats_json_tool_calls_for_qwen3(self):
        class DummyTokenizer:
            name_or_path = "Qwen/Qwen3-8B"
            bos_token = None
            eos_token = "<|im_end|>"

            def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
                rendered = "".join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages)
                if add_generation_prompt:
                    rendered += "<|im_start|>assistant\n"
                return rendered

        parser = QwenChatTemplateParser(DummyTokenizer())
        rendered = parser.parse(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [ToolCall(name="get_weather", arguments={"location": "Beijing"})],
                }
            ]
        )
        assert '"name": "get_weather"' in rendered
        assert '"location": "Beijing"' in rendered
        assert "<function=get_weather>" not in rendered


class TestR1ToolParser:
    @pytest.fixture
    def parser(self):
        return R1ToolParser()

    def test_empty_response(self, parser):
        """Test parsing empty response."""
        result = parser.parse("")
        assert len(result) == 0

    def test_no_tool_calls(self, parser):
        """Test response with no tool calls."""
        response = "This is a normal response without any tool calls."
        result = parser.parse(response)
        assert len(result) == 0

    def test_single_valid_tool_call(self, parser):
        """Test parsing a single valid tool call."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York", "unit": "celsius"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].name == "search_weather"
        assert result[0].arguments == {"location": "New York", "unit": "celsius"}

    def test_multiple_valid_tool_calls(self, parser):
        """Test parsing multiple valid tool calls."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York"}}
        ```
        {parser.tool_call_end}
        {parser.tool_call_begin}function{parser.tool_sep}search_restaurants
        ```json
        {{"location": "New York", "cuisine": "Italian"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 2
        assert result[0].name == "search_weather"
        assert result[1].name == "search_restaurants"

    def test_invalid_json_tool_call(self, parser):
        """Test parsing tool call with invalid JSON."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York", invalid json}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_tool_call_end(self, parser):
        """Test parsing tool call with missing end tag."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York"}}
        ```
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_function_prefix(self, parser):
        """Test parsing tool call with missing function prefix."""
        response = f"""
        {parser.tool_call_begin}search_weather
        ```json
        {{"location": "New York"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_json_block(self, parser):
        """Test parsing tool call with missing JSON block."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0


class TestQwen3CoderToolParser:
    @pytest.fixture
    def parser(self):
        return Qwen3CoderToolParser()

    def test_single_xml_function_call(self, parser):
        response = """<tool_call>
<function=get_weather>
<parameter=location>
Beijing
</parameter>
</function>
</tool_call>"""
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].arguments == {"location": "Beijing"}

    def test_multiple_xml_parameters(self, parser):
        response = """<tool_call>
<function=search_flights>
<parameter=from_city>
Beijing
</parameter>
<parameter=to_city>
Paris
</parameter>
<parameter=date>
2026-09-25
</parameter>
</function>
</tool_call>"""
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].name == "search_flights"
        assert result[0].arguments == {
            "from_city": "Beijing",
            "to_city": "Paris",
            "date": "2026-09-25",
        }

    def test_schema_driven_parameter_conversion(self, parser):
        tools_schema = json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer"},
                                "rerank": {"type": "boolean"},
                                "filters": {"type": "object"},
                            },
                        },
                    },
                }
            ]
        )
        parser.get_tool_prompt(tools_schema)
        response = """<tool_call>
<function=search>
<parameter=query>
qwen3.5
</parameter>
<parameter=top_k>
5
</parameter>
<parameter=rerank>
true
</parameter>
<parameter=filters>
{"source": "docs"}
</parameter>
</function>
</tool_call>"""
        result = parser.parse(response)
        assert len(result) == 1
        assert result[0].arguments == {
            "query": "qwen3.5",
            "top_k": 5,
            "rerank": True,
            "filters": {"source": "docs"},
        }

    def test_multiple_tool_calls_and_missing_end_tag(self, parser):
        response = """<tool_call>
<function=get_weather>
<parameter=location>
Beijing
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
</function>"""
        result = parser.parse(response)
        assert len(result) == 2
        assert result[0].arguments == {"location": "Beijing"}
        assert result[1].arguments == {"location": "Paris"}

    def test_prompt_uses_qwen35_xml_format(self, parser):
        prompt = parser.get_tool_prompt('{"type":"function","function":{"name":"get_weather","parameters":{"properties":{"location":{"type":"string"}}}}}')
        assert "<function=FUNCTION_NAME>" in prompt
        assert "<parameter=PARAMETER_NAME>" in prompt
        assert "Do not put JSON tool-call objects inside <tool_call>" in prompt

    def test_registry_name(self):
        assert get_tool_parser("qwen3_coder") is Qwen3CoderToolParser

    def test_auto_parser_selection_for_qwen35(self):
        class DummyTokenizer:
            name_or_path = "Qwen/Qwen3.5-30B-A3B-Instruct"

        assert isinstance(ToolParser.get_parser(DummyTokenizer()), Qwen3CoderToolParser)

    def test_chat_template_formats_xml_tool_calls_for_qwen35(self):
        class DummyTokenizer:
            name_or_path = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
            bos_token = None
            eos_token = "<|im_end|>"

            def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
                rendered = "".join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in messages)
                if add_generation_prompt:
                    rendered += "<|im_start|>assistant\n"
                return rendered

        parser = QwenChatTemplateParser(DummyTokenizer())
        rendered = parser.parse(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [ToolCall(name="get_weather", arguments={"location": "Beijing"})],
                }
            ]
        )
        assert "<function=get_weather>" in rendered
        assert "<parameter=location>\nBeijing\n</parameter>" in rendered
        assert '{"name":' not in rendered
