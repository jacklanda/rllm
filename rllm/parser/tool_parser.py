import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]:
        """Extract tool calls from the model response."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_parser(cls, tokenizer) -> "ToolParser":
        """Factory method to get the appropriate tool parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser

        Returns:
            ToolParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using R1ToolParser for {tokenizer.name_or_path}")
                return R1ToolParser()
            elif any(x in model_name for x in ("qwen3.5", "qwen3-coder", "qwen3coder")):
                print(f"Using Qwen3CoderToolParser for {tokenizer.name_or_path}")
                return Qwen3CoderToolParser()
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenToolParser for {tokenizer.name_or_path}")
                return QwenToolParser()
        # TODO: add verfication to check equivalence of the parser with that from HuggingFace
        raise ValueError(f"No tool parser found for {tokenizer.name_or_path}")


class R1ToolParser(ToolParser):
    """Parser for R1 tool call format."""

    def __init__(self):
        """Initialize the R1 tool parser.

        Args:
            model (str): Model name for tokenizer (optional)
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        self.tool_output_begin = "<｜tool▁response▁begin｜>"
        self.tool_output_end = "<｜tool_response_end｜>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_r1_tool_calls(model_response)

        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def parse_r1_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from text using the R1 special token format.

        Format:
        <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
        ```json
        {"param1": "value1", "param2": "value2"}
        ```
        <｜tool▁call▁end｜>
        // Additional tool calls follow the same format
        <｜tool▁calls▁end｜>

        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []

        # Look for individual tool calls
        call_idx = 0
        while True:
            # Find the next tool call beginning
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            # Find the end of this tool call
            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            # Extract the content of this tool call
            call_content = text[call_start:call_end].strip()

            # Parse function name
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)

            if func_start != -1:
                # Extract function name after the prefix up to the next newline
                func_name_start = func_start + len(func_prefix)
                func_name_end = call_content.find("\n", func_name_start)

                if func_name_end == -1:
                    function_name = call_content[func_name_start:].strip()
                else:
                    function_name = call_content[func_name_start:func_name_end].strip()
            else:
                # If function prefix not found, skip this call
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Extract JSON arguments
            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")

            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            args_str = call_content[json_start:json_end].strip()

            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Add this tool call to our list
            tool_calls.append({"name": function_name, "arguments": args_json})

            # Move past this call for the next iteration
            call_idx = call_end + len(self.tool_call_end)

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.
<tools>
{tools_schema}
</tools>

Output format for tool calls:

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
```json
{{"param1": "value1", "param2": "value2"}}
```
<｜tool▁call▁end｜>
// Additional tool calls follow the same format
<｜tool▁calls▁end｜>
"""


class QwenToolParser(ToolParser):
    # Alternate tag names that models commonly produce instead of <tool_call>
    _ALTERNATE_TAGS = [
        ("<function_call>", "</function_call>"),
    ]

    # Common hallucinated tool names → likely intended tool
    _TOOL_NAME_ALIASES: dict[str, str] = {
        "function": "web_search",
        "function_name": "web_search",
        "function_call_1": "web_search",
        "functions": "web_search",
        "file": "file_editor",
        "file_edit": "file_editor",
        "file_view": "file_editor",
        "file_viewer": "file_editor",
        "file_str": "file_editor",
        "file_content": "file_editor",
        "file_contents": "file_editor",
        "file_writectl": "file_editor",
        "str_replace": "file_editor",
        "edit_file": "file_editor",
        "view": "file_editor",
        "explore": "file_editor",
        "execute": "execute_bash",
        "run": "execute_bash",
        "python": "execute_bash",
        "python-bash": "execute_bash",
        "mkdir": "execute_bash",
        "tool_call": "web_search",
        "tool_calls": "web_search",
        "tool": "web_search",
        "submit": "finish",
        "test": "execute_bash",
        "search_result": "search",
    }

    def __init__(self, valid_tools: set[str] | None = None):
        """Initialize the parser with specified type and model.

        Args:
            valid_tools: Optional set of valid tool names for this task context.
                         When provided, hallucinated names are mapped to the
                         closest valid tool via _TOOL_NAME_ALIASES.
        """
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"
        self.valid_tools = valid_tools

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_qwen_tool_calls(model_response)
        # Normalize hallucinated tool names if valid_tools is configured
        if self.valid_tools:
            for tc in tool_calls_dicts:
                tc["name"] = self._normalize_tool_name(tc["name"], self.valid_tools)
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def format_tool_call(self, tool_call_dict: dict[str, Any]) -> str:
        arguments_obj = tool_call_dict.get("arguments")
        if isinstance(arguments_obj, str):
            try:
                arguments_obj = json.loads(arguments_obj)
            except json.JSONDecodeError:
                pass

        tool_call_for_dump = dict(tool_call_dict)
        if arguments_obj is not None:
            tool_call_for_dump["arguments"] = arguments_obj
        return f"{self.tool_call_begin}\n{json.dumps(tool_call_for_dump, indent=0, ensure_ascii=False)}\n{self.tool_call_end}"

    @classmethod
    def _normalize_tool_name(cls, name: str, valid_tools: set[str]) -> str:
        """Map a possibly-hallucinated tool name to a valid one.

        Returns the original name unchanged if it's already valid or
        no confident mapping exists.
        """
        if name in valid_tools:
            return name
        # Check alias table
        alias = cls._TOOL_NAME_ALIASES.get(name)
        if alias and alias in valid_tools:
            return alias
        # Substring match: if name is a prefix/substring of exactly one valid tool
        matches = [t for t in valid_tools if name in t or t in name]
        if len(matches) == 1:
            return matches[0]
        return name

    # ------------------------------------------------------------------
    # Internal: parse a single JSON blob into {"name": ..., "arguments": ...}
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        """Extract the first balanced ``{...}`` JSON object from *text*.

        Uses simple brace-depth counting (ignoring braces inside strings for
        speed — LLM tool calls rarely have deeply nested quoted braces).
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # Unbalanced — return from start to end (caller will try to repair)
        return text[start:]

    @staticmethod
    def _try_parse_json(json_content: str) -> dict[str, Any] | None:
        """Attempt to parse *json_content* into a tool-call dict.

        Handles:
        - Standard format: ``{"name": "fn", "arguments": {...}}``
        - Flat format (no ``arguments`` wrapper): ``{"name": "fn", "key": "val", ...}``
        - Incomplete JSON (missing closing braces / trailing commas)
        - JSON followed by trailing non-JSON text
        """
        json_content = json_content.strip()
        if not json_content:
            return None

        # --- Attempt 1: strict parse of the full content ---
        call_data = None
        try:
            call_data = json.loads(json_content)
        except (json.JSONDecodeError, ValueError):
            pass

        # --- Attempt 2: extract the first balanced JSON object ---
        # (handles trailing non-JSON text after the object)
        if call_data is None:
            first_obj = QwenToolParser._extract_first_json_object(json_content)
            if first_obj:
                try:
                    call_data = json.loads(first_obj)
                except (json.JSONDecodeError, ValueError):
                    pass

        # --- Attempt 3: fix common LLM typos (trailing comma, missing braces, missing commas) ---
        if call_data is None:
            first_obj = QwenToolParser._extract_first_json_object(json_content) or json_content
            repaired = first_obj.rstrip()
            # Strip trailing commas before closing braces
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            # Insert missing commas between }"key" or value\n"key" patterns
            repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
            repaired = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', repaired)
            repaired = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', repaired)
            repaired = re.sub(r'\}\s*\n\s*"', '},\n"', repaired)
            # Try appending missing closing braces (up to 3)
            for suffix in ("}", "}}", "}}}", ""):
                candidate = repaired + suffix if suffix else repaired
                try:
                    call_data = json.loads(candidate)
                    break
                except (json.JSONDecodeError, ValueError):
                    continue

        if not isinstance(call_data, dict) or "name" not in call_data:
            return None

        name = call_data["name"]
        if not isinstance(name, str) or not name:
            return None

        # --- Unwrap double-nested tool calls ---
        # Models sometimes produce: {"name": "fn", "arguments": {"name": "fn", "arguments": {...}}}
        if "arguments" in call_data and isinstance(call_data["arguments"], dict):
            inner = call_data["arguments"]
            if "name" in inner and "arguments" in inner and isinstance(inner["name"], str):
                # The real tool call is the inner dict; unwrap one level
                call_data = inner
                name = call_data["name"]

        # --- Normalise arguments ---
        if "arguments" in call_data:
            args = call_data["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
        else:
            # Flat format: all keys except "name" become the arguments dict
            args = {k: v for k, v in call_data.items() if k != "name"}

        return {"name": name, "arguments": args}

    # ------------------------------------------------------------------
    # Extract tool calls from tagged blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_from_tags(
        text: str,
        begin_tag: str,
        end_tag: str,
    ) -> list[dict[str, Any]]:
        """Extract tool call dicts from *text* delimited by *begin_tag* / *end_tag*.

        Handles cases where the closing tag is missing by searching for the
        next JSON object boundary.  Also handles the case where two opening
        tags share a single closing tag (the content is split at the inner
        opening tag).
        """
        tool_calls: list[dict[str, Any]] = []
        search_start = 0

        while True:
            idx = text.find(begin_tag, search_start)
            if idx == -1:
                break

            content_start = idx + len(begin_tag)
            end_idx = text.find(end_tag, content_start)
            next_open = text.find(begin_tag, content_start)

            if end_idx != -1:
                # If another opening tag appears *before* the closing tag,
                # treat the first block as implicitly closed at that point.
                if next_open != -1 and next_open < end_idx:
                    json_content = text[content_start:next_open].strip()
                    search_start = next_open
                else:
                    json_content = text[content_start:end_idx].strip()
                    search_start = end_idx + len(end_tag)
            else:
                # Missing closing tag — grab up to next opening tag or end of text
                if next_open != -1:
                    json_content = text[content_start:next_open].strip()
                    search_start = next_open
                else:
                    json_content = text[content_start:].strip()
                    search_start = len(text)

            parsed = QwenToolParser._try_parse_json(json_content)
            if parsed is not None:
                tool_calls.append(parsed)
            else:
                # Last-resort: regex repair for heavily malformed JSON (e.g.
                # unescaped quotes inside string values from code edits).
                repaired = QwenToolParser._repair_malformed_tool_json(json_content)
                if repaired:
                    tool_calls.append(repaired)

        return tool_calls

    # ------------------------------------------------------------------
    # Bare-JSON fallback (no XML tags at all)
    # ------------------------------------------------------------------

    # Matches the start of a JSON object with a "name" key — we then use
    # _extract_first_json_object to grab the full balanced object.
    _BARE_JSON_START_RE = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"')

    @classmethod
    def _extract_bare_json(cls, text: str) -> list[dict[str, Any]]:
        """Fallback: look for bare ``{"name": ..., ...}`` objects that aren't
        wrapped in any XML tags.  Only used when tag-based extraction yields
        nothing.
        """
        tool_calls: list[dict[str, Any]] = []
        # Only look after the last </think> (if present) to avoid false positives
        # inside the model's chain-of-thought.
        think_end = text.rfind("</think>")
        search_text = text[think_end + len("</think>"):] if think_end != -1 else text

        for m in cls._BARE_JSON_START_RE.finditer(search_text):
            obj_str = cls._extract_first_json_object(search_text[m.start():])
            if not obj_str:
                continue
            parsed = cls._try_parse_json(obj_str)
            if parsed is not None:
                tool_calls.append(parsed)

        # Secondary fallback: bare result JSON without a "name" key
        # (model outputs {"result": [...]} or a list directly instead of a finish call)
        if not tool_calls:
            obj_str = cls._extract_first_json_object(search_text)
            if obj_str:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, (list, dict)) and (not isinstance(obj, dict) or "name" not in obj):
                        tool_calls.append({"name": "finish", "arguments": {"command": "submit", "result": json.dumps(obj, ensure_ascii=False)}})
                except (json.JSONDecodeError, ValueError):
                    pass

        return tool_calls

    # ------------------------------------------------------------------
    # \boxed{} fallback: detect direct answers and wrap as finish call
    # ------------------------------------------------------------------

    @staticmethod
    def _last_balanced_span(text: str, anchors: tuple[str, ...]) -> str | None:
        """Return the content of the last complete braced span after anchors."""
        matches: list[tuple[int, int]] = []
        for anchor in anchors:
            start = text.find(anchor)
            while start != -1:
                matches.append((start, len(anchor)))
                start = text.find(anchor, start + 1)

        for start, anchor_len in sorted(matches, reverse=True):
            i = start + anchor_len
            depth = 1
            j = i
            while depth and j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                return text[i : j - 1]
        return None

    @classmethod
    def _extract_answer_tag(cls, text: str) -> list[dict[str, Any]]:
        """Treat the last <answer>...</answer> after reasoning as a finish."""
        think_end = text.rfind("</think>")
        if think_end == -1:
            if len(text) > 2000:
                return []
            search_text = text
        else:
            search_text = text[think_end + len("</think>"):]

        matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", search_text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            answer = matches[-1].strip()
            return [{"name": "finish", "arguments": {"command": "submit", "result": answer}}]
        return []

    @classmethod
    def _extract_boxed_answer(cls, text: str) -> list[dict[str, Any]]:
        r"""Last-resort fallback for search tasks: if the model wrote
        ``\boxed{answer}`` (or ``$\boxed{answer}$``) without a ``finish``
        tool call, treat it as an implicit ``finish(result=answer)`` so the
        answer is not lost.

        Only searches text *after* the last ``</think>`` to avoid matching
        boxed expressions inside the chain-of-thought.
        """
        think_end = text.rfind("</think>")
        if think_end == -1:
            # No </think> tag — only use this fallback if text is short
            # (to avoid false positives in long code-heavy responses)
            if len(text) > 2000:
                return []
            search_text = text
        else:
            search_text = text[think_end + len("</think>"):]

        answer = cls._last_balanced_span(search_text, ("\\boxed{", "boxed{", "oxed{", "\x08oxed{"))
        if answer is not None:
            answer = answer.strip()
            return [{"name": "finish", "arguments": {"command": "submit", "result": answer}}]
        return []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from text, handling multiple format variations.

        Supported formats (in priority order):
        1. ``<tool_call>{"name": "fn", "arguments": {...}}</tool_call>``
        2. ``<function_call>{"name": "fn", "arguments": {...}}</function_call>``
        3. Flat JSON (``"name"`` present, args as sibling keys instead of nested)
        4. Incomplete JSON (missing closing braces, trailing commas)
        5. Missing closing tags
        6. Bare JSON after ``</think>`` (no XML tags)
        7. ``<answer>answer</answer>`` treated as implicit ``finish`` call
        8. ``\\boxed{answer}`` treated as implicit ``finish`` call
        """
        # 1. Primary: <tool_call> tags
        tool_calls = self._extract_from_tags(text, self.tool_call_begin, self.tool_call_end)

        # 2. Alternate tags (e.g. <function_call>)
        if not tool_calls:
            for begin, end in self._ALTERNATE_TAGS:
                if begin in text:
                    tool_calls = self._extract_from_tags(text, begin, end)
                    if tool_calls:
                        break

        # 3. Bare JSON fallback (no tags at all)
        if not tool_calls:
            tool_calls = self._extract_bare_json(text)

        # 4. <answer> fallback: task-local answer tags outrank generic boxed
        # output when both appear after the reasoning block.
        if not tool_calls:
            tool_calls = self._extract_answer_tag(text)

        # 5. \boxed{} fallback: treat as finish(result=...) for search tasks
        if not tool_calls:
            tool_calls = self._extract_boxed_answer(text)

        return tool_calls

    @staticmethod
    def _repair_malformed_tool_json(s: str) -> dict[str, Any] | None:
        """Best-effort recovery of a tool call dict from malformed JSON.

        Handles the dominant failure mode: unescaped double-quotes inside
        string values (e.g. old_str / new_str containing Python code).
        Returns {"name": ..., "arguments": {...}} or None.
        """
        name_match = re.search(r'"name"\s*:\s*"(\w+)"', s)
        if not name_match:
            return None
        name = name_match.group(1)

        args_start = s.find('"arguments"')
        if args_start < 0:
            return None
        args_brace = s.find('{', args_start)
        if args_brace < 0:
            return None

        args_content = s[args_brace:]
        known_keys = ['command', 'path', 'old_str', 'new_str', 'insert_line', 'cmd', 'view_range',
                      'query', 'top_k', 'result', 'search_term']

        args: dict[str, Any] = {}
        for key in known_keys:
            # Try string value: "key": "..."
            key_pat = f'"{key}"\\s*:\\s*"'
            key_match = re.search(key_pat, args_content)
            if not key_match:
                # Try integer value: "key": 123
                int_pat = f'"{key}"\\s*:\\s*(\\d+)'
                int_match = re.search(int_pat, args_content)
                if int_match:
                    args[key] = int_match.group(1)
                continue

            val_start = key_match.end()
            remaining = args_content[val_start:]

            # Find the end of this value: the earliest next-key boundary or closing braces
            best_end = len(remaining)
            for next_key in known_keys:
                if next_key == key:
                    continue
                nk_match = re.search(f'",\\s*"{next_key}"', remaining)
                if nk_match and nk_match.start() < best_end:
                    best_end = nk_match.start()
            close_match = re.search(r'"\s*\}\s*\}', remaining)
            if close_match and close_match.start() < best_end:
                best_end = close_match.start()

            args[key] = remaining[:best_end]

        if not args:
            return None
        return {"name": name, "arguments": args}

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a valid json object with function name and arguments within a pairwise <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Make sure all curly braces and XML tags are correctly balanced and closed strictly.
""".rstrip()


class Qwen3CoderToolParser(QwenToolParser):
    """Parser for Qwen3-Coder/Qwen3.5 XML-like tool calls.

    Qwen3 uses JSON inside ``<tool_call>`` tags. Qwen3.5/Qwen3-Coder emits
    function and parameter blocks instead:

    ``<tool_call><function=name><parameter=arg>value</parameter></function></tool_call>``
    """

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
    _FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
    _PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)

    def __init__(self, valid_tools: set[str] | None = None):
        super().__init__(valid_tools=valid_tools)
        self.tool_call_prefix = "<function="
        self._tool_parameter_config: dict[str, dict[str, dict[str, Any]]] = {}

    def parse(self, model_response: str) -> list[ToolCall]:
        tool_calls_dicts = self.parse_qwen3_coder_tool_calls(model_response)
        if self.valid_tools:
            for tc in tool_calls_dicts:
                tc["name"] = self._normalize_tool_name(tc["name"], self.valid_tools)
        return [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]

    @staticmethod
    def _strip_outer_newline(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    @staticmethod
    def _iter_tool_schemas(tools_schema: str) -> list[dict[str, Any]]:
        tools_schema = tools_schema.strip()
        if not tools_schema:
            return []

        try:
            loaded = json.loads(tools_schema)
            if isinstance(loaded, list):
                return [item for item in loaded if isinstance(item, dict)]
            if isinstance(loaded, dict):
                return [loaded]
        except (json.JSONDecodeError, ValueError):
            pass

        decoder = json.JSONDecoder()
        schemas: list[dict[str, Any]] = []
        idx = 0
        while idx < len(tools_schema):
            while idx < len(tools_schema) and tools_schema[idx].isspace():
                idx += 1
            if idx >= len(tools_schema):
                break
            try:
                obj, next_idx = decoder.raw_decode(tools_schema, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, dict):
                schemas.append(obj)
            elif isinstance(obj, list):
                schemas.extend(item for item in obj if isinstance(item, dict))
            idx = next_idx
        return schemas

    @classmethod
    def _extract_parameter_config(cls, tools_schema: str) -> dict[str, dict[str, dict[str, Any]]]:
        config: dict[str, dict[str, dict[str, Any]]] = {}
        for schema in cls._iter_tool_schemas(tools_schema):
            function_schema = schema.get("function", schema)
            if not isinstance(function_schema, dict):
                continue
            name = function_schema.get("name")
            parameters = function_schema.get("parameters", {})
            if not isinstance(name, str) or not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties", {})
            if isinstance(properties, dict):
                config[name] = {str(k): v for k, v in properties.items() if isinstance(v, dict)}
        return config

    @staticmethod
    def _convert_param_value(param_value: str, param_name: str, param_config: dict[str, dict[str, Any]], func_name: str) -> Any:
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config:
                logger.warning(
                    "Parsed parameter '%s' is not defined in the tool parameters for tool '%s'; returning string value.",
                    param_name,
                    func_name,
                )
            return param_value

        param_schema = param_config[param_name]
        param_type = str(param_schema.get("type", "string")).strip().lower()

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not an integer in tool '%s'; returning string value.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_value = float(param_value)
                return float_value if float_value - int(float_value) != 0 else int(float_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a float in tool '%s'; returning string value.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        if param_type in ["boolean", "bool", "binary"]:
            bool_value = param_value.lower()
            if bool_value not in ["true", "false"]:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a boolean in tool '%s'; degenerating to false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return bool_value == "true"

        if param_type in ["object", "array"] or param_type.startswith("dict") or param_type.startswith("list"):
            try:
                return json.loads(param_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not valid JSON in tool '%s'; trying ast.literal_eval().",
                    param_value,
                    param_name,
                    func_name,
                )

        try:
            return ast.literal_eval(param_value)
        except Exception:
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be converted via ast.literal_eval() in tool '%s'; returning string value.",
                param_value,
                param_name,
                func_name,
            )
            return param_value

    def _parse_xml_function_call(self, function_name: str, parameters: str) -> dict[str, Any]:
        param_config = self._tool_parameter_config.get(function_name, {})
        arguments: dict[str, Any] = {}

        for param_name, param_value in self._PARAMETER_RE.findall(parameters):
            param_name = param_name.strip()
            param_value = self._strip_outer_newline(str(param_value))
            arguments[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)

        return {"name": function_name.strip(), "arguments": arguments}

    def parse_qwen3_coder_tool_calls(self, text: str) -> list[dict[str, Any]]:
        raw_tool_calls = [match[0] if match[0] else match[1] for match in self._TOOL_CALL_RE.findall(text)]
        if not raw_tool_calls and self.tool_call_prefix in text:
            raw_tool_calls = [text]

        tool_calls: list[dict[str, Any]] = []
        for raw_tool_call in raw_tool_calls:
            for function_name, parameters in self._FUNCTION_RE.findall(raw_tool_call):
                function_name = function_name.strip()
                if function_name:
                    tool_calls.append(self._parse_xml_function_call(function_name, parameters))
        return tool_calls

    def format_tool_call(self, tool_call_dict: dict[str, Any]) -> str:
        name = str(tool_call_dict.get("name", ""))
        arguments_obj = tool_call_dict.get("arguments") or {}
        if isinstance(arguments_obj, str):
            try:
                arguments_obj = json.loads(arguments_obj)
            except json.JSONDecodeError:
                arguments_obj = {}
        if not isinstance(arguments_obj, dict):
            arguments_obj = {}

        lines = [self.tool_call_begin, f"<function={name}>"]
        for param_name, param_value in arguments_obj.items():
            if isinstance(param_value, str):
                param_text = param_value
            else:
                param_text = json.dumps(param_value, ensure_ascii=False)
            lines.extend([f"<parameter={param_name}>", param_text, "</parameter>"])
        lines.extend(["</function>", self.tool_call_end])
        return "\n".join(lines)

    def get_tool_prompt(self, tools_schema: str) -> str:
        self._tool_parameter_config = self._extract_parameter_config(tools_schema)
        return f"""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return the function name and parameters within a pairwise <tool_call></tool_call> XML block:
<tool_call>
<function=FUNCTION_NAME>
<parameter=PARAMETER_NAME>
PARAMETER_VALUE
</parameter>
</function>
</tool_call>

For multiple parameters, emit one <parameter=...></parameter> block per parameter inside the same function block.
For multiple tool calls, emit multiple <tool_call></tool_call> blocks.
Do not put JSON tool-call objects inside <tool_call> for this model family.
""".rstrip()
