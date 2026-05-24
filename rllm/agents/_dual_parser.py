"""Dual parser for SWE-XML and Qwen-style ``<tool_call>`` responses.

Some base models (e.g. Qwen3-Thinking) emit tool calls in their native
``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` JSON form even
when the system prompt asks for SWE's
``<function=NAME><parameter=KEY>VAL</parameter></function>`` XML. This
module accepts either format and normalises to a single ``SWEAction``.

The XML path is byte-for-byte identical to ``swe_agent.parse_xml_response``
so existing SWE behaviour is preserved when the model already speaks XML.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None  # type: ignore[assignment]


_XML_FN_PATTERN = re.compile(r"(?s)(<function=.*?</function>)")
_XML_FN_TRUNC_PATTERN = re.compile(r"(?s)(<function=[^>]+>.*)")
_TOOL_CALL_PATTERN = re.compile(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>")
_TOOL_CALL_TRUNC_PATTERN = re.compile(r"(?s)<tool_call>\s*(\{.*)$")


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Normalise ``arguments`` from ``<tool_call>`` JSON into a flat dict.

    Qwen variants seen in the wild:
      - dict: ``{"command": "view", "path": "/x"}``
      - JSON string: ``"{\"command\": \"view\"}"``
      - double-encoded: dict whose values are still JSON strings.
    Non-string scalar values are stringified so SWEAction stores str-only,
    matching what ``Action.from_string`` would have produced from XML.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return {}
    if not isinstance(arguments, dict):
        return {}
    return {str(k): v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if not isinstance(v, (int, float, bool)) else str(v) for k, v in arguments.items()}


def _parse_tool_call_json(blob: str) -> "SWEAction | None":
    """Parse a JSON ``{name, arguments}`` blob into a SWEAction."""
    if SWEAction is None:
        return None
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        # Truncated / partial JSON: attempt to close common open structures.
        repaired = blob.rstrip().rstrip(",")
        obj = None
        for closing in ("}", '"}', "}}", '"}}', '"}}}'):
            try:
                obj = json.loads(repaired + closing)
                break
            except (ValueError, TypeError):
                continue
        if obj is None:
            # Last resort: pull the function name out via regex so the env
            # at least sees a structured (name, {}) instead of an empty
            # action. Better an "unknown args" error than silent dropping.
            name_match = re.search(r'"(?:name|function)"\s*:\s*"([^"\\]+)"', blob)
            if name_match:
                return SWEAction(function_name=name_match.group(1).strip(), parameters={})
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or ""
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    return SWEAction(function_name=str(name).strip(), parameters=_coerce_arguments(args))


def parse_dual_response(response_text: str) -> tuple[str, "SWEAction"]:
    """Parse SWE-XML first, then Qwen ``<tool_call>`` JSON.

    Returns ``(thought, action)`` where ``action`` is a ``SWEAction``. The
    thought is everything preceding the first matched tool block. If no
    tool call is present the response is returned as-is and the action has
    empty ``function_name``/``parameters`` (env will surface the standard
    "you forgot to use a function call" hint).
    """
    if SWEAction is None:
        raise RuntimeError("r2egym is required for dual parsing.")

    match = _XML_FN_PATTERN.search(response_text)
    if match:
        action_str = match.group(1).strip()
        thought = response_text[: match.start()].strip()
        return thought, SWEAction.from_string(action_str)

    tc_match = _TOOL_CALL_PATTERN.search(response_text)
    if tc_match:
        blob = tc_match.group(1)
        action = _parse_tool_call_json(blob)
        thought = response_text[: tc_match.start()].strip()
        if action is not None:
            return thought, action

    # Truncated XML fallback (max_tokens cut mid-call).
    trunc = _XML_FN_TRUNC_PATTERN.search(response_text)
    if trunc:
        action_str = trunc.group(1).strip()
        if "</function>" not in action_str:
            action_str += "\n</function>"
        thought = response_text[: trunc.start()].strip()
        return thought, SWEAction.from_string(action_str)

    # Truncated <tool_call> fallback (no closing tag).
    tc_trunc = _TOOL_CALL_TRUNC_PATTERN.search(response_text)
    if tc_trunc:
        blob = tc_trunc.group(1)
        action = _parse_tool_call_json(blob)
        thought = response_text[: tc_trunc.start()].strip()
        if action is not None:
            return thought, action

    return response_text.strip(), SWEAction(function_name="", parameters={})
