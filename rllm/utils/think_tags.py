"""Helpers for preserving thinking delimiters in dumped trajectories."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START


def ensure_think_start(text: str) -> str:
    """Prefix ``<think>`` when text contains an orphan ``</think>``."""
    if THOUGHT_DELIMITER_END not in text:
        return text
    first_end = text.find(THOUGHT_DELIMITER_END)
    first_start = text.find(THOUGHT_DELIMITER_START)
    if first_start == -1 or first_start > first_end:
        return THOUGHT_DELIMITER_START + text
    return text


def format_think_block(thought: Any) -> str:
    """Return a complete ``<think>...</think>`` block for dumps."""
    thought_text = "" if thought is None else str(thought)
    stripped = thought_text.strip()
    if stripped.startswith(THOUGHT_DELIMITER_START) and stripped.endswith(THOUGHT_DELIMITER_END):
        return stripped
    if stripped.startswith(THOUGHT_DELIMITER_START):
        stripped = stripped[len(THOUGHT_DELIMITER_START) :].lstrip()
    if stripped.endswith(THOUGHT_DELIMITER_END):
        stripped = stripped[: -len(THOUGHT_DELIMITER_END)].rstrip()
    return f"{THOUGHT_DELIMITER_START}{stripped}{THOUGHT_DELIMITER_END}"


def format_assistant_content_for_dump(content: Any, reasoning: Any = None) -> Any:
    """Normalize assistant content for trajectory dumps without mutating training data."""
    if content is None:
        content_text = ""
    elif not isinstance(content, str):
        return content
    else:
        content_text = content

    if THOUGHT_DELIMITER_START in content_text:
        return content_text
    if THOUGHT_DELIMITER_END in content_text:
        return ensure_think_start(content_text)
    if reasoning:
        if content_text:
            return f"{format_think_block(reasoning)}\n\n{content_text}"
        return format_think_block(reasoning)
    return content


def sanitize_messages_for_dump(messages: Any) -> Any:
    """Return a deep-copied message list with assistant think tags completed."""
    if not isinstance(messages, list):
        return messages
    sanitized = deepcopy(messages)
    for msg in sanitized:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        msg["content"] = format_assistant_content_for_dump(
            msg.get("content"),
            msg.get("reasoning") or msg.get("reasoning_content"),
        )
    return sanitized


def sanitize_trajectory_dump_for_think_tags(value: Any) -> Any:
    """Recursively sanitize dumped trajectory payloads."""
    if isinstance(value, list):
        if all(isinstance(item, dict) and "role" in item for item in value):
            return sanitize_messages_for_dump(value)
        return [sanitize_trajectory_dump_for_think_tags(item) for item in value]
    if isinstance(value, dict):
        if value.get("role") == "assistant":
            sanitized_msg = deepcopy(value)
            sanitized_msg["content"] = format_assistant_content_for_dump(
                sanitized_msg.get("content"),
                sanitized_msg.get("reasoning") or sanitized_msg.get("reasoning_content"),
            )
            return sanitized_msg
        sanitized = {}
        for key, item in value.items():
            if key == "trajectory" and isinstance(item, list):
                sanitized[key] = sanitize_messages_for_dump(item)
            elif key == "chat_completions" and isinstance(item, list):
                sanitized[key] = sanitize_messages_for_dump(item)
            elif key in {"model_response", "response", "assistant"}:
                sanitized[key] = format_assistant_content_for_dump(item)
            else:
                sanitized[key] = sanitize_trajectory_dump_for_think_tags(item)
        return sanitized
    return value
