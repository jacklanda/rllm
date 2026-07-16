"""Tokenizer compatibility helpers for local checkpoints."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

_QWEN_3_RE = re.compile(r"qwen[-_/]?3(?=[^0-9]|$)", re.IGNORECASE)
QWEN_THINK_START = "<think>"
QWEN_THINK_END = "</think>"


def _model_names(model_or_tokenizer: Any) -> list[str]:
    names: list[str] = []
    if isinstance(model_or_tokenizer, str):
        names.append(model_or_tokenizer)
    else:
        for attr in ("name_or_path", "model_name", "model_path"):
            value = getattr(model_or_tokenizer, attr, None)
            if isinstance(value, str):
                names.append(value)
    return names


def is_qwen3_family(model_or_tokenizer: Any) -> bool:
    return any(_QWEN_3_RE.search(name) for name in _model_names(model_or_tokenizer))


def qwen_thinking_chat_template_kwargs(
    model_or_tokenizer: Any, *, disable_thinking: bool
) -> dict[str, bool]:
    if not is_qwen3_family(model_or_tokenizer):
        return {}
    return {"enable_thinking": not disable_thinking}


def apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict],
    *,
    model: str | None = None,
    disable_thinking: bool = False,
    **kwargs: Any,
) -> str:
    template_kwargs = qwen_thinking_chat_template_kwargs(
        model or tokenizer, disable_thinking=disable_thinking
    )
    return tokenizer.apply_chat_template(messages, **kwargs, **template_kwargs)


def configure_parser_for_qwen_thinking(
    parser: Any,
    model_or_tokenizer: Any,
    *,
    disable_thinking: bool,
    no_system_prompt: bool = False,
) -> bool:
    """Use the tokenizer chat template for Qwen3 thinking-mode prompts."""

    if not is_qwen3_family(model_or_tokenizer):
        return False

    tokenizer = getattr(parser, "tokenizer", None)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        return False

    def _parse(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        tools: list | None = None,
        accumulate_reasoning: bool = False,
        **kwargs: Any,
    ) -> str:
        del is_first_msg, accumulate_reasoning
        prompt_kwargs = dict(kwargs)
        prompt_kwargs.pop("tools", None)
        prompt_kwargs["tokenize"] = False
        prompt_kwargs["add_generation_prompt"] = add_generation_prompt
        if tools is not None and not no_system_prompt:
            prompt_kwargs["tools"] = tools
        return apply_chat_template_compat(
            self.tokenizer,
            messages,
            model=model_or_tokenizer,
            disable_thinking=disable_thinking,
            **prompt_kwargs,
        )

    parser.parse = MethodType(_parse, parser)

    stub_messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    with_prompt = apply_chat_template_compat(
        tokenizer,
        stub_messages,
        model=model_or_tokenizer,
        disable_thinking=disable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    without_prompt = apply_chat_template_compat(
        tokenizer,
        stub_messages,
        model=model_or_tokenizer,
        disable_thinking=disable_thinking,
        tokenize=False,
        add_generation_prompt=False,
    )
    parser.generation_prompt = with_prompt[len(without_prompt) :]
    return True


def configure_parser_for_no_system_prompt(parser: Any) -> None:
    """Prevent local parser fallbacks from injecting a default system prompt.

    Some rLLM chat parsers, notably QwenChatTemplateParser, synthesize a
    default system message when the first chat message is not a system role.
    ``--no-system-prompt`` should mean no system text is added by evals, so
    wrap those parsers and skip that synthetic branch.
    """

    if parser.__class__.__name__ != "QwenChatTemplateParser":
        return

    if not all(
        hasattr(parser, attr)
        for attr in (
            "parse_system",
            "parse_user",
            "parse_assistant",
            "parse_tool",
            "generation_prompt",
        )
    ):
        return

    def _parse(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        tools: list | None = None,
        accumulate_reasoning: bool = False,
        **_: Any,
    ) -> str:
        del is_first_msg, tools

        def _parse_system(message: dict) -> str:
            try:
                return self.parse_system(message, "")
            except TypeError:
                return self.parse_system(message)

        result = ""
        for message in messages:
            role = message["role"]
            if role == "system":
                result += _parse_system(message)
            elif role == "user":
                result += self.parse_user(message)
            elif role == "assistant":
                result += self.parse_assistant(
                    message, accumulate_reasoning=accumulate_reasoning
                )
            elif role == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {role}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    parser.parse = MethodType(_parse, parser)


def qwen_completion_text_with_thinking_tags(
    raw_completion_text: str,
    *,
    reasoning: str | None = None,
    strip_special_tokens: Any | None = None,
) -> str:
    text = raw_completion_text or ""
    if callable(strip_special_tokens):
        text = strip_special_tokens(text)
    has_thinking = bool(reasoning) or QWEN_THINK_END in text
    if has_thinking and text and not text.lstrip().startswith(QWEN_THINK_START):
        return f"{QWEN_THINK_START}\n{text}"
    return text


def get_tool_parser_compat(parser_name: str, model: str | None = None) -> Any:
    del model
    from rllm.parser import get_tool_parser

    return get_tool_parser(parser_name)()


def _read_tokenizer_config(model: str) -> dict[str, Any] | None:
    path = Path(model)
    if not path.is_dir():
        return None

    config_path = path / "tokenizer_config.json"
    if not config_path.is_file():
        return None

    try:
        with config_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _legacy_extra_special_tokens(model: str) -> list[str] | None:
    config = _read_tokenizer_config(model)
    if not config:
        return None

    tokens = config.get("extra_special_tokens")
    if not isinstance(tokens, list):
        return None

    return [token for token in tokens if isinstance(token, str)]


def load_tokenizer_compat(model: str, *args: Any, **kwargs: Any):
    """Load tokenizers with legacy list-valued ``extra_special_tokens``.

    Transformers 4.57 expects ``extra_special_tokens`` to be a mapping, while
    some Qwen-style local checkpoints store a legacy list there. Keep those
    tokens registered as special tokens via ``additional_special_tokens``.
    """

    tokens = _legacy_extra_special_tokens(model)
    if tokens:
        kwargs.setdefault("extra_special_tokens", {})
        kwargs.setdefault("additional_special_tokens", tokens)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, *args, **kwargs)


def prepare_vllm_model_path(
    model: str,
) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """Return a model path whose tokenizer config is accepted by vLLM/HF.

    vLLM loads its own tokenizer from the model directory and does not expose a
    tokenizer-kwargs hook for this field. For affected local checkpoints, build
    a temporary directory of symlinks with only ``tokenizer_config.json``
    rewritten. The caller must keep the returned TemporaryDirectory alive.
    """

    tokens = _legacy_extra_special_tokens(model)
    if not tokens:
        return model, None

    src = Path(model)
    config = _read_tokenizer_config(model)
    if config is None:
        return model, None

    tmpdir = tempfile.TemporaryDirectory(prefix="evals-vllm-model-")
    dst = Path(tmpdir.name)

    for child in src.iterdir():
        target = dst / child.name
        if child.name == "tokenizer_config.json":
            continue
        os.symlink(child, target, target_is_directory=child.is_dir())

    patched_config = dict(config)
    patched_config["extra_special_tokens"] = {}
    patched_config.setdefault("additional_special_tokens", tokens)
    with (dst / "tokenizer_config.json").open("w") as f:
        json.dump(patched_config, f, indent=2)
        f.write("\n")

    return str(dst), tmpdir
