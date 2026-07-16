"""In-process SGLang rollout engine for agentic eval."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from evals.tokenizer_compat import (
    configure_parser_for_no_system_prompt,
    configure_parser_for_qwen_thinking,
    load_tokenizer_compat,
    prepare_vllm_model_path,
    qwen_completion_text_with_thinking_tags,
)
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser


class SGLangEngine(RolloutEngine):
    """Drive SGLang's in-process Engine through the rLLM RolloutEngine API."""

    def __init__(
        self,
        model: str,
        max_prompt_length: int = 16384,
        max_response_length: int = 8192,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        tools: list[Any] | None = None,
        accumulate_reasoning: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        disable_thinking: bool = False,
        no_system_prompt: bool = False,
        engine_kwargs: dict | None = None,
        **_: Any,
    ) -> None:
        os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

        print("[evals-sglang] import sglang.Engine", flush=True)
        from sglang import Engine

        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = (
            max_model_length
            if max_model_length is not None
            else max_prompt_length + max_response_length
        )
        self.sampling_params = dict(sampling_params or {})
        self.tools = list(tools or [])
        self.accumulate_reasoning = accumulate_reasoning

        print("[evals-sglang] preparing model path", flush=True)
        model_path, self._model_tmpdir = prepare_vllm_model_path(model)
        print("[evals-sglang] loading tokenizer", flush=True)
        self.tokenizer = load_tokenizer_compat(
            model, trust_remote_code=trust_remote_code
        )
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None
        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer, disable_thinking=disable_thinking
        )
        qwen_thinking_configured = configure_parser_for_qwen_thinking(
            self.chat_parser,
            model,
            disable_thinking=disable_thinking,
            no_system_prompt=no_system_prompt,
        )
        if no_system_prompt and not qwen_thinking_configured:
            configure_parser_for_no_system_prompt(self.chat_parser)

        engine_init_kwargs = {
            "model_path": model_path,
            "tp_size": tensor_parallel_size,
            "mem_fraction_static": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "context_length": self.max_model_length,
            "max_running_requests": 2,
            "max_total_tokens": self.max_model_length,
            "disable_cuda_graph": True,
            "disable_piecewise_cuda_graph": True,
            "watchdog_timeout": 3600,
        }
        engine_init_kwargs.update(engine_kwargs or {})
        print("[evals-sglang] constructing sglang.Engine", flush=True)
        self.engine = Engine(**engine_init_kwargs)
        print("[evals-sglang] sglang.Engine constructed", flush=True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="EvalsSGLang"
        )
        self.validate = False

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        request_id = f"{application_id}-{uuid.uuid4().hex[:8]}"
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop(
            "accumulate_reasoning", self.accumulate_reasoning
        )

        merged = self.sampling_params.copy()
        merged.update(kwargs)
        for ignored in (
            "validate",
            "model",
            "enforce_max_prompt_length",
            "precomputed_prompt_ids",
            "accumulate_reasoning",
            "reasoning_effort",
            "tools",
        ):
            merged.pop(ignored, None)

        max_tokens = int(
            merged.pop(
                "max_tokens", merged.pop("max_new_tokens", self.max_response_length)
            )
        )
        sampling_params: dict[str, Any] = {
            "temperature": merged.pop("temperature", 0.6),
            "top_p": merged.pop("top_p", 0.95),
            "max_new_tokens": max_tokens,
        }
        for key in (
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "skip_special_tokens",
            "ignore_eos",
        ):
            if key in merged:
                sampling_params[key] = merged[key]

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.max_prompt_length or len(prompt_ids) > self.max_model_length:
            from rllm.workflows import TerminationEvent, TerminationReason

            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        remaining_tokens = self.max_model_length - len(prompt_ids) - 1
        if remaining_tokens <= 0:
            from rllm.workflows import TerminationEvent, TerminationReason

            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)
        max_tokens = min(max_tokens, remaining_tokens)
        sampling_params["max_new_tokens"] = max_tokens

        result = await self._generate(prompt, sampling_params, request_id)
        completion_ids = _extract_output_ids(result)
        if completion_ids is None:
            completion_ids = self.tokenizer.encode(
                _extract_text(result), add_special_tokens=False
            )
        completion_ids = completion_ids[:max_tokens]

        finish_reason = _extract_finish_reason(result)
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"

        parsed = self.chat_parser.parse_completion(completion_ids)
        valid_tool_calls = []
        for tc in parsed.get("tool_calls") or []:
            if not getattr(tc, "name", None) or not getattr(tc, "arguments", None):
                continue
            valid_tool_calls.append(tc)

        raw_completion_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )
        completion_text = qwen_completion_text_with_thinking_tags(
            raw_completion_text,
            reasoning=parsed.get("reasoning", ""),
            strip_special_tokens=getattr(
                self.chat_parser, "_strip_special_tokens", None
            ),
        )

        return ModelOutput(
            text=completion_text,
            content=parsed.get("content", ""),
            reasoning=parsed.get("reasoning", ""),
            tool_calls=valid_tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=[],
            prompt_logprobs=[],
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self) -> None:
        return None

    async def sleep(self) -> None:
        return None

    async def _generate(
        self, prompt: str, sampling_params: dict[str, Any], request_id: str
    ) -> Any:
        import asyncio

        loop = asyncio.get_running_loop()
        generate = partial(
            self.engine.generate,
            prompt=prompt,
            sampling_params=sampling_params,
            rid=request_id,
        )
        return await loop.run_in_executor(self._executor, generate)

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        self._executor.shutdown(wait=True, cancel_futures=True)


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("text", "output", "content"):
            if key in result and result[key] is not None:
                return str(result[key])
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                return str(
                    choice.get("text")
                    or choice.get("message", {}).get("content")
                    or ""
                )
    return str(result or "")


def _extract_output_ids(result: Any) -> list[int] | None:
    if isinstance(result, dict):
        output_ids = result.get("output_ids")
        if isinstance(output_ids, list):
            return [int(token_id) for token_id in output_ids]
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                choice_ids = choice.get("output_ids") or choice.get("token_ids")
                if isinstance(choice_ids, list):
                    return [int(token_id) for token_id in choice_ids]
    return None


def _extract_finish_reason(result: Any) -> str:
    if isinstance(result, dict):
        meta = result.get("meta_info") or result.get("meta") or {}
        reason = meta.get("finish_reason") or result.get("finish_reason")
        if reason:
            return str(reason)
    return "stop"


__all__ = ["SGLangEngine"]
