import asyncio
import contextlib
import logging
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


logger = logging.getLogger(__name__)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _shutdown_rollout_server_actor(server, timeout: float | None = 30.0) -> dict:
    """Run inside a VERL rollout HTTP server Ray actor."""
    shutdown_results = {}

    engine = getattr(server, "engine", None)
    if engine is not None and hasattr(engine, "shutdown"):
        engine.shutdown(timeout=timeout)
        shutdown_results["engine"] = True

    engine_manager = getattr(server, "engine_manager", None)
    if engine_manager is not None and hasattr(engine_manager, "shutdown"):
        engine_manager.shutdown(timeout=timeout)
        shutdown_results["engine_manager"] = True

    server_task = getattr(server, "_server_task", None)
    if server_task is not None and hasattr(server_task, "cancel"):
        server_task.cancel()
        shutdown_results["server_task"] = True

    return shutdown_results


def _make_shutdown_rollout_server_actor(timeout: float | None):
    def shutdown_rollout_server_actor(server):
        return _shutdown_rollout_server_actor(server, timeout)

    return shutdown_rollout_server_actor


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        # Per-step generation soft cap (workflow path). train_sampling_params
        # below defaults max_tokens to the full max_response_length, so a
        # single runaway "thinking" turn can decode the entire 64k budget in
        # one shot — hogging a vLLM decode slot and stalling the batch. This
        # caps any single turn; the trajectory-wide budget is still enforced
        # downstream (responses are padded/truncated to max_response_length in
        # AgentWorkflowEngine). null/<=0 disables the cap.
        _per_step = config.get("rllm", {}).get("agent", {}).get("per_step_max_tokens", None)
        self.per_step_max_tokens = int(_per_step) if _per_step and int(_per_step) > 0 else None
        self.apply_per_step_max_tokens_to_eval = _as_bool(
            config.get("rllm", {}).get("agent", {}).get("apply_per_step_max_tokens_to_eval", False),
            default=False,
        )
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)
        self.return_rollout_logprobs = _as_bool(config.get("rllm", {}).get("rollout_logprobs", False))
        enable_sleep_mode = config.get("rllm", {}).get("rollout_enable_sleep_mode", True)
        self.enable_sleep_mode = _as_bool(enable_sleep_mode, default=True)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
            max_tokens=self.max_response_length,
        )
        if self.return_rollout_logprobs:
            self.train_sampling_params["logprobs"] = 1

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
            max_tokens=self.max_response_length,
        )
        if self.return_rollout_logprobs:
            self.val_sampling_params["logprobs"] = 1

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl
        self._shutdown = False

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        precomputed_prompt_ids = kwargs.pop("precomputed_prompt_ids", None)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = int(sampling_params.pop("max_new_tokens", sampling_params.get("max_tokens", self.max_response_length)))
        # Apply the per-step soft cap to training rollouts only (leave
        # validation generation at its full budget so eval behavior is
        # unchanged). Caller-supplied smaller max_tokens still wins.
        #
        # IMPORTANT: hitting this per-step cap must NOT be reported as
        # finish_reason="length", or the workflow would terminate the whole
        # trajectory (MAX_RESPONSE_LENGTH_EXCEEDED) on a merely-long-but-normal
        # thinking turn. We remember the pre-cap budget so a completion that
        # only hit the *soft* cap (and still has trajectory budget left) is
        # reported as finish_reason="stop" and the agent simply takes another
        # turn. Only hitting the true (uncapped) budget reports "length".
        uncapped_max_tokens = max_tokens
        is_validation_request = self.validate or validate
        if self.per_step_max_tokens is not None and (not is_validation_request or self.apply_per_step_max_tokens_to_eval):
            max_tokens = min(max_tokens, self.per_step_max_tokens)
        sampling_params["max_tokens"] = max_tokens

        if precomputed_prompt_ids is not None:
            # Use precomputed prompt IDs to avoid BPE retokenization mismatch
            request_prompt_ids = precomputed_prompt_ids
            prompt_ids = precomputed_prompt_ids
            image_data = None
            multi_modal_inputs = None
        else:
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
            request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

            if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
                image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
                model_inputs = self.processor(text=[prompt], images=image_data)
                prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
                model_inputs.pop("attention_mask")
                multi_modal_inputs = dict(model_inputs)
            else:
                image_data = None
                multi_modal_inputs = None
                prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        token_output: TokenOutput = await self.server_manager.generate(request_id=application_id, prompt_ids=request_prompt_ids, image_data=image_data, sampling_params=sampling_params)  # type: ignore
        completion_ids: list[int] = token_output.token_ids
        logprobs: list[float] | None = list(token_output.log_probs) if token_output.log_probs is not None else None

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            # Truncate to whatever cap actually applied this turn.
            completion_ids = completion_ids[:max_tokens]
            if logprobs is not None:
                logprobs = logprobs[:max_tokens]
            # Only call it "length" (which ends the trajectory) when we hit the
            # real per-trajectory budget. Hitting only the per-step soft cap
            # leaves trajectory budget unspent, so report "stop" and let the
            # agent continue on the next turn.
            if max_tokens >= uncapped_max_tokens:
                finish_reason = "length"

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)
        tool_calls = parsed_output.get("tool_calls", [])

        # Fields validation for "tool_calls"
        valid_tool_calls = []
        for tool_call in tool_calls:
            if not tool_call.name or not tool_call.arguments:
                continue
            valid_tool_calls.append(tool_call)

        parsed_output["tool_calls"] = valid_tool_calls

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        if not self.enable_sleep_mode:
            return
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        if not self.enable_sleep_mode:
            return
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])

    async def shutdown(self, timeout: float | None = 30.0):
        """Drain requests and explicitly stop vLLM engine-core processes."""
        if self._shutdown:
            return
        self._shutdown = True

        with contextlib.suppress(Exception):
            await self.sleep()

        shutdown_refs = []
        if self.config.actor_rollout_ref.rollout.name == "vllm":
            shutdown_actor = _make_shutdown_rollout_server_actor(timeout)
            for replica in getattr(self.rollout_manager, "rollout_replicas", []):
                for server in getattr(replica, "servers", []):
                    shutdown_refs.append(server.__ray_call__.remote(shutdown_actor))

        if not shutdown_refs:
            return

        results = await asyncio.gather(*shutdown_refs, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Failed to shutdown a rollout server actor cleanly: %s", result)
