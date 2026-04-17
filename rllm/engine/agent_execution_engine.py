import asyncio
import logging
import threading
import time
import traceback
import uuid
import json
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.utils import colorful_print

logger = logging.getLogger(__name__)


class InvalidReactStructureError(Exception):
    pass


class DockerConnectionError(Exception):
    """Raised when Docker daemon is unreachable, to fast-fail all trajectories."""
    pass


def _is_docker_connection_error(exc: Exception) -> bool:
    """Check if an exception indicates Docker daemon connectivity failure."""
    msg = str(exc).lower()
    return (
        "error while fetching server api version" in msg
        or ("connection refused" in msg and ("docker" in msg or "/version" in msg or "/containers" in msg))
        or ("connection aborted" in msg and "permission denied" in msg)
        or ("max retries exceeded" in msg and ("/version" in msg or "/containers" in msg))
    )


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1024,  # The number of active agents
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=2,
        max_steps=32,
        max_response_length=36000,
        max_prompt_length=2048,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=1024,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        # Trajectory filtering toggles (read from config, default to True for backward compat)
        _tf = self.config.get("rllm", {}).get("trajectory_filtering", {}) if self.config is not None else {}
        self.validate_boxed_per_step = _tf.get("validate_boxed_per_step", False)       # Per-step \boxed{} / tool_call validation + retry
        self.enforce_react_structure = _tf.get("enforce_react_structure", False)         # Min 5 steps + final \boxed{} check
        self.max_tool_calls_per_turn = _tf.get("max_tool_calls_per_turn", 10)           # Max tool calls per single model turn

        self.incremental_tokenization = self.config.get("rllm", {}).get("incremental_tokenization", False) if self.config is not None else False

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]
        self._docker_healthy = None  # Set per generation round in trajectory_generator
        self._trajectory_logs = []  # Collect all trajectory data for dumping to trajs.json

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e6)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl", "tinker"], "Currently only openai, verl and tinker are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "tinker":
            from rllm.engine.rollout.tinker_engine import TinkerEngine

            self.rollout_engine = TinkerEngine(
                **rollout_engine_args,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "tinker":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents

    @staticmethod
    def _get_task_label(env) -> str:
        """Derive a short task label from the environment for logging."""
        # Prefer the authoritative _task_mode from FusedEnv when available
        task_mode = getattr(env, "_task_mode", None)
        if task_mode:
            return task_mode  # "swe", "mcp", or "search"

        entry = getattr(env, "entry", None)
        if entry and isinstance(entry, dict):
            if entry.get("docker_image"):
                docker_image = entry["docker_image"]
                if "gemcli" in docker_image or "gemswe" in docker_image:
                    return "gemcli"
                return "swe"
            if entry.get("tools_py"):
                return "mcp"
            return "search"
        return "other"

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id
        task_label = self._get_task_label(env)
        is_eval = kwargs.get("meta_info", {}).get("validate", False)

        termination_reason = None
        exception_message = ""  # Track exception message for non-ENV_DONE terminations
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []
        seen_queries = set()
        should_discard = False

        # Loop detection: track recent actions to detect repetitive behavior
        recent_actions = []  # List of serialized action strings
        loop_warning_injected = False  # Whether we've already warned the agent
        consecutive_repeat_count = 0  # How many times the same action has repeated
        LOOP_DETECT_THRESHOLD = 3  # Warn after this many identical consecutive actions
        LOOP_TERMINATE_THRESHOLD = 5  # Terminate after this many identical consecutive actions

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx} ({task_label}): initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        # Incremental tokenization: avoid BPE retokenization mismatch by building
        # prompt_ids from the previous step's accumulated IDs + new observation tokens
        accumulated_prompt_ids = None  # Will be set after step 0's model call
        if self.incremental_tokenization:
            eos_token_id = self.tokenizer.eos_token_id
            newline_token_ids = self.tokenizer.encode("\n", add_special_tokens=False)  # [198] for Qwen
            generation_prompt_ids = self.tokenizer.encode(self.chat_parser.generation_prompt, add_special_tokens=False)

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                # max_tokens = max(self.max_response_length - response_token_len, 2048)
                max_tokens = self.max_response_length - response_token_len
            else:
                # max_tokens = max(self.max_response_length, 2048)
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    exception_message = f"Prompt length {prompt_len} exceeded max_prompt_length {self.max_prompt_length}"
                    break

            kwargs["max_tokens"] = max_tokens

            # Build precomputed prompt IDs incrementally to avoid BPE retokenization mismatch
            if self.incremental_tokenization and accumulated_prompt_ids is not None:
                # Tokenize only the new user/tool message appended after the last step
                new_msg = agent.chat_completions[-1]
                new_text = self.chat_parser.parse([new_msg], is_first_msg=False, add_generation_prompt=False)
                new_msg_ids = self.tokenizer.encode(new_text, add_special_tokens=False)

                # Build suffix: \n (after <|im_end|>) + new message tokens + generation prompt
                if accumulated_prompt_ids[-1] == eos_token_id:
                    # Normal completion ended with <|im_end|>, just need \n separator
                    suffix_ids = newline_token_ids + new_msg_ids + generation_prompt_ids
                else:
                    # Truncated completion (finish_reason == "length"), need to close the turn
                    suffix_ids = [eos_token_id] + newline_token_ids + new_msg_ids + generation_prompt_ids

                kwargs["precomputed_prompt_ids"] = accumulated_prompt_ids + suffix_ids
            else:
                kwargs.pop("precomputed_prompt_ids", None)

            # DAPO-styled dynamic sampling: Retry mechanism for handling invalid outputs
            # Small models sometimes struggle with formatting (e.g., JSON compliance in tool calling)
            # Instead of failing, we feed the error back and let the model retry
            max_step_retries = self.config.get("rllm", {}).get("trajectory_filtering", {}).get("max_step_retries", 4)

            retry_count = 0
            validation_success = False
            final_response = None
            final_model_output = None
            retry_prompt_messages = prompt_messages.copy()  # Work with a copy for retries

            while retry_count <= max_step_retries and not validation_success:
                start_time = time.time()
                model_output = await self.get_model_response(retry_prompt_messages, application_id, **kwargs)
                response = model_output.text
                tool_calls = model_output.tool_calls
                finish_reason = model_output.finish_reason

                delta_time = time.time() - start_time
                llm_time += delta_time
                total_time += delta_time

                # Validate step based on tool_calls and \boxed{} presence
                # - Invalid (retry): tool_calls is empty AND "\boxed" is NOT in step
                # - Valid: tool_calls is empty BUT "\boxed" IS in step (final step)
                # - Valid: tool_calls is NOT empty (action step, regardless of \boxed presence), Tool calls prioritize over final answering, encourage progressive tool usage
                if self.validate_boxed_per_step:
                    is_invalid = (len(tool_calls) == 0 if tool_calls else True) and "\\boxed" not in response or finish_reason == "length"
                else:
                    is_invalid = False  # Skip per-step validation when disabled

                if not is_invalid:
                    # Valid output
                    validation_success = True
                    final_response = response
                    final_model_output = model_output
                    break
                else:
                    # Invalid output --> retry
                    retry_count += 1

                    if retry_count > max_step_retries:
                        # Max retries exhausted, treat as abnormal parse error (5.4.1)
                        print("Trajectory:", idx, "Step:", step_idx, "Response:", response, "Tool calls:", tool_calls, "Finish reason:", finish_reason)
                        self._trajectory_logs.append({
                            "type": "retry_exhausted",
                            "trajectory": idx,
                            "step": step_idx,
                            "response": response,
                            "tool_calls": [str(tc) for tc in tool_calls] if tool_calls else [],
                            "finish_reason": finish_reason,
                        })
                        colorful_print(
                            f"Trajectory {idx} ({task_label}), Step {step_idx}: Invalid output after {max_step_retries} retries. " f"No tool calls and no \\boxed{{}} found. Treat as ABNORMAL_PARSE_ERROR.",
                            "yellow",
                        )
                        # Handled outside loop
                        break

                    """
                    # Add error feedback (hint) to conversation for retry
                    error_msg = (
                        "You must either: "
                        "1) Use a tool call (e.g., search) to gather information, OR "
                        "2) Provide a final answer in \\boxed{} format. "
                        "Please provide a valid response."
                    )

                    # Extend retry prompt with failed attempt and error feedback
                    retry_prompt_messages.append({"role": "assistant", "content": response})
                    retry_prompt_messages.append({"role": "user", "content": error_msg})
                    """

                    print("Trajectory:", idx, "Step:", step_idx, "Response:", response, "Tool calls:", tool_calls, "Finish reason:", finish_reason)
                    self._trajectory_logs.append({
                        "type": "retry",
                        "trajectory": idx,
                        "step": step_idx,
                        "retry_count": retry_count,
                        "max_step_retries": max_step_retries,
                        "response": response,
                        "tool_calls": [str(tc) for tc in tool_calls] if tool_calls else [],
                        "finish_reason": finish_reason,
                    })
                    colorful_print(
                        f"Trajectory {idx} ({task_label}), Step {step_idx}: Invalid output (retry {retry_count}/{max_step_retries}): " f"No tool calls and no \\boxed{{}}, retrying.",
                        "yellow",
                    )
                    continue

            # 5.4.1 Handle abnormal trajectories: Parse Error
            if retry_count >= max_step_retries and not validation_success:
                # print(f"Error parsing step after {retry_count} retries: {response}")
                termination_reason = "ABNORMAL_PARSE_ERROR"
                exception_message = f"Failed to parse valid output after {retry_count} retries. No tool calls and no \\boxed{{}} found."
                reward = 0.0
                done = True
                cur_step = agent.get_current_state()
                if cur_step is not None:
                    cur_step.reward = reward
                    cur_step.done = done
                break

            # Use the final response (successful or last attempt after max retries)
            prompt_messages = retry_prompt_messages
            response = final_response
            model_output = final_model_output
            tool_calls = model_output.tool_calls

            # 5.4.1 Handle abnormal trajectories: Tool Burst (exceeds max_tool_calls_per_turn)
            if tool_calls and len(tool_calls) > self.max_tool_calls_per_turn:
                termination_reason = "ABNORMAL_TOOL_BURST"
                exception_message = f"Tool burst detected: {len(tool_calls)} tool calls in a single step (max {self.max_tool_calls_per_turn} allowed)"
                reward = 0.0
                done = True
                cur_step = agent.get_current_state()
                if cur_step is not None:
                    cur_step.reward = reward
                    cur_step.done = done
                break

            # 5.4.1 Handle abnormal trajectories: Repeated Query
            is_repeated = False
            if tool_calls:
                for tool_call in tool_calls:
                    if hasattr(tool_call, "function") and tool_call.function.name == "web_search":
                        try:
                            # Attempt to parse arguments to find query
                            args_str = tool_call.function.arguments
                            # Handle simple JSON parsing if needed, though arguments usually string
                            # We assume simple string check or parsed dict
                            if isinstance(args_str, str):
                                try:
                                    args = json.loads(args_str)
                                except:
                                    args = {}
                            else:
                                args = args_str

                            query = args.get("query")
                            if query:
                                if query in seen_queries:
                                    is_repeated = True
                                    break
                                seen_queries.add(query)
                        except Exception:
                            # If parsing fails, ignore (or could be strict)
                            pass

            if is_repeated:
                termination_reason = "ABNORMAL_REPEATED_QUERY"
                exception_message = f"Repeated query detected: Agent generated the same query multiple times"
                reward = 0.0
                done = True
                cur_step = agent.get_current_state()
                if cur_step is not None:
                    cur_step.reward = reward
                    cur_step.done = done
                break

            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            episode_steps.append(prompt_response_pair)

            # Update accumulated prompt IDs for incremental tokenization
            if self.incremental_tokenization:
                accumulated_prompt_ids = list(model_output.prompt_ids) + list(model_output.completion_ids)

            # Update agent with model response — may return multiple actions
            actions_result = agent.update_from_model(response)
            # Backward compatibility: wrap single Action in a list
            if isinstance(actions_result, Action):
                actions_result = [actions_result]

            # Enforce max_tool_calls_per_turn: truncate parsed actions to the limit
            if len(actions_result) > self.max_tool_calls_per_turn:
                colorful_print(
                    f"Trajectory {idx} ({task_label}), Step {step_idx}: Truncating {len(actions_result)} "
                    f"parsed tool calls to max_tool_calls_per_turn={self.max_tool_calls_per_turn}.",
                    "yellow",
                )
                actions_result = actions_result[:self.max_tool_calls_per_turn]

            # --- Loop detection: check for repetitive actions ---
            # Serialize all actions into one string for multi-action comparison
            action_str = "|".join(str(a.action).strip() for a in actions_result if a.action)
            if action_str:
                recent_actions.append(action_str)
                if len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
                    consecutive_repeat_count += 1
                else:
                    consecutive_repeat_count = 0
                    loop_warning_injected = False

                if consecutive_repeat_count >= LOOP_TERMINATE_THRESHOLD:
                    termination_reason = "ABNORMAL_ACTION_LOOP"
                    exception_message = f"Action loop detected: {consecutive_repeat_count + 1} identical consecutive actions - {action_str[:200]}"
                    reward = 0.0
                    done = True
                    cur_step = agent.get_current_state()
                    if cur_step is not None:
                        cur_step.reward = reward
                        cur_step.done = done
                    colorful_print(
                        f"Trajectory {idx} ({task_label}), Step {step_idx}: Terminated due to action loop "
                        f"({consecutive_repeat_count + 1} identical consecutive actions).",
                        "red",
                    )
                    self._trajectory_logs.append({
                        "type": "action_loop_terminated",
                        "trajectory": idx,
                        "step": step_idx,
                        "repeated_action": action_str[:200],
                        "repeat_count": consecutive_repeat_count + 1,
                    })
                    break

            # --- Execute all tool calls from this model turn ---
            num_intermediate = 0  # Count of intermediate observations appended
            next_observation = None
            step_terminated = False

            for action_idx, act in enumerate(actions_result):
                action = act.action
                is_last_action = (action_idx == len(actions_result) - 1)

                start_time = time.time()
                try:
                    obs, rew, d, inf = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
                except asyncio.TimeoutError:
                    termination_reason = "ENV_TIMEOUT"
                    exception_message = f"Environment step timed out after {self.trajectory_timeout - total_time:.2f}s"
                    should_discard = True
                    colorful_print(f"Warning: Trajectory {idx} ({task_label}) completed due to: {termination_reason}. Discarding trajectory.\n", "red")
                    cur_step = agent.get_current_state()
                    done = True
                    if cur_step is not None:
                        cur_step.done = done
                    step_terminated = True
                    break

                delta_time = time.time() - start_time
                env_time += delta_time
                total_time += delta_time

                if is_last_action or d:
                    # Last action or env signaled done: this becomes the final observation
                    next_observation = obs
                    reward = rew
                    done = d
                    info = inf
                    break
                else:
                    # Intermediate action: append tool_response to agent messages
                    agent.update_from_env_intermediate(
                        observation=obs, reward=rew, done=d, info=inf,
                    )
                    num_intermediate += 1

                    # Check timeout between intermediate actions
                    if total_time >= self.trajectory_timeout:
                        next_observation = obs
                        reward = rew
                        done = False
                        info = inf
                        break

            # If the multi-action loop ended due to timeout, break outer loop
            if step_terminated:
                break

            # If no observation was produced (shouldn't happen, but guard)
            if next_observation is None:
                next_observation = ""
                reward = 0.0
                done = False
                info = {}

            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # --- Loop detection: inject warning into observation if repeating ---
            if consecutive_repeat_count >= LOOP_DETECT_THRESHOLD and not loop_warning_injected:
                loop_warning = (
                    "\n\n[LOOP DETECTED] You have repeated the same action "
                    f"{consecutive_repeat_count + 1} times consecutively. "
                    "This approach is NOT working. You MUST try a DIFFERENT strategy immediately:\n"
                    "- If an edit keeps failing, view the file first to check the current content\n"
                    "- If a command keeps erroring, investigate why (check paths, syntax, dependencies)\n"
                    "- If you're stuck, step back and reconsider the root cause\n"
                    "- Try a completely different approach to solve the problem\n"
                    "DO NOT repeat the same action again."
                )
                next_observation = str(next_observation) + loop_warning
                loop_warning_injected = True
                colorful_print(
                    f"Trajectory {idx} ({task_label}), Step {step_idx}: Loop warning injected "
                    f"({consecutive_repeat_count + 1} identical consecutive actions).",
                    "yellow",
                )

            # Update agent internal state (final observation for this model turn).
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            # --- Incremental tokenization: include intermediate tool responses ---
            # When multiple tool calls were executed, intermediate <tool_response>
            # messages were appended by update_from_env_intermediate(). We must
            # include them in accumulated_prompt_ids so the next iteration's
            # incremental build (which only tokenizes chat_completions[-1]) works.
            if self.incremental_tokenization and num_intermediate > 0 and accumulated_prompt_ids is not None:
                # Tokenize all intermediate messages that were inserted between
                # the assistant message and the final update_from_env user message.
                # They are at positions: -(num_intermediate + 1) to -2 in chat_completions
                # (the last message is from update_from_env, the ones before it are intermediates)
                intermediate_messages = agent.chat_completions[-(num_intermediate + 1):-1]
                for msg in intermediate_messages:
                    msg_text = self.chat_parser.parse([msg], is_first_msg=False, add_generation_prompt=False)
                    msg_ids = self.tokenizer.encode(msg_text, add_special_tokens=False)
                    if accumulated_prompt_ids and accumulated_prompt_ids[-1] == eos_token_id:
                        accumulated_prompt_ids = accumulated_prompt_ids + newline_token_ids + msg_ids
                    else:
                        accumulated_prompt_ids = accumulated_prompt_ids + [eos_token_id] + newline_token_ids + msg_ids

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update response token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                truncation_length = self.max_response_length - response_token_len
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                exception_message = f"Response length {response_token_len - len(env_msg_tokens)} exceeded max_response_length {self.max_response_length}"
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                exception_message = f"Trajectory timeout: total time {total_time:.2f}s exceeded limit {self.trajectory_timeout}s"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                # 5.4.3 Exceeding search step limit: stop + 0 reward
                termination_reason = "MAX_STEPS"
                exception_message = f"Maximum steps reached: {self.max_steps} steps completed without environment termination"
                reward = 0.0  # Force 0 reward

        # Enforce ReAct workflow: >= 5 steps and only enable odd number of steps
        # Also filter out trajectories ending with a tool call but no boxed answer
        if not should_discard and self.enforce_react_structure:
            step_count = len(episode_steps)
            if step_count < 5:
                termination_reason = "INVALID_REACT_STRUCTURE"
                exception_message = f"Invalid ReAct structure: only {step_count} steps (minimum 5 required)"
                # raise InvalidReactStructureError(f"{termination_reason} (Steps: {step_count})")
                should_discard = True
            else:
                # Only check last response if structure is valid (implies step_count >= 5, so steps exist)
                last_response = episode_steps[-1]["response"]
                # if "<tool_call>" in last_response or "</tool_call>" in last_response or "\\boxed" not in last_response:
                if "\\boxed" in last_response:
                    # termination_reason = "INVALID_FINAL_STEP"
                    termination_reason = "ENV_DONE"
                    # should_discard = True
                    # colorful_print(f"Trajectory {idx} discarded: {termination_reason} (Last step has tool call but no boxed)", "yellow")
                    # colorful_print(f"Trajectory {idx} completed: {termination_reason} (Last step has boxed answer)", "green")
                else:
                    termination_reason = "INVALID_FINAL_STEP"
                    exception_message = f"Invalid final step: last step does not contain \\boxed{{}} answer"
                    # raise InvalidReactStructureError(f"Trajectory {idx} discarded: {termination_reason} (Last step has tool call but no boxed)")
                    should_discard = True

        # 5.4.2 Search errors: discard directly
        if should_discard:
            await loop.run_in_executor(self.executor, env.close)
            dropped_result = {
                "idx": env.idx,
                "dropped": True,
                "termination_reason": termination_reason,
                "chat_completions": agent.chat_completions,
                "steps": episode_steps,
            }
            self._trajectory_logs.append({
                "type": "trajectory",
                "idx": env.idx,
                "task_label": task_label,
                "dropped": True,
                "termination_reason": termination_reason,
                "reward": 0.0,
                "num_steps": len(episode_steps),
                "chat_completions": agent.chat_completions,
            })
            return dropped_result

        masked_out = False
        if self.overlong_filter and not is_eval:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        # Calculate final reward if not stopped abnormally
        abnormal_reasons = {"ABNORMAL_PARSE_ERROR", "ABNORMAL_TOOL_BURST", "ABNORMAL_REPEATED_QUERY", "ABNORMAL_ACTION_LOOP", "INVALID_REACT_STRUCTURE", "INVALID_FINAL_STEP"}
        reward_debug = {}
        reward_metadata = {}
        reward_time = 0.0
        final_reward_computed = False
        if hasattr(env, "compute_final_reward") and not masked_out and termination_reason not in abnormal_reasons:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            final_reward_computed = True
            cur_step.reward = reward
            reward_debug = getattr(env, "reward_debug", {})
            reward_metadata = reward_debug if isinstance(reward_debug, dict) else {}
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            n_steps = len(agent.trajectory.steps) if hasattr(agent, 'trajectory') else step_idx + 1
            colorful_print(
                f"Trajectory {idx} ({task_label}: {n_steps} steps) completed due to: {termination_reason}. Reward is {reward}.",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} ({task_label}) is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        # Log the completed trajectory
        self._trajectory_logs.append({
            "type": "trajectory",
            "idx": env.idx,
            "task_label": task_label,
            "dropped": False,
            "termination_reason": termination_reason,
            "reward": trajectory.reward,
            "num_steps": len(trajectory.steps),
            "chat_completions": agent.chat_completions,
        })

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, is_valid_trajectory = self.assemble_steps(episode_steps)

            reward_metrics = {}
            if trajectory.steps:
                last_step = trajectory.steps[-1]
                if "metadata" in last_step.info:
                    metadata = last_step.info["metadata"]
                    if not reward_metadata:
                        reward_metadata = metadata  # Store full metadata

                    # Extract individual reward components for separate logging
                    # These will be logged as traj/rewards/pass@1, traj/rewards/tool_call, etc.
                    if "f1_score" in metadata:
                        reward_metrics["rewards/pass@1"] = metadata["f1_score"]
                    if "exact_match" in metadata:
                        reward_metrics["rewards/exact_match"] = 1.0 if metadata["exact_match"] else 0.0
                    if "step_bonus" in metadata:
                        reward_metrics["rewards/step_bonus"] = metadata["step_bonus"]
                    if "base_reward" in metadata:
                        reward_metrics["rewards/base_reward"] = metadata["base_reward"]
                    """
                    if "base_reward" in metadata:
                        reward_metrics["rewards/base_reward"] = metadata["base_reward"]
                    if "tool_call_reward" in metadata:
                        reward_metrics["rewards/tool_call"] = metadata["tool_call_reward"]
                    if "repetition_penalty_reward" in metadata and metadata["repetition_penalty_reward"] is not None:
                        reward_metrics["rewards/repetition_penalty"] = metadata["repetition_penalty_reward"]
                    """

                    # Check if step.reward contains intermediate rewards
                    # Sum all intermediate step rewards as a separate metric
                    intermediate_rewards = sum(step.reward for step in trajectory.steps[:-1])
                    if intermediate_rewards != 0:
                        reward_metrics["rewards/intermediate_steps"] = intermediate_rewards

            if reward_metadata:
                if "tests_passed" in reward_metadata:
                    reward_metrics["rewards/tests_passed"] = reward_metadata["tests_passed"]
                if "tests_failed" in reward_metadata:
                    reward_metrics["rewards/tests_failed"] = reward_metadata["tests_failed"]
                if "tests_total" in reward_metadata:
                    reward_metrics["rewards/tests_total"] = reward_metadata["tests_total"]
                if "pass_rate" in reward_metadata:
                    reward_metrics["rewards/pass_rate"] = reward_metadata["pass_rate"]
                if "resolved" in reward_metadata:
                    reward_metrics["rewards/resolved"] = 1.0 if reward_metadata["resolved"] else 0.0
                if "pytest_error_code" in reward_metadata and reward_metadata["pytest_error_code"] is not None:
                    raw_error_code = reward_metadata["pytest_error_code"]
                    try:
                        reward_metrics["rewards/pytest_error_code"] = float(raw_error_code)
                    except (ValueError, TypeError):
                        # error_code can be a string like "Error: Exit code 2" from Docker runtime
                        import re
                        match = re.search(r'(\d+)\s*$', str(raw_error_code))
                        reward_metrics["rewards/pytest_error_code"] = float(match.group(1)) if match else -1.0
                reward_metrics["rewards/verifier_missing"] = 0.0
                reward_metrics["rewards/verifier_error"] = 1.0 if reward_metadata.get("verifier_error") else 0.0
            else:
                reward_metrics["rewards/verifier_missing"] = 1.0

            token_result = {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                "trajectory_reward": trajectory.reward,
                "reward_metadata": reward_metadata,  # Add reward metadata for GDPO
                "reward_debug": reward_debug,
                "idx": env.idx,
                "termination_reason": termination_reason,
                "exception": exception_message,  # Add exception message for non-ENV_DONE terminations
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Task type label for logging
                    "task_label": task_label,
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    "token_mismatch": 0.0 if is_valid_trajectory else 1.0,
                    "reward_computed": 1.0 if final_reward_computed else 0.0,
                    # Add individual reward components
                    **reward_metrics,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "reward_metadata": reward_metadata,  # Add reward metadata for GDPO
                "reward_debug": reward_debug,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
                "termination_reason": termination_reason,
            }
            return steps_result
        else:
            raise ValueError(f"Mode {mode} not supported")

    def assemble_steps(self, steps: list[dict]):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Each step_result contains:
        - steps: List of {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list}

        For training, we need to assemble the full conversation sequence where:
        - prompt_tokens: Initial prompt (first step's prompt_ids)
        - response_tokens: All subsequent conversation (completion_ids + next step's prompt_ids)
        - response_masks: Mask indicating which tokens contribute to loss (only completion_ids)
        """

        # Start with initial prompt from first step
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        is_valid_trajectory = True

        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]

            if i == 0:
                # First step: just add completion
                response_tokens.extend(current_completion_ids)
                response_masks.extend([1] * len(current_completion_ids))  # completion contributes to loss
                accumulated_sequence.extend(current_completion_ids)
            else:
                if current_prompt_ids[: len(accumulated_sequence)] != accumulated_sequence:
                    # Find the first differing position
                    prefix = current_prompt_ids[: len(accumulated_sequence)]
                    diff_pos = None
                    for i, (expected, actual) in enumerate(zip(accumulated_sequence, prefix, strict=False)):
                        if expected != actual:
                            diff_pos = i
                            break

                    if diff_pos is not None:
                        logger.warning(f"When assemble steps, detect the trajectory not accumulative at position {diff_pos}. Expected: {accumulated_sequence[diff_pos : diff_pos + 5]}, Got: {prefix[diff_pos : diff_pos + 5]}. Setting response_masks to all 0s. This is likely due to retokenization.")
                    else:
                        logger.warning(f"When assemble steps, detect length mismatch. Expected length: {len(accumulated_sequence)}, Got length: {len(prefix)}. Setting response_masks to all 0s.")

                    is_valid_trajectory = False
                    break

                response_tokens.extend(current_prompt_ids[len(accumulated_sequence) :] + current_completion_ids)
                response_masks.extend([0] * (len(current_prompt_ids) - len(accumulated_sequence)) + [1] * len(current_completion_ids))  # completion contributes to loss
                accumulated_sequence = current_prompt_ids + current_completion_ids

        assert len(response_masks) == len(response_tokens)

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)

        if self.config.rllm.filter_token_mismatch:
            response_masks = response_masks * int(is_valid_trajectory)

        return prompt_tokens, response_tokens, response_masks, is_valid_trajectory

    async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
        # Allow up to 8 retries for InvalidReactStructureError, but respect self.retry_limit for others
        max_attempts = max(self.retry_limit, 2) + 1
        task_label = self._get_task_label(self.envs[idx])

        for attempt in range(max_attempts):
            # Fast-fail if Docker daemon has been detected as down by another trajectory
            if self._docker_healthy is not None and not self._docker_healthy.is_set():
                colorful_print(f"Trajectory {idx} ({task_label}) skipped: Docker daemon is unreachable (detected by another trajectory).", "red")
                self._trajectory_logs.append({
                    "type": "trajectory",
                    "idx": idx,
                    "dropped": True,
                    "termination_reason": "DOCKER_UNHEALTHY",
                    "reward": 0.0,
                    "num_steps": 0,
                    "chat_completions": [],
                })
                return None

            try:
                application_id = str(uuid.uuid4())
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=self.trajectory_timeout)
            except (TimeoutError, asyncio.TimeoutError) as e:
                colorful_print(f"Trajectory {idx} ({task_label}) timed out after {self.trajectory_timeout}s. Dropping trajectory.", "red")
                self._trajectory_logs.append({
                    "type": "trajectory",
                    "idx": idx,
                    "dropped": True,
                    "termination_reason": "TRAJECTORY_TIMEOUT",
                    "reward": 0.0,
                    "num_steps": 0,
                    "chat_completions": [],
                })
                return None
            except InvalidReactStructureError as e:
                # Retry `max_attempts` times for this specific error
                if attempt < max_attempts - 1:
                    colorful_print(f"Trajectory {idx} ({task_label}) retry {attempt}/{max_attempts-1} due to: {e}", "yellow")
                    continue
                else:
                    colorful_print(f"Trajectory {idx} ({task_label}) failed due to INVALID_REACT_STRUCTURE after {attempt} retries.", "pink")
                    self._trajectory_logs.append({
                        "type": "trajectory",
                        "idx": idx,
                        "dropped": True,
                        "termination_reason": "INVALID_REACT_STRUCTURE",
                        "reward": 0.0,
                        "num_steps": 0,
                        "chat_completions": [],
                    })
                    return None
            except Exception as _:
                # Detect Docker connection errors and signal all trajectories to stop
                if _is_docker_connection_error(_):
                    if self._docker_healthy is not None:
                        self._docker_healthy.clear()  # Signal all trajectories
                    colorful_print(f"Trajectory {idx} ({task_label}) failed due to Docker connection error: {_}. Signaling all trajectories to stop.", "red")
                    self._trajectory_logs.append({
                        "type": "trajectory",
                        "idx": idx,
                        "dropped": True,
                        "termination_reason": "DOCKER_CONNECTION_ERROR",
                        "reward": 0.0,
                        "num_steps": 0,
                        "chat_completions": [],
                    })
                    return None
                # For other exceptions, respect self.retry_limit (total self.retry_limit attempts)
                if attempt < max_attempts - 1:
                    traceback.print_exc()
                    colorful_print(f"Trajectory {idx} ({task_label}) retry {attempt}/{max_attempts-1} due to exception: {_}", "yellow")
                    continue
                else:
                    traceback.print_exc()
                    colorful_print(f"Trajectory {idx} ({task_label}) cannot complete after {self.retry_limit} retries. Skipping this trajectory.", "red")
                    self._trajectory_logs.append({
                        "type": "trajectory",
                        "idx": idx,
                        "dropped": True,
                        "termination_reason": f"EXCEPTION: {type(_).__name__}: {_}",
                        "reward": 0.0,
                        "num_steps": 0,
                        "chat_completions": [],
                    })
                    return None
        return None

    def _dump_trajectory_logs(self):
        """Dump all collected trajectory logs to trajs.json (append to existing entries)."""
        if not self._trajectory_logs:
            return
        # Read existing entries from file if present
        existing = []
        try:
            with open("experiments/logs/trajs.json", "r") as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []
        existing.extend(self._trajectory_logs)
        with open("experiments/logs/trajs.json", "w") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Dumped {len(self._trajectory_logs)} trajectory logs to experiments/logs/trajs.json (total: {len(existing)})")
        self._trajectory_logs = []

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        # Reset Docker health flag for this generation round (threading.Event: set = healthy)
        self._docker_healthy = threading.Event()
        self._docker_healthy.set()  # Assume healthy until proven otherwise

        if self.engine_name == "verl":
            await self.rollout_engine.wake_up()  # type: ignore

        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        async def launch_one_trajectory_task(env_idx: int):
            async with semaphore:
                try:
                    result = await self.run_agent_trajectory_with_retry(
                        idx=env_idx,
                        seed=reset_seed,
                        mode=mode,
                        **kwargs,
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise e
                return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                steps = result.get("metrics", {}).get("steps") if isinstance(result, dict) else None
                label = result.get("metrics", {}).get("task_label") if isinstance(result, dict) else None
                info_parts = []
                if label:
                    info_parts.append(label)
                if steps is not None:
                    info_parts.append(f"{steps} steps")
                info_suffix = f" ({': '.join(info_parts)})" if info_parts else ""
                colorful_print(
                    f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed{info_suffix}\n",
                    "cyan",
                )
                # Dump trajectory logs after each trajectory completes
                # self._dump_trajectory_logs()
                if result is not None:
                    yield result
            except Exception as e:
                # Dump before propagating exception to avoid losing logs
                # self._dump_trajectory_logs()
                raise e

        # Final dump to ensure all remaining logs are flushed (e.g. when all trajectories failed)
        # self._dump_trajectory_logs()

        if self.engine_name == "verl":
            await self.rollout_engine.sleep()  # type: ignore

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]

        self._dump_trajectory_logs()
        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
