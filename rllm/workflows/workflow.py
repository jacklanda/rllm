import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from enum import Enum
from functools import partial

import numpy as np

from rllm.agents.agent import BaseAgent
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.environments.base.base_env import BaseEnv
from rllm.types import Episode, Trajectory
from rllm.workflows.store import Store

# Canonical task-source categories used for per-channel rollout metrics
# (e.g. ``traj/steps/mcp``). Keep this list in sync with the keys produced by
# ``infer_task_source`` so the trainer can split metrics per data source.
TASK_SOURCE_CHANNELS = ("mcp", "web search", "cli")

# Map a normalized task source to the short suffix used in metric keys.
_TASK_SOURCE_METRIC_SUFFIX = {"mcp": "mcp", "web search": "search", "cli": "cli"}


def _normalize_task_source(task_type: object) -> str | None:
    """Normalize a raw task-type/data-source label into a metric channel."""
    if task_type is None:
        return None
    value = str(task_type).strip().lower().replace("_", " ").replace("-", " ")
    if not value:
        return None
    if value in {"mcp"}:
        return "mcp"
    if value in {"web search", "websearch", "search", "web"}:
        return "web search"
    if value in {"cli", "swe", "swe bench", "gemcli", "gemswe", "et", "endless terminals"}:
        return "cli"
    return value


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def infer_task_source(task: object) -> str:
    """Infer the broad task source ("mcp", "web search", "cli", ...) for logging.

    This is the canonical implementation shared by the workflow engine's
    progress logging and the per-channel metric aggregation in
    :meth:`Workflow.collect_metrics`.
    """
    if isinstance(task, str):
        try:
            task = json.loads(task)
        except (json.JSONDecodeError, ValueError):
            return "unknown"

    if not isinstance(task, Mapping):
        return "unknown"

    task_type = _normalize_task_source(task.get("task_type"))
    if task_type is not None:
        return task_type

    data_source = _normalize_task_source(task.get("data_source"))
    if data_source in TASK_SOURCE_CHANNELS:
        return data_source

    if task.get("tools_py"):
        return "mcp"
    if task.get("docker_image"):
        return "cli"
    if task.get("data_source"):
        return "web search"

    return "unknown"


def _step_has_tool_call(step) -> bool:
    """Best-effort detection of whether a step issued at least one tool call.

    Works across the agent families in this repo:
    - OpenAI-style agents put a ``tool_calls`` list on the assistant message.
    - Qwen/SWE-style agents embed ``<tool_call>`` tags in ``model_response``.
    - Some agents record an explicit count in ``step.info``.
    """
    info = step.info if isinstance(getattr(step, "info", None), dict) else {}
    for key in ("tool_calls", "num_tool_calls", "tool_call_count"):
        value = info.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, (int, float)) and value > 0:
            return True
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            return True

    chat_completions = getattr(step, "chat_completions", None)
    if chat_completions:
        for msg in reversed(chat_completions):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                return True
            content = msg.get("content")
            if isinstance(content, str) and "<tool_call" in content:
                return True
            break  # only inspect the final assistant message

    model_response = getattr(step, "model_response", "") or ""
    if isinstance(model_response, str) and "<tool_call" in model_response:
        return True

    action = getattr(step, "action", None)
    if isinstance(action, str) and action.strip():
        return True

    return False


def count_tool_call_turns(trajectory: Trajectory) -> int:
    """Count the number of steps in which the agent issued >=1 tool call."""
    return sum(1 for step in trajectory.steps if _step_has_tool_call(step))


class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"


class TerminationEvent(Exception):
    def __init__(self, reason: TerminationReason = TerminationReason.UNKNOWN):
        super().__init__(f"Terminated: {reason}")
        self.reason = reason


class Workflow(ABC):
    _GENERATION_LOOP_PHRASES = (
        "the tool call is a function",
        "the tool response is a",
        "valid json schema",
        "the user message is",
        "the user is a person",
    )
    _GENERATION_LOOP_SYMBOL_RE = re.compile(r"([{}<>\[\]()/])\1{80,}")
    _GENERATION_LOOP_TOKEN_RE = re.compile(r"\S+")
    _GENERATION_LOOP_MIN_REPEATED_TOKENS = 40

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        timeout=1e6,
        gamma=0.0,
        reward_bonus_coeff=0.0,
        store: Store | None = None,
        strict_eval_accuracy: bool = False,
        **kwargs,
    ):
        """Initialize the Workflow.

        Args:
            rollout_engine: The rollout engine to use.
            executor: The executor to use.
            timeout: The timeout for the workflow.
            gamma: The discount factor for the workflow.
            reward_bonus_coeff: The reward bonus coefficient for the workflow.
            store: Optional cross-episode store shared across all workflow
                instances.  See :class:`rllm.workflows.store.Store`.
            strict_eval_accuracy: When validation is running, count an episode
                as correct only under the evaluator's strict success signal
                (exact match for answer-F1 tasks, full pass rate for tests,
                resolved for issue-style tasks, or full reward fallback).
            **kwargs: Additional keyword arguments.
        """
        self.rollout_engine = rollout_engine
        self.executor = executor
        self.timeout = int(timeout)
        self.gamma = gamma
        self.reward_bonus_coeff = reward_bonus_coeff
        self.strict_eval_accuracy = _coerce_bool(strict_eval_accuracy)
        self.store = store

        self._completed_trajectories: list[Trajectory] = []

    @classmethod
    def detect_abnormal_generation(cls, response: str) -> tuple[bool, str]:
        """Detect pathological assistant output before it reaches the env.

        This catches the common failure mode where the model decodes tool
        schema/meta text until max_response_length instead of emitting one
        useful tool call. It is intentionally phrase/ngram based; ordinary
        long reasoning should not trip it unless it contains a clear loop.
        """
        if not response:
            return False, ""
        lowered = response.lower()
        for phrase in cls._GENERATION_LOOP_PHRASES:
            count = lowered.count(phrase)
            if count >= 8:
                return True, f"phrase {phrase!r} repeated {count} times"
        symbol_match = cls._GENERATION_LOOP_SYMBOL_RE.search(response)
        if symbol_match:
            return True, f"symbol {symbol_match.group(1)!r} repeated excessively"

        tokens = cls._GENERATION_LOOP_TOKEN_RE.findall(lowered)
        same_run = 1
        prev = None
        for tok in tokens:
            if tok == prev:
                same_run += 1
                if same_run >= cls._GENERATION_LOOP_MIN_REPEATED_TOKENS:
                    return True, f"token {tok!r} repeated {same_run} times consecutively"
            else:
                same_run = 1
                prev = tok

        # Repeated schema loops are usually phrase-level, not one-token loops.
        # Check aligned repeated ngrams across several offsets so a loop can
        # begin anywhere in the response.
        for n in (3, 4, 5, 8, 12, 16):
            min_runs = 10 if n <= 4 else 6
            if len(tokens) < n * min_runs:
                continue
            for offset in range(n):
                run = 1
                prev_ngram = None
                for i in range(offset, len(tokens) - n + 1, n):
                    ngram = tuple(tokens[i : i + n])
                    if ngram == prev_ngram:
                        run += 1
                        if run >= min_runs and run * n >= cls._GENERATION_LOOP_MIN_REPEATED_TOKENS:
                            preview = " ".join(ngram[:8])
                            return True, f"{n}-gram loop repeated {run} times: {preview!r}"
                    else:
                        run = 1
                        prev_ngram = ngram
        return False, ""

    def mark_abnormal_generation(
        self,
        response: str,
        reason: str,
        termination_reason: TerminationReason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
    ) -> None:
        """Attach a visible negative guard signal to the latest agent step."""
        debug = {
            "type": "rollout_guard",
            "reward": -1.0,
            "resolved": False,
            "reward_mode": "generation_guard",
            "reward_source": "workflow_generation_guard",
            "is_correct": False,
            "termination_reason": termination_reason.value,
            "guard/abnormal_generation": True,
            "guard/reason": reason,
            "guard/response_chars": len(response or ""),
        }

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, BaseAgent) or not hasattr(attr_value, "trajectory"):
                continue
            trajectory = attr_value.trajectory
            if not trajectory.steps:
                continue
            step = trajectory.steps[-1]
            step.reward = -1.0
            step.done = True
            step.info["reward_debug"] = debug
            step.info["rollout_guard"] = debug
            trajectory.info["reward_debug"] = debug
            break

    @abstractmethod
    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute the workflow on a single task

        Args:
            task: The task to execute.
            uid: The unique identifier for the task.
            **kwargs: Additional keyword arguments.

        Returns:
            Episode: The episode generated by the workflow.
        """
        pass

    async def run_with_termination_handling(self, task: dict, uid: str, **kwargs) -> Episode:
        """Wrapper method around workflow.run that handles termination events, errors, timeouts, and post-processing.

        Args:
            task: The task to execute.
            uid: The unique identifier for the task.
            **kwargs: Additional keyword arguments.
        """
        # Bind these before entering user workflow code. A timeout/cancellation can
        # happen while reset itself is still running, before Workflow.reset assigns
        # uid/task, but postprocess_episode still needs stable episode metadata.
        self.uid = uid
        self.task = task
        # Extract timeout from kwargs, default to self.timeout
        timeout = kwargs.pop("timeout", self.timeout)

        try:
            coro = self.run(task, uid, **kwargs)
            output = await asyncio.wait_for(coro, timeout=timeout)
            if output is not None and isinstance(output, Episode):
                return output  # we assume it's already postprocessed
            return self.postprocess_episode(self.collect_trajectories(), TerminationReason.UNKNOWN)
        except asyncio.TimeoutError:
            return self.postprocess_episode(self.collect_trajectories(), TerminationReason.TIMEOUT)
        except TerminationEvent as e:
            return self.postprocess_episode(self.collect_trajectories(), e.reason)
        except Exception as e:
            import traceback

            error_details = {"error_message": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
            return self.postprocess_episode(self.collect_trajectories(), TerminationReason.ERROR, error=error_details)

    def commit(self, name: str | None = None, agent: BaseAgent | None = None, trajectory: Trajectory | None = None, reset: bool = False) -> None:
        """Commit a trajectory for training.

        Args:
            name: The name of the trajectory.
            agent: The agent that generated the trajectory.
            trajectory: The trajectory to commit.
            reset: Whether to reset the agent.
        """
        assert agent is not None or trajectory is not None, "Either agent or trajectory must be provided to workflow.commit"
        assert agent is None or trajectory is None, "Only one of agent or trajectory can be provided to workflow.commit"

        traj = agent.trajectory if agent is not None else trajectory
        if name:
            traj.name = name
        if traj.steps:
            self._completed_trajectories.append(deepcopy(traj))

        if agent is not None and reset:
            agent.reset()

    def collect_trajectories(self) -> Episode:
        """Collect the trajectories from the workflow

        Returns:
            Episode: The episode generated by the workflow.
        """

        episode = Episode()

        # Start with completed trajectories
        episode.trajectories.extend(self._completed_trajectories)

        # Track completed trajectory uids
        completed_trajectory_uids = {trajectory.uid for trajectory in self._completed_trajectories}

        # Add trajectories from agents that aren't already in completed trajectories
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if (
                isinstance(attr_value, BaseAgent)
                and hasattr(attr_value, "trajectory")
                and getattr(attr_value.trajectory, "uid", None) not in completed_trajectory_uids
                and len(attr_value.trajectory.steps) > 0
            ):
                episode.trajectories.append(deepcopy(attr_value.trajectory))

        return episode

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        """
        Compute the trajectory-level reward.
        Default: sum the step rewards

        Args:
            trajectory: The trajectory to compute the reward for.
        """
        trajectory.reward = np.sum([d.reward for d in trajectory.steps])

    def adjust_step_rewards(self, trajectory: Trajectory) -> None:
        """
        Adjust the step-level rewards. Supports reward shaping and discounting
        self.reward_bonus_coeff and self.gamma are 0.0, so no adjustments are made by default.

        Args:
            trajectory: The trajectory to adjust the rewards for.
        """
        # reward shaping
        # s[i].reward = s[i].reward + bonus * (s[i].reward - s[i-1].reward) for i > 0
        if self.reward_bonus_coeff > 0.0:
            raw_rewards = [step.reward for step in trajectory.steps]
            for i in range(1, len(trajectory.steps)):
                trajectory.steps[i].reward += self.reward_bonus_coeff * (raw_rewards[i] - raw_rewards[i - 1])

        # Compute Monte Carlo returns (backward iteration)
        # G_t = R_{t+1} + γ * R_{t+2} + γ² * R_{t+3} + ... + γ^{T-t-1} * R_T
        if self.gamma > 0.0:
            G = 0.0
            for step in reversed(trajectory.steps):
                G = step.reward + self.gamma * G
                step.reward = G  # Replace the reward with MC return

    @staticmethod
    def _debug_has_strict_success(debug: dict) -> bool | None:
        reward_source = str(debug.get("reward_source") or "").lower()
        reward_mode = str(debug.get("reward_mode") or "").lower()
        debug_type = str(debug.get("type") or "").lower()
        is_search_reward_task = reward_source == "search_reward_fn" or reward_mode in {"em", "f1"} or debug_type in {"web search", "search", "web_search"}

        if is_search_reward_task and ("exact_match" in debug or "f1_score" in debug):
            if "exact_match" in debug:
                return bool(debug.get("exact_match"))
            try:
                return float(debug.get("f1_score") or 0.0) >= 1.0
            except (TypeError, ValueError):
                return False

        if "pass_rate" in debug:
            try:
                return float(debug.get("pass_rate") or 0.0) >= 1.0
            except (TypeError, ValueError):
                return False

        if "tests_passed" in debug and "tests_total" in debug:
            try:
                return int(debug.get("tests_total") or 0) > 0 and int(debug.get("tests_passed") or 0) >= int(debug.get("tests_total") or 0)
            except (TypeError, ValueError):
                return False

        if "resolved" in debug:
            return bool(debug.get("resolved"))

        return None

    def _strict_trajectory_correct(self, trajectory: Trajectory, episode: Episode | None = None) -> bool:
        debug_candidates: list[dict] = []
        debug = trajectory.info.get("reward_debug") if isinstance(trajectory.info, dict) else None
        if isinstance(debug, dict):
            debug_candidates.append(debug)
        for step in reversed(trajectory.steps):
            step_info = step.info if isinstance(step.info, dict) else {}
            for key in ("reward_debug", "metadata"):
                step_debug = step_info.get(key)
                if isinstance(step_debug, dict):
                    debug_candidates.append(step_debug)
        if episode is not None and isinstance(episode.info, dict):
            episode_debug = episode.info.get("reward_debug")
            if isinstance(episode_debug, dict):
                debug_candidates.append(episode_debug)

        for debug in debug_candidates:
            strict_success = self._debug_has_strict_success(debug)
            if strict_success is not None:
                return strict_success

        return float(trajectory.reward or 0.0) >= 1.0

    def assign_episode_correctness(self, episode: Episode, *, is_validation: bool = False) -> None:
        """
        Assign an episode-level correctness flag.
        Default: True if the sum of the trajectory rewards is strictly positive.

        Args:
            episode: The episode to assign the correctness flag to.
        """
        if self.strict_eval_accuracy and is_validation:
            episode.is_correct = any(self._strict_trajectory_correct(trajectory, episode) for trajectory in episode.trajectories)
            return

        total_reward = 0
        for trajectory in episode.trajectories:
            total_reward += trajectory.reward or 0
        episode.is_correct = total_reward > 0

    def collect_metrics(self, episode: Episode, *, is_validation: bool = False) -> None:
        """
        Collect metrics from the episode.

        In addition to the per-agent accuracy (``{name}_acc``), this records
        agent-step counts and tool-call-turn counts, both globally and split by
        task source so distinct task families (mcp / web search / cli) get their
        own channels, e.g. ``traj/steps/mcp`` and ``turn/tool_call_turn/mcp``.

        Only the channel matching the episode's task source is populated, so the
        trainer's mean over the batch is taken over just the episodes of that
        source (matching the legacy AgentExecutionEngine behavior).

        Args:
            episode: The episode to collect metrics from.
        """
        metrics = defaultdict(list)
        for traj in episode.trajectories:
            name = traj.name
            if self.strict_eval_accuracy and is_validation:
                metrics[name].append(1.0 if self._strict_trajectory_correct(traj, episode) else 0.0)
            else:
                metrics[name].append(traj.reward or 0.0)
        episode.metrics = {f"{k}_acc": float(np.mean(v)) for k, v in metrics.items()}

        # Aggregate agent steps and tool-call turns across all trajectories.
        total_steps = sum(len(traj.steps) for traj in episode.trajectories)
        total_tool_call_turns = sum(count_tool_call_turns(traj) for traj in episode.trajectories)

        # Global channels (averaged over every episode).
        episode.metrics["traj/steps"] = float(total_steps)
        episode.metrics["turn/tool_call_turn"] = float(total_tool_call_turns)

        # Per-source channels: only populate the key for this episode's source so
        # the batch mean is restricted to episodes of that source.
        source = infer_task_source(getattr(self, "task", None) if episode.task is None else episode.task)
        suffix = _TASK_SOURCE_METRIC_SUFFIX.get(source)
        if suffix is not None:
            episode.metrics[f"traj/steps/{suffix}"] = float(total_steps)
            episode.metrics[f"turn/tool_call_turn/{suffix}"] = float(total_tool_call_turns)

    def postprocess_episode(self, episode: Episode, termination_reason: TerminationReason = None, error: dict = None) -> Episode:
        """Collect and process the trajectories

        Args:
            episode: The episode to postprocess.
            termination_reason: The termination reason for the episode.
            error: The error details for the episode.
        """

        # 1. assign a task id and task
        episode.id = self.uid
        episode.task = self.task

        for trajectory in episode.trajectories:
            # depending on the terminaiton reason, there may be a trajectry with an additional step with empty chat_completions
            # i.e., if it's thrown between agent.update_from_env() and agent.update_from_model()
            if trajectory.steps and not trajectory.steps[-1].chat_completions:
                trajectory.steps.pop()

        if termination_reason != TerminationReason.ENV_DONE:
            for trajectory in reversed(episode.trajectories):
                if not trajectory.steps:
                    continue
                reward_debug = trajectory.steps[-1].info.get("reward_debug")
                if isinstance(reward_debug, dict):
                    trajectory.info["reward_debug"] = reward_debug
                    episode.info["reward_debug"] = reward_debug
                    break

        if termination_reason == TerminationReason.ENV_DONE:
            final_reward = None
            reward_debug = None
            for attr_name in dir(self):
                if attr_name.startswith("_"):
                    continue
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, BaseEnv) and hasattr(attr_value, "compute_final_reward"):
                    final_reward = float(attr_value.compute_final_reward())
                    reward_debug = getattr(attr_value, "reward_debug", None)
                    break

            if final_reward is not None:
                for trajectory in reversed(episode.trajectories):
                    if trajectory.steps:
                        trajectory.steps[-1].reward = final_reward
                        if isinstance(reward_debug, dict):
                            trajectory.steps[-1].info["reward_debug"] = reward_debug
                            trajectory.info["reward_debug"] = reward_debug
                            episode.info["reward_debug"] = reward_debug
                        break

        for trajectory in episode.trajectories:
            # 2. compute trajectory-level rewards
            self.compute_trajectory_reward(trajectory)

            # 3. adjust the step level rewards (e.g., reward shaping or discounting)
            if len(trajectory.steps) > 1:
                self.adjust_step_rewards(trajectory)

        # 4. assign an episode-level correctness flag
        is_validation = bool(getattr(self.rollout_engine, "validate", False))
        self.assign_episode_correctness(episode, is_validation=is_validation)

        # 5. collect additional metrics workflow
        # by default, we report the acc of each agent using the traj reward
        self.collect_metrics(episode, is_validation=is_validation)

        # 6. store error details if provided
        if error is not None:
            episode.info["error"] = error

        # 7. assign a termination reason
        episode.termination_reason = termination_reason or TerminationReason.UNKNOWN

        return episode

    def reset(self, task: dict | None = None, uid: str | None = None) -> None:
        """Reset the workflow

        Args:
            task: The task to reset the workflow to.
            uid: The unique identifier for the task.
        """
        # set the uid and task
        self.uid = uid
        self.task = task
        self._completed_trajectories = []

        # reset agents (look for class attributes that are BaseAgent subclasses)
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseAgent) and hasattr(attr_value, "reset"):
                attr_value.reset()
                attr_value.trajectory.task = task

        # reset environments (look for class attributes that are BaseEnv subclasses)
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv) and hasattr(attr_value, "reset"):
                attr_value.reset(task=task)

    def close(self) -> None:
        """Close resources owned by this workflow.

        Workflows commonly keep a reusable environment instance on ``self.env``.
        Closing it after an episode releases per-task resources (Docker
        containers, MCP subprocesses, etc.) while keeping the workflow object
        itself reusable for the next reset.
        """
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv) and hasattr(attr_value, "close"):
                attr_value.close()

    def is_multithread_safe(self) -> bool:
        """Check if the workflow is multithread safe

        Returns:
            bool: True if the workflow is multithread safe, False otherwise.
        """
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv) and not attr_value.is_multithread_safe():
                return False
        return True

    async def run_in_executor(self, fn, *args, **kwargs):
        """Run a function in seperate thread pool executor.

        Args:
            fn: The function to run.
            *args: The arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
