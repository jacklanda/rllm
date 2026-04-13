import inspect
import json
import logging
import os

from rllm.environments.cli.cli import CLIEnv

logger = logging.getLogger(__name__)

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

try:
    from examples.search.local_retrieval_tool import LocalRetrievalTool
except ImportError:
    try:
        from rllm.tools.web_tools.tavily_tool import TavilySearchTool as LocalRetrievalTool
    except ImportError:
        LocalRetrievalTool = None


class FusedEnv(CLIEnv):
    """Fused environment combining CLI/SWE Docker tools with external web search.

    Operates in two modes based on data type:

    **SWE mode** (entry has ``docker_image``):
        Extends CLIEnv (which extends SWEEnv). Intercepts ``web_search`` tool
        calls and routes them to a ``LocalRetrievalTool`` running outside the Docker
        container, while all other tool calls are delegated to Docker via the parent.

    **Search mode** (entry has ``data_source`` but no ``docker_image``):
        No Docker container is created. Only ``web_search`` and ``finish``/``submit``
        tools are available. Reward is computed via F1-score against ``ground_truth``.
    """

    def __init__(
        self,
        retrieval_server_url: str | None = None,
        retrieval_max_results: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_server_url = retrieval_server_url or os.environ.get(
            "RETRIEVAL_SERVER_URL", "http://127.0.0.1:65432"
        )
        self.retrieval_max_results = retrieval_max_results
        self._retrieval_tool = None  # Lazy-initialized

        # Detect data type: search mode if no docker_image present
        self._is_search_task = not bool(self.entry.get("docker_image"))

        # Search mode state
        self._search_answer = ""  # Agent's submitted answer for reward computation
        self._search_reward_debug = {}

    def _get_retrieval_tool(self):
        """Lazy-initialize the retrieval tool on first web_search call."""
        if self._retrieval_tool is None:
            if LocalRetrievalTool is None:
                logger.warning(
                    "LocalRetrievalTool not available — web_search will return errors"
                )
                return None
            self._retrieval_tool = LocalRetrievalTool(
                server_url=self.retrieval_server_url,
                max_results=self.retrieval_max_results,
            )
        return self._retrieval_tool

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> tuple[str, dict]:
        if self._is_search_task:
            return self._reset_search()
        return self._reset_swe()

    def _reset_search(self) -> tuple[str, dict]:
        """Reset for search-mode tasks (no Docker)."""
        self.total_steps = 0
        self._search_answer = ""
        self._search_reward_debug = {}

        question = self.entry.get("question", self.entry.get("problem_statement", ""))
        return question, {"task_type": "search"}

    def _reset_swe(self) -> tuple[str, dict]:
        """Reset for SWE-mode tasks (Docker container)."""
        obs, info = super().reset()
        info["task_type"] = "swe"
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action):
        if self._is_search_task:
            return self._step_search(action)
        return self._step_swe(action)

    def _step_swe(self, action):
        """SWE-mode step: web_search goes to retrieval tool, everything else to Docker."""
        if SWEAction is None:
            return super().step(action)

        if isinstance(action, str):
            action_obj = SWEAction.from_string(action)
        else:
            action_obj = action

        if action_obj.function_name == "web_search":
            return self._handle_web_search(action_obj)

        return super().step(action)

    def _step_search(self, action):
        """Search-mode step: handle web_search + finish/submit locally, error on Docker tools."""
        if SWEAction is None:
            # Cannot parse actions without r2egym
            self.total_steps += 1
            return "Error: r2egym not available for action parsing.", 0.0, False, {}

        if isinstance(action, str):
            action_obj = SWEAction.from_string(action)
        else:
            action_obj = action

        fn = action_obj.function_name

        if fn == "web_search":
            return self._handle_web_search(action_obj)

        if fn in ("finish", "submit"):
            return self._handle_search_finish(action_obj)

        # Docker-only tools are not available in search mode
        self.total_steps += 1
        return (
            f"Error: The tool '{fn}' is not available for search tasks. "
            "Use web_search to find information and finish to submit your answer.",
            0.0,
            False,
            {},
        )

    def _handle_web_search(self, action_obj) -> tuple[str, float, bool, dict]:
        """Execute a web_search tool call via LocalRetrievalTool."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        query = params.get("query", "")
        top_k = params.get("top_k", None)
        if top_k is not None:
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                top_k = None

        tool = self._get_retrieval_tool()
        if tool is None:
            observation = "Execution output of [web_search]:\nError: web_search tool is not available. LocalRetrievalTool could not be initialized."
            return observation, 0.0, False, {}

        try:
            result = tool.forward(query=query, top_k=top_k)
            observation = f"Execution output of [web_search]:\n{result.to_string()}"
        except Exception as e:
            logger.error("web_search execution failed: %s", str(e))
            observation = f"Execution output of [web_search]:\nError executing web_search: {str(e)}"

        return observation, 0.0, False, {}

    def _handle_search_finish(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle finish/submit tool call in search mode."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", "")
        self._search_answer = result
        return "Your answer has been submitted.", 0.0, True, {}

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def compute_final_reward(self):
        if self._is_search_task:
            return self._compute_search_reward()
        return super().compute_final_reward()

    def compute_final_reward_metadata(self) -> dict:
        if self._is_search_task:
            self._compute_search_reward()
            return self._search_reward_debug
        return super().compute_final_reward_metadata()

    def _compute_search_reward(self) -> float:
        """Compute F1-based reward for search tasks."""
        from rllm.rewards.reward_types import RewardConfig, RewardInput
        from rllm.rewards.search_reward import RewardSearchFn

        ground_truth = self.entry.get("ground_truth", "")
        answer = self._search_answer

        config = RewardConfig(
            toolcall_bonus=0.0,
            apply_repetition_penalty=False,
            enable_step_bonus=False,
        )
        reward_fn = RewardSearchFn(config)
        reward_input = RewardInput(
            task_info={"ground_truth": ground_truth, "step_count": self.total_steps},
            action=answer,
        )
        reward_output = reward_fn(reward_input)

        self._search_reward_debug = {
            "type": "search",
            "reward": float(reward_output.reward),
            "resolved": reward_output.reward >= 1.0,
            "reward_mode": "f1",
            "reward_source": "search_reward_fn",
            "is_correct": reward_output.is_correct,
            "verifier_error": "",
            **reward_output.metadata,
        }
        self._reward_debug = self._search_reward_debug
        return float(reward_output.reward)

    @property
    def reward_debug(self) -> dict:
        if self._is_search_task:
            return self._search_reward_debug
        return self._reward_debug

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self):
        """Clean up resources."""
        if self._retrieval_tool is not None:
            try:
                self._retrieval_tool.client.close()
            except Exception:
                pass
            self._retrieval_tool = None
        if not self._is_search_task:
            super().close()

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(extra_info: dict | str) -> "FusedEnv":
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(FusedEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return FusedEnv(**init_params)
