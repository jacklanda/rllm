import inspect
import json
import logging
import os
import re
import sys
import uuid

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

try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
except ImportError:
    MCPConnectionManager = None
    MCPEnvironment = None


class FusedEnv(CLIEnv):
    """Fused environment combining CLI/SWE Docker tools with external web search and MCP tools.

    Operates in three modes based on data type:

    **SWE mode** (entry has ``docker_image``):
        Extends CLIEnv (which extends SWEEnv). Intercepts ``web_search`` tool
        calls and routes them to a ``LocalRetrievalTool`` running outside the Docker
        container, while all other tool calls are delegated to Docker via the parent.

    **Search mode** (entry has ``data_source`` but no ``docker_image`` or ``tools_py``):
        No Docker container is created. Only ``web_search`` and ``finish``/``submit``
        tools are available. Reward is computed via F1-score against ``ground_truth``.

    **MCP mode** (entry has ``tools_py``):
        No Docker container is created. Connects to an MCP server that serves the
        tools defined in ``tools_py``. Reward is computed via verifier code.
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

        # Detect task mode: SWE, MCP, or Search
        if self.entry.get("docker_image"):
            self._task_mode = "swe"
        elif self.entry.get("tools_py"):
            self._task_mode = "mcp"
        else:
            self._task_mode = "search"

        # Search mode state
        self._search_answer = ""  # Agent's submitted answer for reward computation
        self._search_reward_debug = {}

        # MCP mode state
        self._mcp_connection_manager: "MCPConnectionManager | None" = None
        self._mcp_tool_schemas: list[dict] = []
        self._mcp_answer = ""
        self._mcp_reward_debug: dict = {}
        self._mcp_has_used_tools = False

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
        if self._task_mode == "mcp":
            return self._reset_mcp()
        if self._task_mode == "search":
            return self._reset_search()
        return self._reset_swe()

    def _reset_search(self) -> tuple[str, dict]:
        """Reset for search-mode tasks (no Docker)."""
        self.total_steps = 0
        self._search_answer = ""
        self._search_reward_debug = {}

        question = self.entry.get("question") or self.entry.get("query") or self.entry.get("input") or self.entry.get("problem_statement", "")
        # Strip stale answer-format instructions that conflict with FUSED_SEARCH_USER_PROMPT
        question = re.sub(r"\s*When ready, output the final answer enclosed in <answer> and </answer> tags\. Do not generate any content after the </answer> tag\.?", "", question).strip()
        return question, {"task_type": "search"}

    def _reset_swe(self) -> tuple[str, dict]:
        """Reset for SWE-mode tasks (Docker container)."""
        obs, info = super().reset()
        info["task_type"] = "swe"
        return obs, info

    def _reset_mcp(self) -> tuple[str, dict]:
        """Reset for MCP-mode tasks (tool-based tasks via MCP server)."""
        self.total_steps = 0
        self._mcp_answer = ""
        self._mcp_reward_debug = {}
        self._mcp_has_used_tools = False

        # Resolve tools_py path
        tools_py = self.entry.get("tools_py", "")
        data_root = self.entry.get("data_root", "")
        if data_root and tools_py and not os.path.isabs(tools_py):
            tools_py_abs = os.path.join(data_root, os.path.basename(tools_py))
        else:
            tools_py_abs = tools_py

        # If tools_py_abs doesn't exist, try the original path directly
        if not os.path.exists(tools_py_abs):
            tools_py_abs = tools_py

        # Start MCP server and discover tools
        if MCPConnectionManager is not None and MCPEnvironment is not None:
            try:
                from pathlib import Path
                tools_path = Path(tools_py_abs)
                if tools_path.exists() and tools_path.is_file():
                    server_script = MCPEnvironment._ensure_server_script(tools_path.parent)
                    mcp_server_command = sys.executable
                    mcp_server_args = [str(server_script)]
                    self._mcp_connection_manager = MCPConnectionManager(
                        mcp_server_command, mcp_server_args
                    )
                    self._mcp_connection_manager.start()
                    # Extract tool schemas from discovered tools (deduplicate by name)
                    seen = set()
                    self._mcp_tool_schemas = []
                    for tool in self._mcp_connection_manager.tool_map.values():
                        name = getattr(tool, "name", None)
                        if not name or name in seen:
                            continue
                        seen.add(name)
                        self._mcp_tool_schemas.append(tool.json)
                else:
                    logger.error("tools_py not found: %s", tools_py_abs)
                    self._mcp_tool_schemas = []
            except Exception as e:
                logger.error("Failed to start MCP server for %s: %s", tools_py_abs, e)
                self._mcp_tool_schemas = []
        else:
            logger.warning("MCP dependencies not available — MCP task will have no tools")
            self._mcp_tool_schemas = []

        question = self.entry.get("question", self.entry.get("problem_statement", ""))
        return question, {
            "task_type": "mcp",
            "tools_json": self._mcp_tool_schemas,
            "difficulty": self.entry.get("difficulty", ""),
        }

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action):
        if self._task_mode == "mcp":
            return self._step_mcp(action)
        if self._task_mode == "search":
            return self._step_search(action)
        return self._step_swe(action)

    def _step_swe(self, action):
        """SWE-mode step: web_search goes to retrieval tool, everything else to Docker."""
        if SWEAction is None:
            return super().step(action)

        # Unwrap list[Action] → list[SWEAction] and process each
        action_objs = self._unwrap_actions(action)

        # Check if the first action is web_search; rest go to Docker
        if action_objs and action_objs[0].function_name == "web_search":
            return self._handle_web_search(action_objs[0])

        # For SWE tools, pass the first action as its XML string to the parent
        if action_objs:
            return super().step(action_objs[0].to_xml_string())
        return super().step(action)

    def _step_search(self, action):
        """Search-mode step: handle web_search + finish/submit locally, error on Docker tools."""
        action_objs = self._unwrap_actions(action)
        if not action_objs:
            self.total_steps += 1
            return "Error: could not parse any actions from model output.", 0.0, False, {}

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            if fn == "web_search":
                obs, reward, done, info = self._handle_web_search(action_obj)
                if done:
                    return obs, reward, done, info
                observations.append(obs)
                continue

            if fn in ("finish", "submit"):
                return self._handle_search_finish(action_obj)

            # Docker-only tools are not available in search mode
            self.total_steps += 1
            observations.append(
                f"Error: The tool '{fn}' is not available for search tasks. "
                "Use web_search to find information and finish to submit your answer."
            )

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, {}

    # ------------------------------------------------------------------
    # Action unwrap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_actions(action) -> "list[SWEAction]":
        """Convert any action format from the workflow into a list of SWEAction objects.

        Handles:
        - ``str`` (XML-encoded SWEAction)
        - ``SWEAction`` instance
        - ``Action`` dataclass (``action.action`` is the XML string)
        - ``list[Action | str]`` from CLIAgent.update_from_model()
        """
        from rllm.agents.agent import Action as AgentAction

        raw_items: list = []
        if isinstance(action, list):
            raw_items = action
        else:
            raw_items = [action]

        swe_actions: list[SWEAction] = []
        for item in raw_items:
            if isinstance(item, AgentAction):
                item = item.action  # unwrap the dataclass
            if isinstance(item, str):
                if SWEAction is None:
                    # r2egym not available — try parsing as JSON {"name": ..., "arguments": ...}
                    try:
                        d = json.loads(item)
                        if isinstance(d, dict) and "name" in d:
                            import types
                            obj = types.SimpleNamespace(
                                function_name=d["name"],
                                parameters=d.get("arguments", {}),
                            )
                            swe_actions.append(obj)
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    logger.warning("Failed to parse action string (no r2egym): %s", item[:120])
                else:
                    try:
                        swe_actions.append(SWEAction.from_string(item))
                    except Exception:
                        logger.warning("Failed to parse action string: %s", item[:120])
            elif SWEAction is not None and isinstance(item, SWEAction):
                swe_actions.append(item)
            else:
                logger.warning("Unknown action type in _unwrap_actions: %s", type(item))
        return swe_actions

    # ------------------------------------------------------------------
    # MCP step
    # ------------------------------------------------------------------

    def _step_mcp(self, action):
        """MCP-mode step: route tool calls to MCP server, handle finish locally.

        Supports receiving a ``list[Action]`` from CLIAgent (multiple parsed
        tool calls per model turn).  Non-finish tool calls are executed
        sequentially and their results concatenated; a finish call terminates.
        """
        action_objs = self._unwrap_actions(action)
        if not action_objs:
            self.total_steps += 1
            return "Error: could not parse any actions from model output.", 0.0, False, {}

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            # Handle finish/submit — terminates immediately
            if fn in ("finish", "submit"):
                return self._handle_mcp_finish(action_obj)

            # Handle submit_result_difficulty_xxx
            if fn.startswith("submit_result_difficulty_"):
                return self._handle_mcp_submit_result(action_obj)

            # Regular MCP tool call
            self._mcp_has_used_tools = True
            self.total_steps += 1

            params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
            # Restore original types lost during SWEAction string round-trip.
            # SWEAction stringifies all parameter values (int→"0", bool→"false",
            # list→'["a","b"]').  json.loads recovers the original types so the
            # MCP server receives correctly-typed arguments.
            restored_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        restored_params[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        restored_params[k] = v
                else:
                    restored_params[k] = v
            tool_call_id = str(uuid.uuid4())
            tool_calls = [{
                "id": tool_call_id,
                "function": {
                    "name": fn,
                    "arguments": json.dumps(restored_params, ensure_ascii=False),
                }
            }]

            if self._mcp_connection_manager is None:
                observations.append(f"Execution output of [{fn}]:\nError: MCP server not available.")
                continue

            try:
                tool_outputs = self._mcp_connection_manager.execute_tool_calls(tool_calls)
                output_str = tool_outputs.get(tool_call_id, "No output")
                observations.append(f"Execution output of [{fn}]:\n{output_str}")
            except Exception as e:
                logger.error("MCP tool execution failed for %s: %s", fn, e)
                observations.append(f"Execution output of [{fn}]:\nError: {str(e)}")

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, {}

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

    def _handle_mcp_finish(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle finish/submit tool call in MCP mode.

        Tries to preserve the submitted result as a proper Python object so
        that the verifier receives a dict/list instead of a stringified blob.
        """
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", params.get("response", ""))

        # Try to get a structured object out of the result
        parsed = result
        if isinstance(result, str) and result.strip():
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                parsed = result

        if isinstance(parsed, (dict, list)):
            self._mcp_answer = json.dumps(parsed, ensure_ascii=False)
        elif isinstance(parsed, str) and parsed.strip():
            self._mcp_answer = parsed
        else:
            self._mcp_answer = str(result) if result else ""
        return "Your answer has been submitted.", 0.0, True, {}

    def _handle_mcp_submit_result(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle submit_result_difficulty_xxx tool call in MCP mode."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", "")

        # Try to get a structured object out of the result
        parsed = result
        if isinstance(result, str) and result.strip():
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                parsed = result

        if isinstance(parsed, (dict, list)):
            self._mcp_answer = json.dumps(parsed, ensure_ascii=False)
        elif isinstance(parsed, str) and parsed.strip():
            self._mcp_answer = parsed
        else:
            self._mcp_answer = str(result) if result else ""

        # Also execute on the MCP server if available (for side effects)
        if self._mcp_connection_manager is not None:
            fn = action_obj.function_name
            # Restore types for MCP server execution (same as _step_mcp)
            restored_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        restored_params[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        restored_params[k] = v
                else:
                    restored_params[k] = v
            tool_call_id = str(uuid.uuid4())
            tool_calls = [{
                "id": tool_call_id,
                "function": {
                    "name": fn,
                    "arguments": json.dumps(restored_params, ensure_ascii=False),
                }
            }]
            try:
                self._mcp_connection_manager.execute_tool_calls(tool_calls)
            except Exception:
                pass

        return "Your answer has been submitted.", 0.0, True, {}

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def compute_final_reward(self):
        if self._task_mode == "mcp":
            return self._compute_mcp_reward()
        if self._task_mode == "search":
            return self._compute_search_reward()
        return super().compute_final_reward()

    def compute_final_reward_metadata(self) -> dict:
        if self._task_mode == "mcp":
            self._compute_mcp_reward()
            return self._mcp_reward_debug
        if self._task_mode == "search":
            self._compute_search_reward()
            return self._search_reward_debug
        return super().compute_final_reward_metadata()

    def _compute_search_reward(self) -> float:
        """Compute F1-based reward for search tasks."""
        from rllm.rewards.reward_types import RewardConfig, RewardInput
        from rllm.rewards.search_reward import RewardSearchFn

        ground_truth = self.entry.get("ground_truth") or self.entry.get("answer") or self.entry.get("gt_answer") or self.entry.get("ground_truth_answer", "")
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

    def _compute_mcp_reward(self) -> float:
        """Compute verifier-based reward for MCP tasks."""
        from rllm.rewards.reward_types import RewardOutput
        from rllm.rewards.verifier_reward import verifier_reward_fn

        task_info = {
            **self.entry,
            "tool_call_stats": {
                "submit_called": bool(self._mcp_answer),
                "non_submit_tool_calls": self.total_steps - (1 if self._mcp_answer else 0),
                "step_count": self.total_steps,
            },
        }
        answer = self._mcp_answer

        try:
            reward_output = verifier_reward_fn(task_info=task_info, action=answer)
        except Exception as e:
            logger.error("MCP reward computation failed: %s", e)
            reward_output = RewardOutput(reward=0.0, metadata={"verifier_error": str(e)})

        self._mcp_reward_debug = {
            "type": "mcp",
            "reward": float(reward_output.reward),
            "resolved": reward_output.reward >= 1.0,
            "reward_mode": "verifier",
            "reward_source": "verifier_reward_fn",
            "is_correct": reward_output.is_correct,
            "verifier_error": reward_output.metadata.get("error", ""),
            **reward_output.metadata,
        }
        self._reward_debug = self._mcp_reward_debug
        return float(reward_output.reward)

    @property
    def reward_debug(self) -> dict:
        if self._task_mode == "mcp":
            return self._mcp_reward_debug
        if self._task_mode == "search":
            return self._search_reward_debug
        return self._reward_debug

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self):
        """Clean up resources."""
        # Stop MCP connection manager if running
        if self._mcp_connection_manager is not None:
            try:
                self._mcp_connection_manager.stop()
            except Exception:
                pass
            self._mcp_connection_manager = None
        # Clean up retrieval tool
        if self._retrieval_tool is not None:
            try:
                self._retrieval_tool.client.close()
            except Exception:
                pass
            self._retrieval_tool = None
        # Clean up Docker (SWE mode only)
        if self._task_mode == "swe":
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
