import inspect
import json
import logging
import os
import re
import sys
import threading
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

try:
    from rllm.environments.endless_terminals.et_env import ETEnv
except ImportError:
    ETEnv = None


def _is_et_entry(entry: dict) -> bool:
    """Heuristic detector for Endless Terminals rows.

    Two signals (either is sufficient):
      1. ``data_source == "endless_terminals"`` (stamped by convert_to_parquet.py).
      2. Schema match: docker_image + final_state_test + no repo_name/commit_hash
         (ET rows are self-contained CLI tasks, not SWE-Bench-style repo issues).
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("data_source") == "endless_terminals":
        return True
    if entry.get("docker_image") and entry.get("final_state_test") and entry.get("instruction") and not entry.get("repo_name") and not entry.get("commit_hash"):
        return True
    return False


class FusedEnv(CLIEnv):
    """Fused environment combining CLI/SWE Docker tools with external web search and MCP tools.

    Operates in four modes based on data type:

    **CLI mode** (entry has ``docker_image`` and SWE-Bench schema):
        Extends CLIEnv (which extends SWEEnv). Intercepts ``web_search`` tool
        calls and routes them to a ``LocalRetrievalTool`` running outside the Docker
        container, while all other tool calls are delegated to Docker via the parent.

    **ET mode** (entry has ``data_source="endless_terminals"`` or matches the
        ET schema — docker_image + final_state_test, no repo_name):
        Wraps a standalone ``ETEnv`` instance. ETEnv talks to the remote Docker
        daemon directly (no r2egym RepoEnv), runs the ET initial-state pytest,
        executes function-call XML actions, and returns a binary reward from
        ``/logs/verifier/reward.txt`` after ``tests/test.sh``.

    **Web search mode** (entry has ``data_source`` but no ``docker_image`` or ``tools_py``):
        No Docker container is created. Only ``web_search`` and ``finish``/``submit``
        tools are available. Reward is computed via F1-score against ``ground_truth``.

    **MCP mode** (entry has ``tools_py``):
        No Docker container is created. Connects to an MCP server that serves the
        tools defined in ``tools_py``. Reward is computed via verifier code.
    """

    # Shared retrieval tool singleton — avoids creating one httpx.Client per env instance
    _shared_retrieval_tool = None
    _retrieval_lock = threading.Lock()

    def __init__(
        self,
        retrieval_server_url: str | None = None,
        retrieval_max_results: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_server_url = retrieval_server_url or os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:65432")
        self.retrieval_max_results = retrieval_max_results

        # Detect task mode: ET (checked before CLI since ET rows also carry a
        # docker_image), CLI, MCP, or Web Search.
        self._task_mode = self._resolve_task_mode(self.entry)

        # Web search mode state
        self._search_answer = ""  # Agent's submitted answer for reward computation
        self._search_reward_debug = {}

        # MCP mode state
        self._mcp_connection_manager: "MCPConnectionManager | None" = None
        self._mcp_tool_schemas: list[dict] = []
        self._mcp_answer = ""
        self._mcp_reward_debug: dict = {}
        self._mcp_has_used_tools = False
        self._mcp_consecutive_unknown = 0
        self._mcp_unknown_total = 0
        self._mcp_distinct_tools: set[str] = set()

        # ET mode state — built lazily in _reset_et so failures during ETEnv
        # construction surface in reset() (which the engine wraps in retry
        # logic) rather than the constructor (which it does not).
        self._et_inner: "ETEnv | None" = None
        self._et_reward_debug: dict = {}
        self._et_consecutive_unknown = 0
        self._et_unknown_total = 0

    def _get_retrieval_tool(self):
        """Return the class-level shared retrieval tool (lazy-initialized, thread-safe)."""
        if FusedEnv._shared_retrieval_tool is None:
            with FusedEnv._retrieval_lock:
                if FusedEnv._shared_retrieval_tool is None:
                    if LocalRetrievalTool is None:
                        logger.warning("LocalRetrievalTool not available — web_search will return errors")
                        return None
                    FusedEnv._shared_retrieval_tool = LocalRetrievalTool(
                        server_url=self.retrieval_server_url,
                        max_results=self.retrieval_max_results,
                    )
        return FusedEnv._shared_retrieval_tool

    @property
    def supports_parallel_step(self) -> bool:
        return self._task_mode in ("web search", "mcp")

    def _resolve_task_mode(self, entry: dict | None) -> str:
        entry = entry or {}
        if _is_et_entry(entry):
            return "et"
        if entry.get("docker_image"):
            return "cli"
        if entry.get("tools_py"):
            return "mcp"
        return "web search"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task: dict | str | None = None) -> tuple[str, dict]:
        next_task = self._normalize_entry(task)
        if next_task is not None and next_task != self.entry:
            self.close()
            self.env = None
        self._bind_task(next_task)
        self._task_mode = self._resolve_task_mode(self.entry)

        if self._task_mode == "et":
            return self._reset_et()
        if self._task_mode == "mcp":
            return self._reset_mcp()
        if self._task_mode == "web search":
            return self._reset_search()
        return self._reset_swe()

    def _reset_et(self) -> tuple[str, dict]:
        """Reset for ET-mode tasks (delegates to ETEnv)."""
        if ETEnv is None:
            raise RuntimeError("ETEnv import failed; cannot run endless_terminals tasks. " "Check that rllm.environments.endless_terminals.et_env is importable.")
        if self._et_inner is None:
            self._et_inner = ETEnv.from_dict(self.entry)
        obs, info = self._et_inner.reset()
        info["task_type"] = "et"
        return obs, info

    def _reset_search(self) -> tuple[str, dict]:
        """Reset for web-search-mode tasks (no Docker)."""
        self.total_steps = 0
        self._search_answer = ""
        self._search_reward_debug = {}
        # Fix #2/#5: per-rollout parser-health + bypass counters.
        self._search_web_search_calls = 0
        self._search_consecutive_unknown = 0
        self._search_unknown_total = 0
        # P0-2: track low-content (junk) retrieval responses so the
        # reward can punish "finish after 2 searches that returned
        # nothing useful" — the dominant failure mode for
        # simpleqa/hotpotqa/medqa at step 0.
        self._search_low_content_responses = 0
        self._search_retrieval_seen_docs = set()
        self._search_retrieval_duplicate_hits = 0

        question = self.entry.get("question") or self.entry.get("query") or self.entry.get("input") or self.entry.get("problem_statement", "")
        # Strip stale answer-format instructions that conflict with FUSED_SEARCH_USER_PROMPT
        question = re.sub(r"\s*When ready, output the final answer enclosed in <answer> and </answer> tags\. Do not generate any content after the </answer> tag\.?", "", question).strip()
        return question, {"task_type": "web search"}

    def _reset_swe(self) -> tuple[str, dict]:
        """Reset for CLI-mode tasks (Docker container)."""
        obs, info = super().reset()
        info["task_type"] = "cli"
        return obs, info

    def _reset_mcp(self) -> tuple[str, dict]:
        """Reset for MCP-mode tasks (tool-based tasks via MCP server)."""
        self.total_steps = 0
        self._mcp_answer = ""
        self._mcp_reward_debug = {}
        self._mcp_has_used_tools = False
        self._mcp_consecutive_unknown = 0
        self._mcp_unknown_total = 0
        self._mcp_distinct_tools: set[str] = set()

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
                    self._mcp_connection_manager = MCPConnectionManager(mcp_server_command, mcp_server_args)
                    self._mcp_connection_manager.start()
                    # Extract tool schemas from discovered tools (deduplicate by name)
                    seen = set()
                    self._mcp_tool_schemas = []
                    for tool in self._mcp_connection_manager.tool_map.values():
                        name = getattr(tool, "name", None)
                        if not name or name in seen:
                            continue
                        seen.add(name)
                        self._mcp_tool_schemas.append(self._slim_tool_schema(tool.json))
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

    _MAX_DESC_CHARS = 240
    _MAX_PARAM_DESC_CHARS = 120

    @classmethod
    def _slim_tool_schema(cls, schema: dict) -> dict:
        """Strip verbose sub-fields from a JSON-Schema tool definition.

        Fix #8: the system prompt reached ~13 k chars (close to the 13 k
        tool-call write limit) because each of the 1000+ per-task MCP
        tools carried redundant ``title`` fields on every property and
        multi-paragraph descriptions. We keep ``name``, ``description``,
        ``parameters`` / ``inputSchema`` with only ``type``,
        ``properties``, ``required``, ``items``; truncate descriptions
        to one sentence; drop ``title`` entirely (JSON Schema treats it
        as cosmetic).

        Handles both flat schemas and OpenAI-style wrapped schemas
        ``{"type": "function", "function": {...}}`` (what ``MCPTool.json``
        returns). The wrapped form is preserved and its inner body is
        slimmed; a bug in the flat-only version caused every MCP tool to
        collapse to ``{}`` and disappear from the prompt.
        """
        if not isinstance(schema, dict):
            return schema
        if isinstance(schema.get("function"), dict):
            inner = cls._slim_tool_schema(schema["function"])
            wrapped: dict = {"type": schema.get("type", "function"), "function": inner}
            return wrapped
        out: dict = {}
        if "name" in schema:
            out["name"] = schema["name"]
        desc = schema.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_DESC_CHARS]
        params_key = "parameters" if "parameters" in schema else ("inputSchema" if "inputSchema" in schema else None)
        if params_key:
            out[params_key] = cls._slim_json_schema(schema[params_key])
        return out

    @classmethod
    def _slim_json_schema(cls, node):
        if not isinstance(node, dict):
            return node
        out: dict = {}
        keep_keys = ("type", "properties", "required", "items", "enum", "oneOf", "anyOf")
        for k in keep_keys:
            if k in node:
                v = node[k]
                if k == "properties" and isinstance(v, dict):
                    out[k] = {p: cls._slim_property(pv) for p, pv in v.items()}
                elif k in ("oneOf", "anyOf") and isinstance(v, list):
                    out[k] = [cls._slim_json_schema(x) for x in v]
                elif k == "items":
                    out[k] = cls._slim_json_schema(v)
                else:
                    out[k] = v
        desc = node.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_PARAM_DESC_CHARS]
        return out

    @classmethod
    def _slim_property(cls, prop):
        if not isinstance(prop, dict):
            return prop
        out: dict = {}
        for k in ("type", "enum", "items", "properties", "required", "oneOf", "anyOf"):
            if k in prop:
                v = prop[k]
                if k == "items":
                    out[k] = cls._slim_json_schema(v)
                elif k == "properties" and isinstance(v, dict):
                    out[k] = {p: cls._slim_property(pv) for p, pv in v.items()}
                elif k in ("oneOf", "anyOf") and isinstance(v, list):
                    out[k] = [cls._slim_json_schema(x) for x in v]
                else:
                    out[k] = v
        desc = prop.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_PARAM_DESC_CHARS]
        return out

    # Fix #7: single-turn guards to prevent runaway generation. The eval
    # dump at step-10 contained one musique rollout whose assistant turn was
    # 488,222 characters of pure token repetition (e.g. ``"Jennifer"`` × 10k)
    # yet terminated normally with reward 0. With no truncation, no
    # repetition check, and no abnormal-termination flag, the signal was
    # invisible to training. The thresholds below are generous (32k chars,
    # 100 consecutive repeats of the same whitespace-separated token) and
    # intentionally conservative so normal long answers are never flagged.
    _MAX_TURN_CHARS = 32_000
    _MAX_CONSECUTIVE_TOKEN_REPEATS = 100

    @classmethod
    def _detect_runaway(cls, raw: str) -> tuple[bool, str]:
        """Return (is_runaway, reason) for pathological assistant output."""
        if not raw:
            return False, ""
        if len(raw) > cls._MAX_TURN_CHARS:
            return True, f"assistant turn exceeded {cls._MAX_TURN_CHARS} chars (got {len(raw)})"
        tokens = raw.split()
        if len(tokens) >= cls._MAX_CONSECUTIVE_TOKEN_REPEATS:
            run, prev = 1, None
            for tok in tokens:
                if tok == prev:
                    run += 1
                    if run >= cls._MAX_CONSECUTIVE_TOKEN_REPEATS:
                        return True, f"same token {tok!r} repeated {run}× consecutively"
                else:
                    run, prev = 1, tok
        return False, ""

    def step(self, action):
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first
        bad, reason = self._detect_runaway(raw_text)
        if bad:
            self.total_steps += 1
            info = {
                "termination_reason": "TRUNCATION",
                "termination_message": f"Runaway generation: {reason}",
                "guard/runaway_chars": len(raw_text),
            }
            return (
                f"Error: runaway generation detected ({reason}); terminating rollout.",
                0.0,
                True,
                info,
            )
        if self._task_mode == "mcp":
            return self._step_mcp(action)
        if self._task_mode == "web search":
            return self._step_search(action)
        if self._task_mode == "et":
            return self._step_et(action)
        return self._step_swe(action)

    def _step_swe(self, action):
        """CLI-mode step: web_search goes to retrieval tool, everything else to Docker."""
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

    def _step_et(self, action):
        """ET-mode step: route into the wrapped ETEnv, with structural-error
        bookkeeping that mirrors the CLI/MCP modes so the engine sees the
        same termination_reason taxonomy across data sources.
        """
        if self._et_inner is None:
            return (
                "Error: ET env not initialized; call reset() first.",
                0.0,
                True,
                {
                    "termination_reason": "ENV_INIT_ERROR",
                },
            )

        # Keep raw text so we can rescue \boxed{...} as an implicit submit and
        # report structural parse failures explicitly.
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first

        action_objs = self._unwrap_actions(action) if SWEAction is not None else []

        if not action_objs:
            if raw_text:
                try:
                    from rllm.parser.tool_parser import QwenToolParser as _QTP

                    tcs = _QTP().parse_qwen_tool_calls(raw_text)
                    if tcs and tcs[0].get("name") in ("finish", "submit"):
                        action_objs = [SWEAction(function_name="submit", parameters={})]
                except Exception:
                    pass
            if not action_objs:
                boxed = self._extract_boxed_from_raw(raw_text) if raw_text else None
                if boxed is not None:
                    action_objs = [SWEAction(function_name="submit", parameters={})]
                else:
                    self._et_consecutive_unknown += 1
                    self._et_unknown_total += 1
                    if self._et_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            "parser/consecutive_unknown": self._et_consecutive_unknown,
                            "parser/unknown_total": self._et_unknown_total,
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return "Error: could not parse any actions from model output.", 0.0, False, {}

        # ETEnv accepts one action at a time. Run them sequentially; the
        # first finish/submit terminates the rollout.
        observations: list[str] = []
        last_info: dict = {}
        for action_obj in action_objs:
            self._et_consecutive_unknown = 0
            obs, _r, done, info = self._et_inner.step(action_obj)
            observations.append(str(obs))
            last_info = info
            if done:
                combined = "\n".join(observations) if observations else str(obs)
                return combined, 0.0, True, info

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, last_info

    _MAX_CONSECUTIVE_UNKNOWN = 3

    @staticmethod
    def _extract_boxed_from_raw(raw: str) -> str | None:
        """Regex-rescue a ``\\boxed{…}`` payload from a raw assistant turn.

        Used when the tool-call parser cannot recover a tool name — in the
        eval dump the 64-step MAX_STEPS trajectory was exactly this loop:
        the model answered ``\\boxed{B}`` as free text and the env replied
        ``Error: The tool '' is not available`` 62 turns in a row.
        """
        if not raw:
            return None
        for tok in ("\\boxed{", "boxed{", "oxed{", "\x08oxed{"):
            i = raw.find(tok)
            if i < 0:
                continue
            i += len(tok)
            depth = 1
            j = i
            while depth and j < len(raw):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                return raw[i : j - 1]
        return None

    def _step_search(self, action):
        """Web-search-mode step: handle web_search + finish/submit locally, error on Docker tools."""
        if SWEAction is None:
            # Cannot parse actions without r2egym
            self.total_steps += 1
            return "Error: r2egym not available for action parsing.", 0.0, False, {}

        # Keep the raw model output so we can rescue \boxed{...} when the
        # tool-call parser returns nothing useful.
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first

        action_objs = self._unwrap_actions(action)
        if not action_objs:
            # Last-resort: try QwenToolParser directly on the raw string
            if raw_text:
                from rllm.parser.tool_parser import QwenToolParser as _QTP

                tcs = _QTP().parse_qwen_tool_calls(raw_text)
                if tcs and tcs[0].get("name") in ("finish", "submit"):
                    result = tcs[0].get("arguments", {}).get("result", "")
                    action_objs = [SWEAction(function_name="finish", parameters={"result": result})]
            if not action_objs:
                # Parser exhausted: try \boxed{...} as implicit finish.
                boxed = self._extract_boxed_from_raw(raw_text)
                if boxed is not None:
                    action_objs = [SWEAction(function_name="finish", parameters={"result": boxed})]
                else:
                    self.total_steps += 1
                    self._search_consecutive_unknown += 1
                    self._search_unknown_total += 1
                    if self._search_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            "parser/consecutive_unknown": self._search_consecutive_unknown,
                            "parser/unknown_total": self._search_unknown_total,
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return "Error: could not parse any actions from model output.", 0.0, False, {}

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            if fn == "web_search":
                self._search_consecutive_unknown = 0
                self._search_web_search_calls += 1
                obs, reward, done, info = self._handle_web_search(action_obj)
                if done:
                    return obs, reward, done, info
                observations.append(obs)
                continue

            if fn in ("finish", "submit"):
                self._search_consecutive_unknown = 0
                return self._handle_search_finish(action_obj)

            # Docker-only tools are not available in web search mode
            self.total_steps += 1
            self._search_consecutive_unknown += 1
            self._search_unknown_total += 1
            if self._search_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                info = {
                    "termination_reason": "ABNORMAL_PARSE_ERROR",
                    "parser/consecutive_unknown": self._search_consecutive_unknown,
                    "parser/unknown_total": self._search_unknown_total,
                }
                return (
                    f"Error: tool '{fn}' is not available; terminating after {self._search_consecutive_unknown} consecutive parse failures.",
                    0.0,
                    True,
                    info,
                )
            observations.append(f"Error: The tool '{fn}' is not available for web search tasks. " "Use web_search to find information and finish to submit your answer.")

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
        if SWEAction is None:
            self.total_steps += 1
            return "Error: r2egym not available for action parsing.", 0.0, False, {}

        # Keep raw text so we can rescue a `\boxed{…}` implicit finish and
        # report structural parse failures explicitly (fix #6).
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first

        action_objs = self._unwrap_actions(action)
        if not action_objs:
            # Last-resort: try QwenToolParser, then \boxed{...} as implicit finish.
            if raw_text:
                try:
                    from rllm.parser.tool_parser import QwenToolParser as _QTP

                    tcs = _QTP().parse_qwen_tool_calls(raw_text)
                    if tcs and tcs[0].get("name") in ("finish", "submit"):
                        result = tcs[0].get("arguments", {}).get("result", "")
                        action_objs = [SWEAction(function_name="finish", parameters={"result": result})]
                except Exception:
                    pass
            if not action_objs:
                boxed = self._extract_boxed_from_raw(raw_text)
                if boxed is not None:
                    action_objs = [SWEAction(function_name="finish", parameters={"result": boxed})]
                else:
                    self.total_steps += 1
                    self._mcp_consecutive_unknown = getattr(self, "_mcp_consecutive_unknown", 0) + 1
                    self._mcp_unknown_total = getattr(self, "_mcp_unknown_total", 0) + 1
                    if self._mcp_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            "termination_message": (f"MCP: {self._mcp_consecutive_unknown} consecutive turns " "without a parseable <tool_call>"),
                            "parser/consecutive_unknown": self._mcp_consecutive_unknown,
                            "parser/unknown_total": self._mcp_unknown_total,
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return (
                        "Error: could not parse any actions from model output. " 'Emit exactly one <tool_call>{"name": ..., "arguments": {...}}</tool_call> ' "block; use finish/submit to end the task.",
                        0.0,
                        False,
                        {"parser/unknown_total": self._mcp_unknown_total},
                    )

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            # Handle finish/submit — terminates immediately
            if fn in ("finish", "submit"):
                self._mcp_consecutive_unknown = 0
                return self._handle_mcp_finish(action_obj)

            # Handle submit_result_difficulty_xxx
            if fn.startswith("submit_result_difficulty_"):
                self._mcp_consecutive_unknown = 0
                return self._handle_mcp_submit_result(action_obj)

            # Regular MCP tool call
            # Empty function name means the parser found a <tool_call> block
            # but couldn't recover the ``name`` key — treat as structural
            # failure so the engine sees an explicit INVALID_REACT_STRUCTURE
            # bucket instead of silently consuming a step with a confusing
            # "Tool  not found" error.
            if not fn:
                self.total_steps += 1
                self._mcp_consecutive_unknown += 1
                self._mcp_unknown_total += 1
                if self._mcp_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                    info = {
                        "termination_reason": "INVALID_REACT_STRUCTURE",
                        "termination_message": (f"MCP: {self._mcp_consecutive_unknown} consecutive " "tool_calls with empty/unparseable `name` field"),
                        "parser/consecutive_unknown": self._mcp_consecutive_unknown,
                        "parser/unknown_total": self._mcp_unknown_total,
                    }
                    return (
                        "Error: could not parse a tool `name` from the <tool_call> " "block; terminating rollout.",
                        0.0,
                        True,
                        info,
                    )
                observations.append("Error: empty tool name. Each <tool_call> must be valid JSON with a " '"name" string (e.g. {"name": "finish", "arguments": {...}}).')
                continue

            self._mcp_consecutive_unknown = 0
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
            tool_calls = [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(restored_params, ensure_ascii=False),
                    },
                }
            ]

            if self._mcp_connection_manager is None:
                observations.append(f"Execution output of [{fn}]:\nError: MCP server not available.")
                continue

            try:
                tool_outputs = self._mcp_connection_manager.execute_tool_calls(tool_calls)
                output_str = tool_outputs.get(tool_call_id, "No output")
                observations.append(f"Execution output of [{fn}]:\n{output_str}")
                # Fix #3: track distinct successful tool names so the
                # verifier can reward genuine exploration (gated on
                # is_correct) rather than raw call count. A call is counted
                # as "successful" only if the output doesn't start with the
                # MCP server's "Error:" prefix.
                if not str(output_str).lstrip().lower().startswith("error"):
                    self._mcp_distinct_tools.add(fn)
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
            raw = result.to_string()
            # P2-8: dedup passages already seen in this rollout so the
            # model is forced to issue queries that surface new evidence.
            # simpleqa/gpqa at step-0 showed 864 / 138 consecutive
            # near-identical responses respectively.
            seen: set = getattr(self, "_search_retrieval_seen_docs", set())
            new_chunks: list[str] = []
            dup_count = 0
            for chunk in raw.split("\n\n"):
                sig = chunk.strip()[:400]
                if not sig:
                    continue
                h = hash(sig)
                if h in seen:
                    dup_count += 1
                    continue
                seen.add(h)
                new_chunks.append(chunk)
            if not new_chunks and raw.strip():
                # All chunks dedup'd away — surface a hint instead of
                # returning the identical passage a second time.
                body = "All returned passages were already surfaced by a previous " "search. Rephrase the query (add entities, dates, or " "constraints) to retrieve new evidence."
            else:
                body = "\n\n".join(new_chunks) if new_chunks else raw
            self._search_retrieval_seen_docs = seen
            self._search_retrieval_duplicate_hits = getattr(self, "_search_retrieval_duplicate_hits", 0) + dup_count
            # P0-2: count low-content responses (fewer than 20 words
            # after stripping the header) for reward shaping downstream.
            word_count = len(body.split())
            if word_count < 20:
                self._search_low_content_responses = getattr(self, "_search_low_content_responses", 0) + 1
            observation = f"Execution output of [web_search]:\n{body}"
        except Exception as e:
            logger.error("web_search execution failed: %s", str(e))
            observation = f"Execution output of [web_search]:\nError executing web_search: {str(e)}"

        return observation, 0.0, False, {}

    def _handle_search_finish(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle finish/submit tool call in web search mode."""
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
            tool_calls = [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(restored_params, ensure_ascii=False),
                    },
                }
            ]
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
        if self._task_mode == "web search":
            return self._compute_search_reward()
        if self._task_mode == "et":
            return self._compute_et_reward()
        return super().compute_final_reward()

    def compute_final_reward_metadata(self) -> dict:
        if self._task_mode == "mcp":
            self._compute_mcp_reward()
            return self._mcp_reward_debug
        if self._task_mode == "web search":
            self._compute_search_reward()
            return self._search_reward_debug
        if self._task_mode == "et":
            self._compute_et_reward()
            return self._et_reward_debug
        return super().compute_final_reward_metadata()

    def _compute_et_reward(self) -> float:
        """ET-mode reward: delegate to ETEnv.compute_final_reward_metadata,
        then surface the binary 1.0/0.0 as the rollout reward.
        """
        if self._et_inner is None:
            self._et_reward_debug = {
                "type": "endless_terminals",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "et_env_uninitialized",
                "verifier_error": "ETEnv was never reset; cannot run verifier.",
            }
            self._reward_debug = self._et_reward_debug
            return 0.0
        meta = {}
        try:
            meta = self._et_inner.compute_final_reward_metadata() or {}
        except Exception as exc:
            meta = {
                "type": "endless_terminals",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "et_env_exception",
                "verifier_error": f"{type(exc).__name__}: {exc}"[:512],
            }
        meta.setdefault("type", "endless_terminals")
        meta.setdefault("reward_mode", "binary")
        meta.setdefault("parser/unknown_total", self._et_unknown_total)
        reward = float(meta.get("reward", 0.0))
        meta["reward"] = reward
        meta.setdefault("resolved", reward >= 1.0)
        self._et_reward_debug = meta
        self._reward_debug = self._et_reward_debug
        return reward

    def _compute_search_reward(self) -> float:
        """Compute F1-based reward for web search tasks."""
        from rllm.rewards.reward_types import RewardConfig, RewardInput
        from rllm.rewards.search_reward import RewardSearchFn

        ground_truth = self.entry.get("ground_truth") or self.entry.get("answer") or self.entry.get("gt_answer") or self.entry.get("ground_truth_answer", "")
        answer = self._search_answer

        config = RewardConfig(
            toolcall_bonus=0.0,
            apply_repetition_penalty=True,
            repetition_penalty_weight=0.2,
            apply_length_penalty=True,
            length_penalty_weight=0.15,
            enable_step_bonus=False,
        )
        reward_fn = RewardSearchFn(config)
        question_text = self.entry.get("question") or self.entry.get("query") or self.entry.get("input") or self.entry.get("problem_statement", "")
        reward_input = RewardInput(
            task_info={
                "ground_truth": ground_truth,
                "step_count": self.total_steps,
                "question": question_text,
                "data_source": self.entry.get("data_source"),
            },
            action=answer,
        )
        reward_output = reward_fn(reward_input)

        ws_calls = getattr(self, "_search_web_search_calls", 0)
        low_content = getattr(self, "_search_low_content_responses", 0)
        dup_hits = getattr(self, "_search_retrieval_duplicate_hits", 0)

        # Penalize finish-without-search (bypass penalty).
        bypass_penalty = -0.5 if ws_calls == 0 else 0.0

        # P0-2: punish "early finish on junk" — if the rollout made <=2
        # searches AND the majority of them returned low-content
        # responses AND we got a wrong answer, the model is exploiting
        # the old "submit after 2 searches" shortcut. We apply a small
        # negative nudge on top of the existing 0 reward so the policy
        # has a clear signal to keep searching when evidence is thin.
        early_junk_penalty = 0.0
        if not reward_output.is_correct and ws_calls > 0 and ws_calls <= 2 and low_content >= ws_calls:
            early_junk_penalty = -0.1

        final_reward = max(0.0, min(1.0, float(reward_output.reward) + bypass_penalty + early_junk_penalty))

        self._search_reward_debug = {
            "type": "web search",
            "reward": final_reward,
            "resolved": final_reward >= 1.0,
            "reward_mode": "f1",
            "reward_source": "search_reward_fn",
            "is_correct": bool(reward_output.is_correct) if reward_output.is_correct is not None else False,
            "verifier_error": "",
            "reward/bypass_penalty": bypass_penalty,
            "reward/early_junk_penalty": early_junk_penalty,
            "reward/web_search_calls": ws_calls,
            "reward/low_content_responses": low_content,
            "reward/duplicate_passage_hits": dup_hits,
            "parser/unknown_total": getattr(self, "_search_unknown_total", 0),
            **reward_output.metadata,
        }
        self._reward_debug = self._search_reward_debug
        return final_reward

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
                "distinct_successful_tools": len(self._mcp_distinct_tools),
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
            "is_correct": bool(reward_output.is_correct) if reward_output.is_correct is not None else False,
            "verifier_error": reward_output.metadata.get("error", ""),
            **reward_output.metadata,
        }
        self._reward_debug = self._mcp_reward_debug
        return float(reward_output.reward)

    @property
    def reward_debug(self) -> dict:
        if self._task_mode == "mcp":
            return self._mcp_reward_debug
        if self._task_mode == "web search":
            return self._search_reward_debug
        if self._task_mode == "et":
            return self._et_reward_debug
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
        # Do NOT close _shared_retrieval_tool — it is shared across all FusedEnv instances
        # ET mode owns its own ETEnv instance — close it (best-effort, gated
        # internally by RLLM_ET_KEEP_CONTAINER for debugging).
        if self._task_mode == "et" and self._et_inner is not None:
            try:
                self._et_inner.close()
            except Exception:
                pass
            self._et_inner = None
        # Clean up Docker (CLI mode only)
        if self._task_mode == "cli":
            super().close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(extra_info: dict | str) -> "FusedEnv":
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        # Walk the MRO to collect all accepted __init__ params, since
        # FusedEnv.__init__ forwards **kwargs to parent classes.
        accepted = set()
        for cls in FusedEnv.__mro__:
            if cls is object:
                continue
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                accepted.add(name)

        init_params = {k: v for k, v in extra_info.items() if k in accepted}
        init_params["entry"] = extra_info
        return FusedEnv(**init_params)
