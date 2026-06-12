import json
import logging
import os
import re

from rllm.agents.cli_agent import (
    CLIAgent,
    R2EGYM_TOOL_FILES,
    _is_qwen3_coder_model,
    generate_tool_schemas,
    make_qwen_tool_parser,
    parse_r2egym_tool_docstring,
)
from rllm.agents.system_prompts import (
    COT_SYSTEM_PROMPT,
    COT_USER_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT,
    FUSED_AGENT_SYSTEM_PROMPT,
    FUSED_ET_SYSTEM_PROMPT,
    FUSED_ET_USER_PROMPT,
    FUSED_MCP_SYSTEM_PROMPT,
    FUSED_MCP_USER_PROMPT,
    FUSED_SEARCH_SYSTEM_PROMPT,
    FUSED_SEARCH_USER_PROMPT,
    FUSED_UNIFIED_SYSTEM_PROMPT,
)
from rllm.types import Action

logger = logging.getLogger(__name__)

# Path to the web_search tool docstring file (used only for schema generation)
WEB_SEARCH_TOOL_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "environments",
    "fused",
    "web_search.py",
)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _format_system_prompt(base_prompt: str, schemas: list[dict], model_name: str | None = None) -> str:
    if _is_qwen3_coder_model(model_name):
        base_prompt = base_prompt.replace(
            '<tool_call>{"name": "finish", "arguments": {"command": "submit", "result": "[{\\"key\\": \\"val\\"}, {\\"key\\": \\"val2\\"}]"}}</tool_call>',
            """<tool_call>
<function=finish>
<parameter=command>
submit
</parameter>
<parameter=result>
[{"key": "val"}, {"key": "val2"}]
</parameter>
</function>
</tool_call>""",
        )
    tool_parser = make_qwen_tool_parser(model_name)
    schemas_str = "\n".join(json.dumps(s, indent=0, ensure_ascii=False) for s in schemas)
    tools_prompt = tool_parser.get_tool_prompt(schemas_str)
    return base_prompt.strip() + "\n" + tools_prompt


_PROMPT_ONLY_HARNESSES = {"cot", "bare"}
_VALID_HARNESSES = {"react", "unified_gem", "gem", *_PROMPT_ONLY_HARNESSES}


def _normalize_harness(harness: str | None = None) -> str:
    if harness is None:
        return "gem"

    normalized = str(harness).strip().lower().replace("-", "_")
    aliases = {
        "unified": "unified_gem",
        "fused_unified": "unified_gem",
        "specific_gem": "gem",
        "fused": "gem",
        "chain_of_thought": "cot",
        "no_system": "bare",
        "no_system_prompt": "bare",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_HARNESSES:
        raise ValueError(f"Invalid fused harness: {harness!r}. Expected one of {sorted(_VALID_HARNESSES)}")
    return normalized


def _base_prompt_for_harness(harness: str, gem_prompt: str) -> str:
    if harness == "cot":
        return COT_SYSTEM_PROMPT
    if harness == "react":
        return REACT_SYSTEM_PROMPT
    if harness == "unified_gem":
        return FUSED_UNIFIED_SYSTEM_PROMPT
    return gem_prompt


def _prompt_only_system_prompt(harness: str) -> str | None:
    if harness == "cot":
        return COT_SYSTEM_PROMPT
    if harness == "bare":
        return None
    return None


def _build_fused_tools_system_prompt(scaffold: str = "r2egym", harness: str = "gem", model_name: str | None = None) -> str:
    """Build the full system prompt with r2egym tools + web_search tool.

    Returns:
        The fused system prompt string with all tool schemas.
    """
    prompt_only_system_prompt = _prompt_only_system_prompt(harness)
    if harness in _PROMPT_ONLY_HARNESSES:
        return prompt_only_system_prompt or ""

    schemas = generate_tool_schemas(scaffold)
    web_search_schema = parse_r2egym_tool_docstring(WEB_SEARCH_TOOL_FILE)
    schemas.append(web_search_schema)

    base_prompt = _base_prompt_for_harness(harness, FUSED_AGENT_SYSTEM_PROMPT)
    return _format_system_prompt(base_prompt, schemas, model_name=model_name)


def _build_fused_search_system_prompt(scaffold: str = "r2egym", harness: str = "gem", model_name: str | None = None) -> str:
    """Build a search-only system prompt with only web_search + finish tools.

    Search tasks should NOT see SWE tools (file_editor, execute_bash, search)
    to prevent the model from wasting steps on forbidden tool calls.
    """
    prompt_only_system_prompt = _prompt_only_system_prompt(harness)
    if harness in _PROMPT_ONLY_HARNESSES:
        return prompt_only_system_prompt or ""

    # Only include web_search and finish tool schemas
    web_search_schema = parse_r2egym_tool_docstring(WEB_SEARCH_TOOL_FILE)
    # Build finish schema from the r2egym finish tool file
    finish_tool_files = [f for f in R2EGYM_TOOL_FILES if "finish" in os.path.basename(f)]
    schemas = []
    for f in finish_tool_files:
        schemas.append(parse_r2egym_tool_docstring(f))
    schemas.append(web_search_schema)

    base_prompt = _base_prompt_for_harness(harness, FUSED_SEARCH_SYSTEM_PROMPT)
    return _format_system_prompt(base_prompt, schemas, model_name=model_name)


def _build_fused_mcp_system_prompt(tools_json: list[dict], scaffold: str = "r2egym", harness: str = "gem", model_name: str | None = None) -> str:
    """Build MCP-mode system prompt with dynamically discovered tool schemas.

    Includes the MCP tools from the environment plus the standard finish tool.

    Args:
        tools_json: List of OpenAI-style function schemas from MCP tool discovery.
        scaffold: Which scaffold's finish tool to include.
    """
    prompt_only_system_prompt = _prompt_only_system_prompt(harness)
    if harness in _PROMPT_ONLY_HARNESSES:
        return prompt_only_system_prompt or ""

    # Always include the finish tool schema
    finish_tool_files = [f for f in R2EGYM_TOOL_FILES if "finish" in os.path.basename(f)]
    schemas = list(tools_json)  # Start with MCP tool schemas
    for f in finish_tool_files:
        try:
            schemas.append(parse_r2egym_tool_docstring(f))
        except Exception:
            pass

    base_prompt = _base_prompt_for_harness(harness, FUSED_MCP_SYSTEM_PROMPT)
    return _format_system_prompt(base_prompt, schemas, model_name=model_name)


def _build_fused_et_system_prompt(scaffold: str = "r2egym", harness: str = "gem", model_name: str | None = None) -> str:
    """Build the ET-mode system prompt: execute_bash + str_replace_editor + finish only.

    ET tasks are offline (no web_search) and have no repo-wide ripgrep helper
    (no `search` tool), so we whitelist the minimal toolset and use an
    ET-specific framing that drops the github-issue language.
    """
    prompt_only_system_prompt = _prompt_only_system_prompt(harness)
    if harness in _PROMPT_ONLY_HARNESSES:
        return prompt_only_system_prompt or ""

    # ET supports execute_bash, str_replace_editor (file_editor in r2egym
    # naming), and finish. No web_search, no `search` (rg) tool.
    keep = {"execute_bash", "file_editor", "str_replace_editor", "finish"}
    schemas = []
    for f in R2EGYM_TOOL_FILES:
        try:
            schema = parse_r2egym_tool_docstring(f)
            name = schema.get("function", schema).get("name") if isinstance(schema, dict) else None
            if name in keep:
                schemas.append(schema)
        except Exception:
            pass

    base_prompt = _base_prompt_for_harness(harness, FUSED_ET_SYSTEM_PROMPT)
    return _format_system_prompt(base_prompt, schemas, model_name=model_name)


class FusedAgent(CLIAgent):
    """Fused Agent combining CLI/SWE tools with web search capability.

    Extends CLIAgent by adding a web_search tool to the system prompt.
    All other behavior (multi-tool-call, pre-submission validation,
    loop detection, etc.) is inherited from CLIAgent.

    Automatically detects web search vs CLI tasks via the ``task_type`` key
    in the info dict returned by FusedEnv.reset() and uses the appropriate
    user prompt template.
    """

    # Valid tool sets per task type
    _VALID_TOOLS_FUSED_SWE = {"file_editor", "search", "execute_bash", "finish", "web_search"}
    _VALID_TOOLS_FUSED_SEARCH = {"web_search", "finish"}
    _VALID_TOOLS_FUSED_ET = {"execute_bash", "str_replace_editor", "file_editor", "finish", "submit"}

    def __init__(
        self,
        scaffold: str = "r2egym",
        harness: str | None = None,
        unified_system_prompt: bool | str | None = None,
        model_name: str | None = None,
    ):
        # Call CLIAgent.__init__ — it sets up tool_parser, system_prompt, etc.
        super().__init__(scaffold=scaffold, model_name=model_name)
        if harness is None and unified_system_prompt is not None:
            harness = "unified_gem" if _coerce_bool(unified_system_prompt) else "gem"
        self.harness = _normalize_harness(harness)
        # Expand parser's valid_tools to include web_search (SWE default)
        self.tool_parser.valid_tools = self._VALID_TOOLS_FUSED_SWE
        # Override the system prompt with the fused version that includes web_search
        self.system_prompt = _build_fused_tools_system_prompt(scaffold, self.harness, model_name=model_name)
        # Pre-build the search-only system prompt (lightweight, reusable)
        self._search_system_prompt = _build_fused_search_system_prompt(scaffold, self.harness, model_name=model_name)
        # Pre-build the ET system prompt (lightweight, reusable)
        self._et_system_prompt = _build_fused_et_system_prompt(scaffold, self.harness, model_name=model_name)
        # Re-initialize messages with the new system prompt. The bare harness
        # intentionally sends no system-role message.
        self.messages = [] if self.harness == "bare" else [{"role": "system", "content": self.system_prompt}]

    def update_from_env(self, observation, reward, done, info):
        """Update agent state from environment observation.

        On the first step, checks ``info["task_type"]`` to select the
        appropriate user prompt template and system prompt before delegating
        to CLIAgent.
        """
        if not self._trajectory.steps:
            # First step: pick user prompt template based on task type
            task_type = info.get("task_type", "cli")
            if self.harness in _PROMPT_ONLY_HARNESSES:
                self.user_prompt_template = COT_USER_PROMPT
                self.messages = [] if self.harness == "bare" else [{"role": "system", "content": COT_SYSTEM_PROMPT}]
                self.tool_parser.valid_tools = set()
            elif task_type == "web search":
                self.user_prompt_template = REACT_USER_PROMPT if self.harness == "react" else FUSED_SEARCH_USER_PROMPT
                # Swap system prompt to search-only (web_search + finish only)
                self.messages[0] = {
                    "role": "system",
                    "content": self._search_system_prompt,
                }
                # Restrict parser to search-only tools for name normalization
                self.tool_parser.valid_tools = self._VALID_TOOLS_FUSED_SEARCH
            elif task_type == "et":
                self.user_prompt_template = REACT_USER_PROMPT if self.harness == "react" else FUSED_ET_USER_PROMPT
                # Swap to ET system prompt (execute_bash + str_replace_editor +
                # finish only — no web_search, no rg-style `search`).
                self.messages[0] = {
                    "role": "system",
                    "content": self._et_system_prompt,
                }
                self.tool_parser.valid_tools = self._VALID_TOOLS_FUSED_ET
            elif task_type == "mcp":
                self.user_prompt_template = REACT_USER_PROMPT if self.harness == "react" else FUSED_MCP_USER_PROMPT
                # Build dynamic system prompt from MCP tool schemas
                tools_json = info.get("tools_json", [])
                mcp_system_prompt = _build_fused_mcp_system_prompt(
                    tools_json,
                    scaffold=self.scaffold,
                    harness=self.harness,
                    model_name=self.model_name,
                )
                self.messages[0] = {
                    "role": "system",
                    "content": mcp_system_prompt,
                }
                # Set valid tools dynamically from MCP schemas + finish
                mcp_tool_names = set()
                for schema in tools_json:
                    func_info = schema.get("function", schema)
                    name = func_info.get("name", "")
                    if name:
                        mcp_tool_names.add(name)
                        # Also add underscore variant (MCP tools often use hyphens)
                        mcp_tool_names.add(name.replace("-", "_"))
                mcp_tool_names.add("finish")
                mcp_tool_names.add("submit")
                self.tool_parser.valid_tools = mcp_tool_names
                # Append difficulty-based submit hint if available
                difficulty = info.get("difficulty", "")
                if difficulty:
                    submit_fn = f"submit_result_difficulty_{difficulty}"
                    mcp_tool_names.add(submit_fn)
            elif self.harness == "react":
                self.user_prompt_template = REACT_USER_PROMPT
        super().update_from_env(observation, reward, done, info)

    def update_from_model(self, response: str, **kwargs):
        actions = super().update_from_model(response, **kwargs)
        if self.harness not in _PROMPT_ONLY_HARNESSES:
            return actions

        has_tool_call_markup = bool(
            re.search(r"<\s*/?\s*(?:tool_call|function_call)\b|<function=", response or "")
        )
        cur_step = self._trajectory.steps[-1]
        if not has_tool_call_markup:
            cur_step.action = response
            cur_step.info = {
                **(cur_step.info or {}),
                "prompt_only_implicit_answer": True,
                f"{self.harness}_implicit_answer": True,
            }
            return [Action(action=response)]

        allowed_actions = []
        for action in actions:
            action_text = str(action.action or "")
            if re.search(r"(function_name=|<function=)(finish|submit)\b", action_text):
                allowed_actions.append(action)

        if allowed_actions:
            return allowed_actions

        cur_step.action = ""
        cur_step.info = {
            **(cur_step.info or {}),
            "suppressed_tool_action": True,
            f"{self.harness}_suppressed_tool_action": True,
        }
        return [Action(action="")]
