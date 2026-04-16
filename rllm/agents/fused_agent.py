import json
import logging
import os

from rllm.agents.cli_agent import (
    CLIAgent,
    R2EGYM_TOOL_FILES,
    generate_tool_schemas,
    parse_r2egym_tool_docstring,
)
from rllm.agents.system_prompts import FUSED_SYSTEM_PROMPT, FUSED_SEARCH_USER_PROMPT, FUSED_MCP_USER_PROMPT
from rllm.parser.tool_parser import QwenToolParser

logger = logging.getLogger(__name__)

# Path to the web_search tool docstring file (used only for schema generation)
WEB_SEARCH_TOOL_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "environments",
    "fused",
    "web_search.py",
)


def _build_fused_tools_system_prompt(scaffold: str = "r2egym") -> str:
    """Build the full system prompt with r2egym tools + web_search tool.

    Returns:
        The fused system prompt string with all tool schemas.
    """
    schemas = generate_tool_schemas(scaffold)
    web_search_schema = parse_r2egym_tool_docstring(WEB_SEARCH_TOOL_FILE)
    schemas.append(web_search_schema)

    tool_parser = QwenToolParser()
    schemas_str = "\n".join(
        json.dumps(s, indent=0, ensure_ascii=False) for s in schemas
    )
    tools_prompt = tool_parser.get_tool_prompt(schemas_str)
    return FUSED_SYSTEM_PROMPT.strip() + "\n" + tools_prompt


def _build_fused_search_system_prompt(scaffold: str = "r2egym") -> str:
    """Build a search-only system prompt with only web_search + finish tools.

    Search tasks should NOT see SWE tools (file_editor, execute_bash, search)
    to prevent the model from wasting steps on forbidden tool calls.
    """
    # Only include web_search and finish tool schemas
    web_search_schema = parse_r2egym_tool_docstring(WEB_SEARCH_TOOL_FILE)
    # Build finish schema from the r2egym finish tool file
    finish_tool_files = [f for f in R2EGYM_TOOL_FILES if "finish" in os.path.basename(f)]
    schemas = []
    for f in finish_tool_files:
        schemas.append(parse_r2egym_tool_docstring(f))
    schemas.append(web_search_schema)

    tool_parser = QwenToolParser()
    schemas_str = "\n".join(
        json.dumps(s, indent=0, ensure_ascii=False) for s in schemas
    )
    tools_prompt = tool_parser.get_tool_prompt(schemas_str)
    return FUSED_SYSTEM_PROMPT.strip() + "\n" + tools_prompt


def _build_fused_mcp_system_prompt(tools_json: list[dict], scaffold: str = "r2egym") -> str:
    """Build MCP-mode system prompt with dynamically discovered tool schemas.

    Includes the MCP tools from the environment plus the standard finish tool.

    Args:
        tools_json: List of OpenAI-style function schemas from MCP tool discovery.
        scaffold: Which scaffold's finish tool to include.
    """
    # Always include the finish tool schema
    finish_tool_files = [f for f in R2EGYM_TOOL_FILES if "finish" in os.path.basename(f)]
    schemas = list(tools_json)  # Start with MCP tool schemas
    for f in finish_tool_files:
        try:
            schemas.append(parse_r2egym_tool_docstring(f))
        except Exception:
            pass

    tool_parser = QwenToolParser()
    schemas_str = "\n".join(
        json.dumps(s, indent=0, ensure_ascii=False) for s in schemas
    )
    tools_prompt = tool_parser.get_tool_prompt(schemas_str)
    return FUSED_SYSTEM_PROMPT.strip() + "\n" + tools_prompt


class FusedAgent(CLIAgent):
    """Fused Agent combining CLI/SWE tools with web search capability.

    Extends CLIAgent by adding a web_search tool to the system prompt.
    All other behavior (multi-tool-call, pre-submission validation,
    loop detection, etc.) is inherited from CLIAgent.

    Automatically detects search vs SWE tasks via the ``task_type`` key
    in the info dict returned by FusedEnv.reset() and uses the appropriate
    user prompt template.
    """

    # Valid tool sets per task type
    _VALID_TOOLS_FUSED_SWE = {"file_editor", "search", "execute_bash", "finish", "web_search"}
    _VALID_TOOLS_FUSED_SEARCH = {"web_search", "finish"}

    def __init__(self, scaffold: str = "r2egym"):
        # Call CLIAgent.__init__ — it sets up tool_parser, system_prompt, etc.
        super().__init__(scaffold=scaffold)
        # Expand parser's valid_tools to include web_search (SWE default)
        self.tool_parser.valid_tools = self._VALID_TOOLS_FUSED_SWE
        # Override the system prompt with the fused version that includes web_search
        self.system_prompt = _build_fused_tools_system_prompt(scaffold)
        # Pre-build the search-only system prompt (lightweight, reusable)
        self._search_system_prompt = _build_fused_search_system_prompt(scaffold)
        # Re-initialize messages with the new system prompt
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def update_from_env(self, observation, reward, done, info):
        """Update agent state from environment observation.

        On the first step, checks ``info["task_type"]`` to select the
        appropriate user prompt template and system prompt before delegating
        to CLIAgent.
        """
        if not self._trajectory.steps:
            # First step: pick user prompt template based on task type
            task_type = info.get("task_type", "swe")
            if task_type == "search":
                self.user_prompt_template = FUSED_SEARCH_USER_PROMPT
                # Swap system prompt to search-only (web_search + finish only)
                self.messages[0] = {
                    "role": "system",
                    "content": self._search_system_prompt,
                }
                # Restrict parser to search-only tools for name normalization
                self.tool_parser.valid_tools = self._VALID_TOOLS_FUSED_SEARCH
            elif task_type == "mcp":
                self.user_prompt_template = FUSED_MCP_USER_PROMPT
                # Build dynamic system prompt from MCP tool schemas
                tools_json = info.get("tools_json", [])
                mcp_system_prompt = _build_fused_mcp_system_prompt(tools_json)
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
        super().update_from_env(observation, reward, done, info)
