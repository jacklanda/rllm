import json
import logging
import os
import re
from typing import Any

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

try:
    import r2egym

    R2EGYM_PATH = os.path.dirname(r2egym.__file__)
except (ImportError, Exception):
    r2egym = None
    R2EGYM_PATH = ""

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import CLI_AGENT_SYSTEM_PROMPT, CLI_AGENT_USER_PROMPT
from rllm.parser.tool_parser import QwenToolParser

logger = logging.getLogger(__name__)

TOKEN_WARNING_THRESHOLD = 65536

# Tool file paths for each scaffold
R2EGYM_TOOL_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_TOOL_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]


def parse_r2egym_tool_docstring(tool_file_path: str) -> dict:
    """Parse a r2egym tool Python file's module docstring into an OpenAI-style function schema.

    Extracts the tool name from the filename, description from the 'Description:' line,
    and parameters from numbered parameter lines like '(1) name (type, required): desc'.

    Args:
        tool_file_path: Path to the r2egym tool Python file.

    Returns:
        Dict in OpenAI function-calling schema format.
    """
    tool_name = os.path.splitext(os.path.basename(tool_file_path))[0]

    try:
        with open(tool_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        logger.warning(f"Tool file not found: {tool_file_path}")
        return {"type": "function", "function": {"name": tool_name, "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}

    # Extract module docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if not docstring_match:
        docstring_match = re.search(r"'''(.*?)'''", content, re.DOTALL)
    if not docstring_match:
        return {"type": "function", "function": {"name": tool_name, "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}

    docstring = docstring_match.group(1)

    # Extract description: everything from 'Description:' to 'Parameters:' or 'Notes'
    desc_match = re.search(r"Description:\s*(.*?)(?=\n\s*(?:Parameters|Notes|\*\*Parameters))", docstring, re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()
        # Clean up: collapse newlines and leading asterisks into a single string
        description = re.sub(r"\n\s*\*\s*", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
    else:
        # Fallback: first non-empty line
        lines = [line.strip() for line in docstring.strip().split("\n") if line.strip()]
        description = lines[0] if lines else ""

    # Extract parameters
    # Pattern 1: (N) name (type, required/optional): description
    # Also handles: N. **name** (`type`, required): description
    param_pattern = re.compile(
        r"(?:\((\d+)\)|(\d+)\.)\s+\*{0,2}(\w+)\*{0,2}\s+"
        r"\(?[`]?(\w+)[`]?,\s*(required|optional)\)?\s*:\s*(.*?)(?=(?:\n\s*(?:\(\d+\)|\d+\.)|\Z))",
        re.DOTALL,
    )

    # Pattern 2: --name (type, required/optional): description
    alt_param_pattern = re.compile(
        r"--(\w+)\s+\((\w+),\s*(required|optional)\)\s*:\s*(.*?)(?=(?:\n\s*--|\Z))",
        re.DOTALL,
    )

    properties = {}
    required = []

    for match in param_pattern.finditer(docstring):
        param_name = match.group(3)
        param_type = match.group(4).lower()
        is_required = match.group(5).lower() == "required"
        param_desc = match.group(6).strip()
        # Clean up multiline descriptions
        param_desc = re.sub(r"\n\s*", " ", param_desc).strip()
        # Remove trailing content that belongs to next section
        param_desc = re.sub(r"\s*Allowed values:.*", "", param_desc).strip()

        # Map types
        type_map = {
            "string": "string",
            "integer": "integer",
            "boolean": "boolean",
            "array": "array",
            "number": "number",
        }
        json_type = type_map.get(param_type, "string")

        prop: dict[str, Any] = {"type": json_type, "description": param_desc}
        properties[param_name] = prop

        if is_required:
            required.append(param_name)

    # Fallback: try --name (type, required/optional) format if no params found
    if not properties:
        for match in alt_param_pattern.finditer(docstring):
            param_name = match.group(1)
            param_type = match.group(2).lower()
            is_required = match.group(3).lower() == "required"
            param_desc = match.group(4).strip()
            param_desc = re.sub(r"\n\s*", " ", param_desc).strip()

            type_map = {
                "string": "string",
                "integer": "integer",
                "boolean": "boolean",
                "array": "array",
                "number": "number",
            }
            json_type = type_map.get(param_type, "string")

            prop = {"type": json_type, "description": param_desc}
            properties[param_name] = prop

            if is_required:
                required.append(param_name)

    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    return schema


def generate_tool_schemas(scaffold: str = "r2egym") -> list[dict]:
    """Generate JSON tool schemas from r2egym tool files for the given scaffold.

    Args:
        scaffold: Either 'r2egym' or 'sweagent'.

    Returns:
        List of OpenAI-style function schemas.
    """
    tool_files = R2EGYM_TOOL_FILES if scaffold == "r2egym" else SWEAGENT_TOOL_FILES
    schemas = []
    for tool_file in tool_files:
        schema = parse_r2egym_tool_docstring(tool_file)
        schemas.append(schema)
    return schemas


def _build_tools_system_prompt(scaffold: str = "r2egym") -> str:
    """Build the full system prompt including tool schemas.

    Returns:
        The system prompt string with tool schemas formatted by QwenToolParser.
    """
    schemas = generate_tool_schemas(scaffold)
    tool_parser = QwenToolParser()
    schemas_str = "\n".join(json.dumps(s, indent=0, ensure_ascii=False) for s in schemas)
    tools_prompt = tool_parser.get_tool_prompt(schemas_str)
    return CLI_AGENT_SYSTEM_PROMPT.strip() + "\n" + tools_prompt


def _tool_call_to_swe_action(tool_call_dict: dict) -> "SWEAction":
    """Convert a parsed tool call dict to an r2egym Action object.

    Args:
        tool_call_dict: Dict with 'name' and 'arguments' keys.

    Returns:
        r2egym Action object.
    """
    function_name = tool_call_dict.get("name", "")
    arguments = tool_call_dict.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    # Ensure all parameter values are strings (r2egym Action expects string parameters)
    str_arguments = {}
    for k, v in arguments.items():
        if isinstance(v, list):
            str_arguments[k] = json.dumps(v)
        elif isinstance(v, bool):
            str_arguments[k] = str(v).lower()
        elif v is None:
            continue
        else:
            str_arguments[k] = str(v)
    return SWEAction(function_name=function_name, parameters=str_arguments)


class CLIAgent(BaseAgent):
    """SWE Agent using Qwen-style <tool_call>/<tool_response> format.

    This agent uses JSON tool schemas in the system prompt and communicates
    tool calls via <tool_call> tags and observations via <tool_response> tags,
    following the CLI agent loop definition.
    """

    def __init__(self, scaffold: str = "r2egym"):
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.scaffold = scaffold
        self.tool_parser = QwenToolParser()
        self.system_prompt = _build_tools_system_prompt(scaffold)
        self.user_prompt_template = CLI_AGENT_USER_PROMPT

        self._trajectory = Trajectory()
        # Pre-submission validation state
        self._has_run_tests = False  # Whether agent has executed any test command
        self._has_made_edits = False  # Whether agent has made any file edits
        self._submission_block_count = 0  # Number of times submission has been blocked
        self._max_submission_blocks = 3  # Cap to avoid infinite blocking loops
        # Repeated edit detection state
        self._last_failed_edit_key = None  # Serialized (path, old_str) of last failed str_replace
        self._failed_edit_repeat_count = 0
        self.reset()

    def update_from_env(self, observation, reward, done, info):
        """Update agent state from environment observation.

        On the first step (no prior trajectory steps), wraps the observation
        in the user prompt template. On subsequent steps, wraps in
        <tool_response> tags.
        """
        if self._trajectory.steps:
            # Subsequent steps: wrap observation in <tool_response> tags
            observation = str(observation)
            # Track test execution from observation content.
            # Only match pytest-specific markers that confirm a test suite actually ran,
            # NOT generic words like "passed"/"failed"/"test_" which appear in normal output.
            obs_lower = observation.lower()
            if "test session starts" in obs_lower or re.search(r"\d+ passed", obs_lower):
                self._has_run_tests = True

            # Track repeated str_replace failures to break edit loops early.
            # If the observation indicates a str_replace failure, record the key;
            # if the agent retries the exact same edit, we'll intercept in update_from_model.
            if any(marker in observation for marker in (
                "No occurrences of", "Multiple occurrences of",
                "No replacement was performed", "did not appear verbatim",
            )):
                # This was a failed edit — _last_failed_edit_key was set in update_from_model
                pass
            else:
                # Edit succeeded or this wasn't an edit — reset tracking
                self._last_failed_edit_key = None
                self._failed_edit_repeat_count = 0
        else:
            # First step: format as the initial user message with problem statement
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)

        # Add step budget / token budget warnings
        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        # Update the prior step with environment feedback
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        # Compose user message
        if self._trajectory.steps:
            # Wrap in <tool_response> tags for non-first turns
            user_content = f"{self.tool_parser.tool_output_begin}\n{observation}\n{self.tool_parser.tool_output_end}"
        else:
            # First turn: raw user prompt
            user_content = observation

        self.messages.append({"role": "user", "content": user_content})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs) -> "list[Action]":
        """Update agent state from model response.

        Parses the response for <tool_call> tags, converts each to an r2egym
        Action, and returns all of them for the engine to execute sequentially.
        """
        self._trajectory.steps.append(self.cur_step)

        # Parse tool calls from response
        tool_calls = self.tool_parser.parse(response)

        actions = []
        if tool_calls:
            for tc in tool_calls:
                action_dict = {"name": tc.name, "arguments": tc.arguments}

                # Track edits and test execution from tool calls
                if tc.name in ("file_editor", "str_replace_editor"):
                    cmd = tc.arguments.get("command", "")
                    if cmd in ("str_replace", "create", "insert"):
                        self._has_made_edits = True
                elif tc.name in ("execute_bash",):
                    cmd_str = str(tc.arguments.get("cmd", "") or tc.arguments.get("command", ""))
                    test_cmd_patterns = [
                        r"\bpytest\b",
                        r"\bpython\s+-m\s+pytest\b",
                        r"\bpython\s+-m\s+unittest\b",
                        r"\bruntests\b",
                        r"\bpy\.test\b",
                    ]
                    if any(re.search(p, cmd_str) for p in test_cmd_patterns):
                        self._has_run_tests = True

                # Detect repeated failing str_replace and redirect to view the file.
                is_str_replace = (
                    tc.name in ("file_editor", "str_replace_editor")
                    and tc.arguments.get("command") == "str_replace"
                )
                if is_str_replace:
                    edit_key = (tc.arguments.get("path", ""), tc.arguments.get("old_str", ""))
                    if edit_key == self._last_failed_edit_key:
                        self._failed_edit_repeat_count += 1
                        if self._failed_edit_repeat_count >= 2:
                            path = tc.arguments.get("path", "")
                            swe_action = SWEAction(
                                function_name="execute_bash",
                                parameters={"cmd": (
                                    f"echo '[REPEATED EDIT BLOCKED] You have retried the same failing str_replace "
                                    f"{self._failed_edit_repeat_count + 1} times. Viewing the file instead:' && "
                                    f"cat -n {path}"
                                )},
                            )
                            actions.append(Action(action=swe_action.to_xml_string()))
                            continue  # Skip remaining processing for this tool call
                    else:
                        self._last_failed_edit_key = edit_key
                        self._failed_edit_repeat_count = 0

                # Pre-submission validation: block premature submission until tests run.
                is_submit = tc.name in ("finish", "submit")
                if is_submit and self._has_made_edits and not self._has_run_tests and self._submission_block_count < self._max_submission_blocks:
                    self._submission_block_count += 1
                    swe_action = SWEAction(
                        function_name="execute_bash",
                        parameters={"cmd": "echo '[SUBMISSION BLOCKED] You have made edits but have not run any tests. Please run the relevant test suite (e.g., python -m pytest <test_file> -x) to verify your fix before submitting.'"},
                    )
                    actions.append(Action(action=swe_action.to_xml_string()))
                    continue  # Skip remaining tool calls after blocked submission
                else:
                    swe_action = _tool_call_to_swe_action(action_dict)
                    actions.append(Action(action=swe_action.to_xml_string()))

        if not actions:
            # No tool call found - model is either finishing or malformed
            actions.append(Action(action=""))

        # Extract thought (everything before the first <tool_call>)
        tc_idx = response.find(self.tool_parser.tool_call_begin)
        if tc_idx >= 0:
            thought = response[:tc_idx].strip()
        else:
            thought = response.strip()

        # Update trajectory step (first action stored for trajectory logging)
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = actions[0].action
        cur_step.model_response = response

        # Append assistant message (raw response preserves <tool_call> tags)
        self.messages.append({"role": "assistant", "content": response})

        self.step += 1
        return actions

    def update_from_env_intermediate(self, observation, reward, done, info):
        """Append an intermediate tool response without creating a new Step.

        Called between tool calls within a single model turn. Wraps the
        observation in <tool_response> tags and appends to messages, but
        does NOT create a new trajectory Step or add budget warnings.
        """
        observation = str(observation)

        # Track test execution from observation content
        obs_lower = observation.lower()
        if "test session starts" in obs_lower or re.search(r"\d+ passed", obs_lower):
            self._has_run_tests = True

        # Track repeated str_replace failures
        if any(marker in observation for marker in (
            "No occurrences of", "Multiple occurrences of",
            "No replacement was performed", "did not appear verbatim",
        )):
            pass  # _last_failed_edit_key was set in update_from_model
        else:
            self._last_failed_edit_key = None
            self._failed_edit_repeat_count = 0

        # Wrap in <tool_response> tags and append as user message
        user_content = f"{self.tool_parser.tool_output_begin}\n{observation}\n{self.tool_parser.tool_output_end}"
        self.messages.append({"role": "user", "content": user_content})

    def get_current_state(self) -> Step | None:
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0
        self._has_run_tests = False
        self._has_made_edits = False
        self._submission_block_count = 0
        self._last_failed_edit_key = None
        self._failed_edit_repeat_count = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
