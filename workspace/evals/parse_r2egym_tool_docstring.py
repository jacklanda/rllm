import re
import os
from typing import Any

def parse_r2egym_tool_docstring(tool_file_path: str, tool_name) -> dict:
    """Parse a r2egym tool Python file's module docstring into an OpenAI-style function schema.

    Extracts the tool name from the filename, description from the 'Description:' line,
    and parameters from numbered parameter lines like '(1) name (type, required): desc'.

    Args:
        tool_file_path: Path to the r2egym tool Python file.

    Returns:
        Dict in OpenAI function-calling schema format.
    """

    try:
        with open(tool_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
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
        r"(?:\((\d+)\)|(\d+)\.)\s+\*{0,2}(\w+)\*{0,2}\s+" r"\(?[`]?(\w+)[`]?,\s*(required|optional)\)?\s*:\s*(.*?)(?=(?:\n\s*(?:\(\d+\)|\d+\.)|\Z))",
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