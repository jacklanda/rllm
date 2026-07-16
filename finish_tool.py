from __future__ import annotations
import os
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput
from parse_r2egym_tool_docstring import parse_r2egym_tool_docstring
"""
Description: A simple finish tool with a "submit" command.

Notes about the `submit` command:
* When invoked with `--result`, the provided string is used for submitting required task results (e.g., localization files).
* If no `--result` is provided, it defaults to an empty string.

**Parameters:**
  1. **command** (`string`, required): The command to run. Currently allowed option is: `submit`.
     - Allowed value: [`submit`]
  2. **result** (`string`, optional): The result text to submit. Defaults to an empty string.
"""

class FinishTool(Tool):
    DESCRIPTION = "Submit the final answer and finish the task."

    def __init__(self, name: str = "finish", description: str | None = None) -> None:
        super().__init__(name=name, description=description or self.DESCRIPTION)

    @property
    def json(self) -> dict[str, Any]:
        return parse_r2egym_tool_docstring("/home/qinshuhan/eval_space/workspace/evals/finish_tool.py", self.name)

    def forward(self, result: str = "", command: str = "submit", **_: Any) -> ToolOutput:
        del command
        return ToolOutput(name=self.name or "finish", output=result)
