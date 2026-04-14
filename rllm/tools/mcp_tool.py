import logging
import traceback

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    def __init__(self, session, tool_name, tool_description, tool_schema):
        self._tool_schema = tool_schema
        self.session = session

        super().__init__(name=tool_name, description=tool_description)

    @property
    def json(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self._tool_schema}}

    async def async_forward(self, **kwargs) -> ToolOutput:
        try:
            logger.debug(f"Calling MCP tool: {self.name} with args: {kwargs}")

            result = await self.session.call_tool(self.name, kwargs)
            if hasattr(result, "content"):
                if hasattr(result.content, "text"):
                    content_str = result.content.text
                elif isinstance(result.content, list):
                    # MCP may return multiple content blocks (e.g., one per chunk / item).
                    # Join all text parts to preserve structured outputs like large JSON arrays.
                    parts: list[str] = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            parts.append(getattr(item, "text"))
                        else:
                            parts.append(str(item))
                    content_str = "".join(parts)
                else:
                    content_str = str(result.content)
            else:
                content_str = str(result)

            logger.debug(f"MCP tool result: {content_str}")
            return ToolOutput(name=self.name or "mcp_tool", output=content_str)
        except Exception as e:
            logger.debug(f"Error executing MCP tool {self.name}: {str(e)}")
            traceback.print_exc()
            return ToolOutput(
                name=self.name or "mcp_tool",
                error=f"Error calling MCP tool: {e}",
            )
