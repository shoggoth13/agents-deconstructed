"""Methods for formatting tools."""
from langchain.tools import BaseTool


def format_tools_description(tools: BaseTool) -> str:
    """Format tools just with their description."""
    tool_string = ""
    for tool in tools:
        tool_string += f"{tool.name}: {tool.description}\n"
    return tool_string