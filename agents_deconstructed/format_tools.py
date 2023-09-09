"""Methods for formatting tools."""
from typing import Sequence
from langchain.tools import BaseTool


def format_tools_description(tools: Sequence[BaseTool]) -> str:
    """Format tools just with their description."""
    tool_string = ""
    for tool in tools:
        tool_string += f"{tool.name}: {tool.description}\n"
    return tool_string

def format_tools_args(tools: Sequence[BaseTool]) -> str:
    """Format tools with their description and args."""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.name}\n"
            f"Tool Description: {tool.description}\n"
            f"Tool Args: {tool.args}\n"
        )
        tool_descs.append(tool_desc)
    return "\n".join(tool_descs)
