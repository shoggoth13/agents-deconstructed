import json
from json import JSONDecodeError
from typing import List, Tuple, Union

from langchain.schema import (
    AgentAction,
    AgentFinish,
    BasePromptTemplate,
    OutputParserException,
)

from dataclasses import dataclass
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain.schema import (
    ChatGeneration,
    Generation,
    OutputParserException,
)
from langchain.schema.output_parser import BaseGenerationOutputParser

@dataclass
class FunctionsAgentAction(AgentAction):
    message_log: List[BaseMessage]

def _convert_agent_action_to_messages(
    agent_action: AgentAction, observation: str
) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentAction):
        return agent_action.message_log + [
            _create_function_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]


def _create_function_message(
    agent_action: AgentAction, observation: str
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def format_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Format intermediate steps.
    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations
    Returns:
        list of messages to send to the LLM for the next prediction
    """
    messages = []

    for intermediate_step in intermediate_steps:
        agent_action, observation = intermediate_step
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages

class FunctionsAgentOutputParser(BaseGenerationOutputParser[Union[AgentAction, AgentFinish]]):

    def parse_result(self, result: List[Generation]) -> Union[AgentAction, AgentFinish]:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message

        function_call = message.additional_kwargs.get("function_call", {})

        if function_call:
            function_name = function_call["name"]
            try:
                _tool_input = json.loads(function_call["arguments"])
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function_call} because "
                    f"the `arguments` is not valid JSON."
                )

            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = "responded: {content}\n" if message.content else "\n"

            return FunctionsAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
                message_log=[message],
            )

        return AgentFinish(return_values={"output": message.content}, log=message.content)



