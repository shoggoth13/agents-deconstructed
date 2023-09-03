# 🤖🪟 Agents Deconstructed

Deconstructing agents to let you harness their power in a more understandable and customizable way.

## 🚀 Overview

[LangChain](https://github.com/langchain-ai/langchain) Agents (and agents in other frameworks) are powerful but tough to customize.
At the heart of those agents lies a `while` loop.
This package deconstructs agents by providing helper functions and then forcing the user to implement the `while` loop.
This makes agents both more understandable and more customizable.

## 📄 Installation
`pip install agents_deconstructed`

## 👨‍👩‍👧‍👦  Quickstart

We currently have two notebooks showing how use `agents_deconstructed`:

- [XML Agent](notebooks/xml.ipynb): An agent that uses XML formatting (tested with OpenAI and Anthropic Models.)
- [OpenAI Functions](notebooks/openai_functions.ipynb): An agent that uses OpenAI Functions

## 🔧 Components

`agents_deconstructed` provides a few types of helper functions:

**Tool Formatters**

Logic for converting tools into a format such that the language model can understand how to work with them.

**Output Parsers**

Logic for converting the output of an LLM into an `AgentAction` or `AgentFinish`.

**Intermediate Steps**

Logic for converting intermediate steps (tuples of `AgentAction` and `observation`) into a format such that the language model can understand how to work with them.

**Prompts**

The main instructions to the language model, this tells it out to think and how to output it's decisions.
This is pretty tightly coupled to a `Output Parser` and way of formatting `Intermediate Steps`

## 💻 Algorithm

With the above components, we can now easily create agent-like logic.
Pseudo-code for that logic:

```python
# Create agent as a prompt + LLM + output parser
# This uses LangChain Expression Language for ease of composability
agent = prompt | llm | output_parser
# Initialize empty list to track steps
steps = []
# Use this to group all traces together
with trace_as_chain_group("agent-run") as group_manager:
    # Get the first prediction
    inputs = {
        "input": ...,
        # Format intermediate steps as expected
        "intermediate_steps": format_steps(steps),
        # Format tools as expected
        "tools": format_tools(tools),
    }
    action = agent.invoke(inputs, config={"callbacks": group_manager})
    # Enter a loop until an AgentFinish is reached
    while not isinstance(action, AgentFinish):
        # Select the tool to use
        tool = ...
        # Run the tool
        observation = tool.run(action.tool_input, callbacks=group_manager)
        # Construct intermediate steps
        steps += [(action, observation)]
        # Do the next iteration
        inputs = {
            "input": ...,
            # Format intermediate steps as expected
            "intermediate_steps": format_steps(steps),
            # Format tools as expected
            "tools": format_tools(tools),
        }
        action = agent.invoke(inputs, config={"callbacks": group_manager})
```

## ⚖️ Comparison to LangChain Agents

Here is a (biased) comparison against LangChain agents:

**Pros**:

- Uses LangChain Expression Language (LangChain Agents haven't switched to use this)
- More customizable
- More understandable

**Cons**:

- Doesn't yet have same level of robustness (LangChain agents have safeguards against tools erroring, selecting tools that don't exist, etc)
- More work to set up

**The Same**:

- Logging to [LangSmith](https://smith.langchain.com/) (since we can use `trace_as_chain_group`)
