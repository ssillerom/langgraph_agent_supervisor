from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.tools import tavily_tool, python_repl_tool
from src.agents import supervisor_node, LLM, State
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from typing import Literal


# Initialize a research agent with Tavily search tool
# This agent is restricted from doing mathematical calculations
research_agent = create_react_agent(
    LLM, tools=[tavily_tool], prompt="You are a researcher. DO NOT do any math."
)


def research_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Research node that processes the current state using the research agent
    and routes the result back to the supervisor.
    
    Args:
        state: Current state of the conversation
    
    Returns:
        Command directing flow back to supervisor with research results
    """
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


# Initialize a code agent with Python REPL tool
# WARNING: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_react_agent(LLM, tools=[python_repl_tool])


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Code node that processes the current state using the code agent
    and routes the result back to the supervisor.
    
    Args:
        state: Current state of the conversation
    
    Returns:
        Command directing flow back to supervisor with code execution results
    """
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )

def build_graph():
    """
    Builds and returns the agent workflow graph with a supervisor orchestrating
    between research and code nodes.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create a new state graph
    builder = StateGraph(State)
    # Define the starting point to connect to the supervisor
    builder.add_edge(START, "supervisor")
    # Add the supervisor, researcher, and coder nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("investigador", research_node)
    builder.add_node("programador", code_node)
    # Compile the graph into an executable form
    graph = builder.compile()

    return graph