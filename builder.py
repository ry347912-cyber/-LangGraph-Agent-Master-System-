"""
graph/builder.py — LangGraph StateGraph Assembly
Wires all agents, tools, and memory into a compiled graph.
"""
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.supervisor import supervisor_node, should_continue
from src.agents.researcher import researcher_node
from src.agents.reflector import reflector_node
from src.agents.executor import executor_node


def build_graph():
    """Assemble and compile the multi-agent LangGraph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to agents
    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "researcher": "researcher",
            "executor": "executor",
            "reflector": "reflector",
            "end": END
        }
    )

    # All agents return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("executor", "supervisor")
    graph.add_edge("reflector", "supervisor")

    return graph.compile()


# Singleton compiled graph
agent_graph = build_graph()
