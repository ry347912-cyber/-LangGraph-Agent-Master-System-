"""
agents/supervisor.py — Supervisor / Orchestrator Agent
Routes tasks to specialized agents based on intent classification.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import AgentState
from src.utils.llm_factory import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI supervisor that routes tasks to specialized agents.

Available agents:
- researcher: For information retrieval, web search, RAG queries
- executor: For task execution, code writing, data processing  
- reflector: For critique, quality checking, self-improvement

Analyze the user task and respond with ONLY the agent name to route to.
If the task is complete, respond with: FINISH"""),
    ("human", "Task: {task}\nContext so far: {context}\nIteration: {iteration}")
])


def supervisor_node(state: AgentState) -> AgentState:
    """Route to the appropriate agent based on task analysis."""
    logger.info(f"Supervisor routing task: {state['task'][:80]}...")

    llm = get_llm(temperature=0)
    chain = SUPERVISOR_PROMPT | llm | StrOutputParser()

    decision = chain.invoke({
        "task": state["task"],
        "context": "\n".join(state.get("context", [])),
        "iteration": state.get("iteration", 0)
    })

    next_agent = decision.strip().lower()
    logger.info(f"Supervisor decision: {next_agent}")

    return {**state, "current_agent": next_agent}


def should_continue(state: AgentState) -> str:
    """Conditional edge: decide next node from supervisor output."""
    agent = state.get("current_agent", "").lower()
    iteration = state.get("iteration", 0)

    # Safety guard: max 5 iterations
    if iteration >= 5 or agent == "finish":
        return "end"

    if "research" in agent:
        return "researcher"
    elif "execut" in agent:
        return "executor"
    elif "reflect" in agent:
        return "reflector"
    else:
        return "end"
