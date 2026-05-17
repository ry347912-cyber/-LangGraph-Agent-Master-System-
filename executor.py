"""
agents/executor.py — Task Execution Agent
Synthesizes context into final answers; handles code/data tasks.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import AgentState
from src.utils.llm_factory import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise task executor. Given the task and research context,
produce the final, complete answer. Be structured, clear, and actionable.
Format your response appropriately for the task type."""),
    ("human", "Task: {task}\n\nResearch Context:\n{context}\n\nPrevious Critique:\n{critique}")
])


def executor_node(state: AgentState) -> AgentState:
    """Generate final answer from accumulated context."""
    logger.info("Executor agent activated")

    llm = get_llm(temperature=0.2)
    chain = EXECUTOR_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "task": state["task"],
        "context": "\n\n".join(state.get("context", [])),
        "critique": state.get("critique", "No prior critique")
    })

    return {
        **state,
        "final_answer": answer,
        "iteration": state.get("iteration", 0) + 1,
        "current_agent": "reflector"
    }
