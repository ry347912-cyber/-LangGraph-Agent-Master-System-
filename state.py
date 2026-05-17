"""
graph/state.py — Central AgentState for LangGraph
All agents read/write to this shared state.
"""
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # Conversation history — append-only
    messages: Annotated[List[BaseMessage], operator.add]

    # Current task being worked on
    task: str

    # Which agent is currently active
    current_agent: str

    # Accumulated research/context
    context: List[str]

    # Final synthesized answer
    final_answer: Optional[str]

    # Self-reflection feedback
    critique: Optional[str]

    # Number of reflection iterations
    iteration: int

    # Tool outputs collected
    tool_results: List[dict]

    # Token usage for cost tracking
    token_usage: dict
