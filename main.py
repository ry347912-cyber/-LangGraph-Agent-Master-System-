"""
api/main.py — FastAPI Application with Streaming
Production-ready API with SSE streaming, health checks, and CORS.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from src.graph.builder import agent_graph
from src.graph.state import AgentState
from src.utils.logger import get_logger
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

app = FastAPI(
    title="LangGraph AI Agent API",
    description="Production multi-agent AI system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str
    stream: bool = False
    max_iterations: Optional[int] = 5


class TaskResponse(BaseModel):
    answer: str
    iterations: int
    critique: Optional[str]
    token_usage: dict


@app.get("/health")
async def health():
    return {"status": "ok", "service": "langgraph-agent"}


@app.post("/run", response_model=TaskResponse)
async def run_agent(req: TaskRequest):
    """Run multi-agent workflow synchronously."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=req.task)],
        "task": req.task,
        "current_agent": "supervisor",
        "context": [],
        "final_answer": None,
        "critique": None,
        "iteration": 0,
        "tool_results": [],
        "token_usage": {}
    }

    try:
        result = agent_graph.invoke(initial_state)
        return TaskResponse(
            answer=result.get("final_answer", "No answer generated"),
            iterations=result.get("iteration", 0),
            critique=result.get("critique"),
            token_usage=result.get("token_usage", {})
        )
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_agent(req: TaskRequest):
    """Stream agent output via Server-Sent Events."""
    async def event_generator():
        initial_state: AgentState = {
            "messages": [HumanMessage(content=req.task)],
            "task": req.task,
            "current_agent": "supervisor",
            "context": [],
            "final_answer": None,
            "critique": None,
            "iteration": 0,
            "tool_results": [],
            "token_usage": {}
        }

        async for event in agent_graph.astream_events(initial_state, version="v1"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            elif event["event"] == "on_chain_end":
                yield f"data: {json.dumps({'done': True})}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
