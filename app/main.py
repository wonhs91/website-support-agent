from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.graph import app as graph_app
from app.state import ChatState


app = FastAPI(title="Product Support Chatbot")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    lead_status: str | None = None


# Simple in-memory session store for now.
SESSIONS: Dict[str, ChatState] = {}


def get_initial_state() -> ChatState:
    return {
        "messages": [],
        "user_profile": {},
        "lead_status": "none",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    state = SESSIONS.get(request.session_id) or get_initial_state()

    # Append the latest user message.
    state["messages"] = (state.get("messages") or []) + [
        HumanMessage(content=request.message)
    ]

    # Run one step of the LangGraph app.
    new_state: ChatState = await graph_app.ainvoke(state)

    # Persist merged state back to the session store.
    SESSIONS[request.session_id] = new_state

    messages = new_state.get("messages") or []
    # Find the last assistant message for this turn.
    reply_text = ""
    for msg in reversed(messages):
        if msg.type == "ai":
            reply_text = str(msg.content)
            break

    if not reply_text:
        raise HTTPException(status_code=500, detail="No reply generated.")

    lead_status = new_state.get("lead_status")
    return ChatResponse(reply=reply_text, lead_status=lead_status)
