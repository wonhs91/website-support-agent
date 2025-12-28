from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.state import ChatState


def route(state: ChatState) -> str:
    """Simple intent router to pick QA vs. lead capture.

    This is rule-based for now for reliability and speed.
    As the product grows, this can be swapped for an LLM-based
    classifier without changing the graph topology.
    """
    # If we are in the middle of lead capture, stay there.
    if state.get("lead_status") == "collecting":
        return "lead_capture"

    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        return "qa"

    text = last_human.content.lower()
    lead_keywords = [
        "contact",
        "reach out",
        "sales",
        "talk to someone",
        "book a call",
        "demo",
        "speak with",
        "pricing",
        "quote",
    ]

    if any(keyword in text for keyword in lead_keywords):
        return "lead_capture"

    return "qa"
