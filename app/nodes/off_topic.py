from __future__ import annotations

from langchain_core.messages import AIMessage

from app.state import ChatState


def off_topic_node(state: ChatState) -> ChatState:
    """Handle messages that are outside supported topics."""
    reply = AIMessage(
        content=(
            "I'm here to help with questions about our products and how we work. "
            "Feel free to ask about features, pricing, or to get in touch with the team—"
            "I can collect your contact details and connect you."
        )
    )
    return {"messages": [reply]}
