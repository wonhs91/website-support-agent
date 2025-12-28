from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.knowledge import COMPANY_KNOWLEDGE
from app.state import ChatState


llm = ChatOpenAI(model=settings.model_name, temperature=0.2)


def qa_node(state: ChatState) -> ChatState:
    """Answer product/company questions using the static company knowledge."""
    messages = state.get("messages") or []

    system_content = (
        "You are a helpful AI assistant for a company. "
        "Use ONLY the following company and product information as the source of truth. "
        "If the answer is not in this information, say you are not sure.\n\n"
        f"{COMPANY_KNOWLEDGE}"
    )

    full_messages = [SystemMessage(content=system_content)] + messages
    response = llm.invoke(full_messages)

    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response.content))

    # LangGraph will append this to the existing message list.
    return {"messages": [response]}
