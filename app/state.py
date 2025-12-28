from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages


LeadStatus = Literal["none", "collecting", "sent", "failed"]


class UserProfile(TypedDict, total=False):
    name: str
    email: str
    company: str


class Lead(TypedDict):
    name: str
    email: str
    company: str
    message: str
    source: str
    created_at: str


class ChatState(TypedDict, total=False):
    """Global conversation state stored in LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    lead_status: LeadStatus
    lead_step: Literal["name", "email", "company", "message"]
    lead: Lead


def new_lead(
    profile: UserProfile,
    message: str,
    source: str = "webchat",
) -> Lead:
    return Lead(
        name=profile.get("name", ""),
        email=profile.get("email", ""),
        company=profile.get("company", ""),
        message=message,
        source=source,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
