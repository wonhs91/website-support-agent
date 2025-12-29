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
    phone: str


class Lead(TypedDict):
    name: str
    email: str
    company: str
    phone: str
    message: str
    source: str
    created_at: str


class ChatState(TypedDict, total=False):
    """Global conversation state stored in LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    lead_status: LeadStatus
    lead_step: Literal[
        "intro",
        "gather",
        "collect_name",
        "confirm_name",
        "collect_email",
        "confirm_email",
        "collect_phone",
        "confirm_phone",
        "collect_company",
        "collect_message",
        "review",
        "send",
        "done",
    ]
    lead: Lead
    lead_attempts: dict[str, int]
    lead_message: str


def new_lead(
    profile: UserProfile,
    message: str,
    source: str = "webchat",
) -> Lead:
    return Lead(
        name=profile.get("name", ""),
        email=profile.get("email", ""),
        company=profile.get("company", ""),
        phone=profile.get("phone", ""),
        message=message,
        source=source,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
