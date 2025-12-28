from __future__ import annotations

import re
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage

from app.integrations.discord import send_lead_to_discord
from app.state import ChatState, LeadStatus, new_lead


EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _get_last_human_message(state: ChatState) -> HumanMessage | None:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _validate_email(value: str) -> bool:
    return bool(EMAIL_RE.fullmatch(value.strip()))


async def lead_capture_node(state: ChatState) -> ChatState:
    """Guide the user through sharing contact details and send them to Discord."""
    last_human = _get_last_human_message(state)
    if last_human is None:
        reply = AIMessage(
            content="I'd be happy to connect you with our team. Could you start by sharing your name?"
        )
        return {
            "messages": [reply],
            "lead_status": cast(LeadStatus, "collecting"),
            "lead_step": "name",
        }

    user_profile = dict(state.get("user_profile") or {})
    lead_status = cast(LeadStatus, state.get("lead_status", "none"))
    lead_step = state.get("lead_step")
    text = last_human.content.strip()

    # Start the flow if needed.
    if lead_status == "none" or lead_step is None:
        reply = AIMessage(
            content="I'd be happy to connect you with our team. First, what's your name?"
        )
        return {
            "messages": [reply],
            "lead_status": cast(LeadStatus, "collecting"),
            "lead_step": "name",
        }

    if lead_step == "name":
        user_profile["name"] = text
        reply = AIMessage(
            content="Great, thanks. What's the best email address for our team to reach you?"
        )
        return {
            "messages": [reply],
            "user_profile": user_profile,
            "lead_status": cast(LeadStatus, "collecting"),
            "lead_step": "email",
        }

    if lead_step == "email":
        if not _validate_email(text):
            reply = AIMessage(
                content="That doesn't look like a valid email address. "
                "Please share it in the format name@example.com."
            )
            return {"messages": [reply]}

        user_profile["email"] = text
        reply = AIMessage(
            content="Thanks. What company or organization are you with?"
        )
        return {
            "messages": [reply],
            "user_profile": user_profile,
            "lead_status": cast(LeadStatus, "collecting"),
            "lead_step": "company",
        }

    if lead_step == "company":
        user_profile["company"] = text
        reply = AIMessage(
            content="Got it. Finally, is there anything specific you'd like to discuss "
            "or any context you'd like to share with our team?"
        )
        return {
            "messages": [reply],
            "user_profile": user_profile,
            "lead_status": cast(LeadStatus, "collecting"),
            "lead_step": "message",
        }

    if lead_step == "message":
        # Create and send lead to Discord.
        lead = new_lead(user_profile, message=text, source="webchat")
        success = await send_lead_to_discord(lead)

        if success:
            reply_text = (
                "Thank you for sharing your details. I've passed them along to our team, "
                "and someone will be in touch soon."
            )
            new_status: LeadStatus = "sent"
        else:
            reply_text = (
                "Thank you for sharing your details. I wasn't able to send them to our team "
                "automatically, but they have been recorded."
            )
            new_status = "failed"

        reply = AIMessage(content=reply_text)
        return {
            "messages": [reply],
            "user_profile": user_profile,
            "lead_status": new_status,
            "lead_step": "message",
            "lead": lead,
        }

    # Fallback: restart flow.
    reply = AIMessage(
        content="Let's start over. Could you share your name so I can connect you with our team?"
    )
    return {
        "messages": [reply],
        "lead_status": cast(LeadStatus, "collecting"),
        "lead_step": "name",
    }
