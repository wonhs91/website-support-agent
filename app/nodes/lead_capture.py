from __future__ import annotations

from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.integrations.discord import send_lead_to_discord
from app.state import ChatState, LeadStatus, new_lead


LEAD_FLOW_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "lead_flow",
            "description": "Generate a natural reply and propose state updates for lead capture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {
                        "type": "string",
                        "description": "The assistant's natural, human reply to send.",
                    },
                    "intent": {
                        "type": "string",
                        "enum": ["gather", "review", "send", "other"],
                        "description": "Where to move the flow next.",
                    },
                    "fields": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                            "company": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                    "advance_to_review": {
                        "type": "boolean",
                        "description": "True if enough info to summarize/confirm.",
                    },
                    "advance_to_send": {
                        "type": "boolean",
                        "description": "True if the user confirmed to send.",
                    },
                },
                "required": ["reply", "intent"],
            },
        },
    }
]

reply_llm = ChatOpenAI(model=settings.model_name, temperature=0.6).bind_tools(
    LEAD_FLOW_TOOL, tool_choice="lead_flow"
    )


def _get_last_human_message(state: ChatState) -> HumanMessage | None:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _looks_like_email(value: str) -> bool:
    return "@" in value and "." in value and " " not in value


def _looks_like_phone(value: str) -> bool:
    digits = [c for c in value if c.isdigit()]
    return len(digits) >= 7


async def lead_capture_node(state: ChatState) -> ChatState:
    """Guide the user through sharing contact details with a natural, LLM-led flow."""
    last_human = _get_last_human_message(state)
    text = last_human.content.strip() if last_human else ""

    user_profile = dict(state.get("user_profile") or {})
    lead_status = cast(LeadStatus, state.get("lead_status", "none"))
    lead_step = state.get("lead_step") or "intro"
    lead_message = state.get("lead_message", "")
    messages = state.get("messages") or []

    if lead_status != "collecting":
        lead_status = cast(LeadStatus, "collecting")

    # Build context summary.
    known = {k: v for k, v in user_profile.items() if v}
    missing_fields: list[str] = []
    if not user_profile.get("email"):
        missing_fields.append("email")
    if not user_profile.get("name"):
        missing_fields.append("name")
    if not user_profile.get("phone"):
        missing_fields.append("phone")
    if not user_profile.get("company"):
        missing_fields.append("company")
    if not lead_message:
        missing_fields.append("message")

    stage = lead_step if lead_step in ("review", "send", "done") else "gather"

    # Recent conversation for continuity.
    recent_lines: list[str] = []
    for msg in (messages or [])[-8:]:
        role = "User" if msg.type == "human" else "Assistant"
        recent_lines.append(f"{role}: {msg.content}")
    recent_context = "\n".join(recent_lines) if recent_lines else "None"

    known_str = "; ".join(f"{k}: {v}" for k, v in known.items()) if known else "none yet"
    missing_str = ", ".join(missing_fields) if missing_fields else "none"

    system = SystemMessage(
        content=(
            "You are a warm, concise teammate in a webchat. Tone: brief (1-2 sentences), human, no bullets. "
            "Goals: (1) acknowledge/answer the user, (2) gently gather helpful contact details without sounding like a form. "
            "Soft priorities: name and email; phone/company/message are nice-to-have. "
            "If enough info is present, offer to share with the team and ask if they want tweaks; if they confirm, advance to send. "
            "If they hesitate, allow partial info. Return a tool call with your reply and state updates."
        )
    )

    prompt = (
        f"Known: {known_str}. Missing: {missing_str}. Stage: {stage}. "
        f"Recent conversation:\n{recent_context}\nUser said: {text!r}. Respond naturally in 'reply'. "
        "If the user provides or corrects details, place them in fields. "
        "If you have name+email or the user asks to send, set advance_to_review=true. "
        "If they clearly approve sending, set advance_to_send=true. "
        "Intent: choose gather while collecting, review to summarize/confirm, send when ready to dispatch."
        " If the user already said they only want to share some details, respect that and avoid re-asking unless they invite it."
    )

    try:
        result = reply_llm.invoke([system, HumanMessage(content=prompt)])
    except Exception:
        return {
            "messages": [
                AIMessage(content="Sorry, I hit a snag. Could you share the best email to reach you?")
            ],
            "user_profile": user_profile,
            "lead_status": lead_status,
            "lead_step": "gather",
            "lead_message": lead_message,
        }

    tool_calls = getattr(result, "tool_calls", None) or []
    reply_text = None
    intent = "gather"
    advance_to_review = False
    advance_to_send = False
    fields: dict[str, str] = {}
    for call in tool_calls:
        if call.get("name") == "lead_flow":
            args = call.get("args", {}) or {}
            reply_text = args.get("reply") or reply_text
            intent = args.get("intent", intent) or intent
            fields = args.get("fields") or {}
            advance_to_review = bool(args.get("advance_to_review", advance_to_review))
            advance_to_send = bool(args.get("advance_to_send", advance_to_send))
            break

    if fields:
        for key in ("name", "email", "phone", "company"):
            if key in fields and fields[key]:
                user_profile[key] = fields[key]
        if "message" in fields and fields["message"]:
            lead_message = fields["message"]

    if not reply_text:
        reply_text = "Thanks for sharing. Would you like me to pass this along to the team?"

    # Decide stage transitions.
    if advance_to_send:
        intent = "send"
    elif advance_to_review or (user_profile.get("name") and user_profile.get("email")):
        intent = "review"

    if intent == "send":
        lead = new_lead(user_profile, message=lead_message or "", source="webchat")
        # Minimal sanity check on email/phone; if clearly bad, fall back to review.
        email_ok = bool(user_profile.get("email") and _looks_like_email(user_profile["email"]))
        phone_ok = True
        if user_profile.get("phone"):
            phone_ok = _looks_like_phone(user_profile["phone"])
        if not email_ok:
            intent = "review"
        elif not phone_ok:
            intent = "review"
        else:
            success = await send_lead_to_discord(lead)
            if success:
                reply_text = "Thanks—I've shared your details with the team. They'll be in touch soon."
                new_status: LeadStatus = "sent"
            else:
                reply_text = "Thanks for the details. I couldn't auto-share them, but I've noted everything for the team."
                new_status = "failed"
            return {
                "messages": [AIMessage(content=reply_text)],
                "user_profile": user_profile,
                "lead_status": new_status,
                "lead_step": "done",
                "lead": lead,
                "lead_message": lead_message,
            }

    # Stay in gather/review with the new reply.
    next_step = "review" if intent == "review" else "gather"
    return {
        "messages": [AIMessage(content=reply_text)],
        "user_profile": user_profile,
        "lead_status": lead_status,
        "lead_step": next_step,
        "lead_message": lead_message,
    }
