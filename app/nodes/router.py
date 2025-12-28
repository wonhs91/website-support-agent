from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.state import ChatState

# Tool-constrained classifier to stabilize routing decisions.
router_llm = ChatOpenAI(model=settings.model_name, temperature=0)

ROUTER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "route_intent",
            "description": "Choose where to route the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["qa", "lead_capture", "off_topic"],
                        "description": (
                            "qa: product/company questions; "
                            "lead_capture: wants sales/demo/pricing/contact; "
                            "off_topic: unrelated or general chit-chat."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short justification for the chosen intent.",
                    },
                },
                "required": ["intent", "reason"],
            },
        },
    }
]

router_llm = router_llm.bind_tools(ROUTER_TOOL, tool_choice="route_intent")


def _get_last_human_message(state: ChatState) -> HumanMessage | None:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


async def route(state: ChatState) -> str:
    """LLM-based intent router with guardrails and off-topic handling."""
    lead_status = state.get("lead_status")

    # Hard guardrail: stay in lead capture while collecting details.
    if lead_status == "collecting":
        return "lead_capture"

    last_human = _get_last_human_message(state)
    if last_human is None:
        return "qa"

    system = SystemMessage(
        content=(
            "You are an intent classifier for a support chatbot. "
            "Select exactly one intent via the route_intent tool:\n"
            "- qa: answer product/company/support questions.\n"
            "- lead_capture: user wants to speak with sales, book a demo, pricing, quote, or share contact details.\n"
            "- off_topic: unrelated, chit-chat, or requests outside company/product scope.\n"
            "When unsure, choose qa. If lead_status is 'collecting', do not switch away from lead_capture. "
            f"Current lead_status: {lead_status}. Prior lead_step: {state.get('lead_step')}."
        )
    )

    try:
        result = router_llm.invoke([system] + (state.get("messages") or []))
    except Exception:
        return "qa"

    intent = "qa"
    tool_calls = getattr(result, "tool_calls", None) or []
    for call in tool_calls:
        if call.get("name", "") == "route_intent":
            args = call.get("args", {}) or {}
            candidate = args.get("intent")
            if candidate in ("qa", "lead_capture", "off_topic"):
                intent = candidate
            break

    # Final guardrail in case the model ignored constraints.
    if lead_status == "collecting":
        return "lead_capture"

    return intent
