from __future__ import annotations

from typing import Mapping

import httpx

from app.config import settings


async def send_lead_to_discord(lead: Mapping[str, str]) -> bool:
    """Send a captured lead to the configured Discord webhook channel."""
    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        # Treat missing webhook as a soft failure.
        return False

    content_lines = [
        "**New webchat lead**",
        f"**Name:** {lead.get('name', '')}",
        f"**Email:** {lead.get('email', '')}",
        f"**Phone:** {lead.get('phone', '')}",
        f"**Company:** {lead.get('company', '')}",
        f"**Message:** {lead.get('message', '')}",
        f"**Source:** {lead.get('source', '')}",
        f"**Created at:** {lead.get('created_at', '')}",
    ]

    payload = {
        "content": "\n".join(content_lines),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(webhook_url, json=payload)
            return response.status_code // 100 == 2
        except Exception as e:
            print(e)
            return False
