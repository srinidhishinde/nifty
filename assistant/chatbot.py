from __future__ import annotations

from typing import Any

from config.settings import settings


SYSTEM_INSTRUCTIONS = """
You are the AI assistant inside an AI derivatives trading terminal.

Your job is to explain the dashboard, market-data status, strategy/risk logic,
backtests, ML/ensemble outputs, option-chain observations, and software behavior
clearly and conservatively.

Rules:
- Treat Kotak Neo as the production/live decision data source.
- Yahoo Finance is research/backtest/independent validation only.
- Never claim Yahoo is live data.
- Never invent market prices, option-chain values, OI, PCR, IV, news, fills,
  positions, or performance.
- If a live value is missing or stale, say so and explain that the system should WAIT.
- Explain why a trade was rejected when the dashboard context contains a gate/reason.
- ML is advisory; it cannot override data quality, deterministic strategy, EV,
  risk, or execution gates.
- You cannot place, modify, cancel, or recommend bypassing an order-safety gate.
- Do not expose API keys, TOTP, MPIN, access tokens, secrets, or credentials.
- For financial questions, distinguish educational explanation from a live trade
  decision. The dashboard's deterministic gates are authoritative.
"""


def _context_text(context: dict[str, Any] | None) -> str:
    if not context:
        return "No dashboard snapshot is currently available."
    safe = {}
    allowed = {
        "instrument", "environment", "data_source", "data_mode", "data_status",
        "data_timestamp", "quality_status", "quality_reasons", "rule_signal",
        "rule_confidence", "option_count", "pcr_oi", "pcr_volume", "regime",
        "rule_weight", "ml_weight", "ensemble_ce", "ensemble_pe",
        "stronger_side", "no_trade_reason", "historical_research_source",
    }
    for key in allowed:
        if key in context:
            safe[key] = context[key]
    return repr(safe)


def answer(user_message: str, history: list[dict[str, str]] | None = None,
           dashboard_context: dict[str, Any] | None = None) -> str:
    if not settings.openai_api_key:
        return (
            "ChatGPT assistant is not configured yet. Set OPENAI_API_KEY in .env "
            "and restart the dashboard."
        )

    try:
        from openai import OpenAI
    except ImportError:
        return "The OpenAI Python package is not installed. Add it to requirements.txt and reinstall dependencies."

    client = OpenAI(api_key=settings.openai_api_key)

    messages: list[dict[str, str]] = []
    for item in (history or [])[-12:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    prompt = (
        f"CURRENT DASHBOARD SNAPSHOT (treat as authoritative for current values): "
        f"{_context_text(dashboard_context)}\n\n"
        f"USER QUESTION:\n{user_message}"
    )
    messages.append({"role": "user", "content": prompt})

    response = client.responses.create(
        model=settings.openai_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=messages,
    )
    return response.output_text.strip()
