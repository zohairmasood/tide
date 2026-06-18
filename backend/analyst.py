"""
Claude analyst note (FinRobot-inspired).

For a single symbol, write a short narrative that EXPLAINS the real,
pre-computed statistics and the retrieved catalyst context — and never invents,
restates differently, or "adjusts" any probability. The model is fed only
numbers we already computed (calibrated P(pop)/P(safe), their calibration
context, the empirical history record, top feature drivers) plus real news
headlines. A lightweight post-check flags any percentage in the note that
wasn't in the supplied stats.

Uses the Anthropic Python SDK with claude-opus-4-8 (default per the project's
claude-api guidance), adaptive thinking, medium effort, and structured output.
Runs only for the top 5, lazily, and is cached aggressively.
"""
from __future__ import annotations
import os
import re
import time

ANALYST_MODEL = os.getenv("ANALYST_MODEL", "claude-opus-4-8")
ANALYST_TTL = float(os.getenv("ANALYST_TTL", "900"))

_cache: dict[str, tuple[dict, float]] = {}
_client = None

_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "cited_stats": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["note", "cited_stats", "caveats"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a markets analyst writing a 2-4 sentence note for a momentum "
    "dashboard. You are given PRE-COMPUTED, calibrated statistics and retrieved "
    "news. Your job is to EXPLAIN what the numbers mean and how the catalyst "
    "context relates to them. Hard rules: never invent, restate with a different "
    "value, or 'adjust' any probability or statistic; cite only numbers that "
    "appear in the provided data; if the sample size is small or the model is "
    "flagged untrustworthy, say so plainly; never give buy/sell advice or price "
    "targets; never imply certainty. If there is no real catalyst, say the move "
    "is not backed by an identifiable catalyst. Put numeric claims you used in "
    "cited_stats, and risks/limitations in caveats."
)


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def _payload_numbers(payload: dict) -> set[float]:
    """All numeric values present in the supplied stats, as floats. A percentage
    in the note is considered grounded if its value matches one of these (most
    stats are already stored in percent, e.g. base_rate_pct=2.8, freq_target=21.0).
    A probability stored as a fraction (0.028) is also matched against its *100
    form so '2.8%' is recognized."""
    nums: set[float] = set()
    for m in re.findall(r"-?\d+(?:\.\d+)?", repr(payload)):
        try:
            v = float(m)
        except ValueError:
            continue
        nums.add(round(v, 2))
        if 0.0 < v < 1.0:  # fraction -> percent
            nums.add(round(v * 100, 2))
    return nums


def _fabricated_pcts(note: str, payload: dict) -> list[str]:
    """Return percentages in the note whose value isn't grounded in the payload
    (within a small tolerance), so we don't false-positive on rounding."""
    allowed = _payload_numbers(payload)
    out = []
    for tok in re.findall(r"\d+(?:\.\d+)?\s*%", note):
        val = float(tok.replace("%", "").strip())
        if not any(abs(val - a) <= 0.6 for a in allowed):
            out.append(tok.replace(" ", ""))
    return out


def generate_note(symbol: str, payload: dict) -> dict:
    """payload: {prediction, calibration, empirical, contributions, news}.
    Returns {available, note, cited_stats, caveats, flagged, news, generated_at}."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"available": False, "reason": "analyst notes disabled (no ANTHROPIC_API_KEY)"}

    now = time.time()
    cached = _cache.get(symbol)
    if cached and (now - cached[1] < ANALYST_TTL):
        out = dict(cached[0])
        out["cached"] = True
        return out

    import json as _json
    user_msg = (
        f"Symbol: {symbol}\n\n"
        f"Pre-computed statistics (do not alter):\n{_json.dumps(payload, indent=2)}\n\n"
        "Write the note now."
    )
    try:
        client = _get_client()
        resp = client.messages.create(
            model=ANALYST_MODEL,
            max_tokens=700,
            system=_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"analyst call failed: {type(e).__name__}: {e}"}

    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        data = _json.loads(text)
    except Exception:
        data = {"note": text, "cited_stats": [], "caveats": []}

    # honesty post-check: any % in the note must be grounded in the supplied stats
    fabricated = _fabricated_pcts(data.get("note", ""), payload)
    flagged = bool(fabricated)

    out = {
        "available": True,
        "symbol": symbol,
        "note": data.get("note", ""),
        "cited_stats": data.get("cited_stats", []),
        "caveats": data.get("caveats", []),
        "news": payload.get("news", []),
        "flagged": flagged,
        "fabricated_numbers": fabricated,
        "model_version": payload.get("calibration", {}).get("model_version"),
        "generated_at": now,
        "cached": False,
    }
    _cache[symbol] = (out, now)
    return out
