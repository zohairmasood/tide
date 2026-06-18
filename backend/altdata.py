"""
Alternative data — congressional "smart money" overlay.

Plain terms: US members of Congress must disclose their stock trades by law.
Some consistently beat the market because they sit on committees with early
information. This module surfaces recent disclosed trades for a ticker as RAW
FACTS (who, side, size, disclosure lag) — shown like the empirical panel, a
separate labeled side-signal. It is NEVER folded into the calibrated P(pop); at
most it breaks ties among already-gated picks.

Sources (pick whichever key is set; no brokerage / no citizenship requirement):
  - Quiver Quantitative congresstrading  (QUIVER_API_KEY)   — canonical dataset
  - Finnhub congressional-trading         (FINNHUB_API_KEY) — cheaper

Degrades gracefully: no key / no data -> {"available": False}. Disk-cached ~12h.
Caveats surfaced to the UI: disclosure lag is often weeks, and congressional-
tracking ETF evidence (NANC/KRUZ) is mixed — this is context, not an edge.
"""
from __future__ import annotations
import os
import json
import time
import datetime as dt
from pathlib import Path

CACHE_DIR = Path(os.getenv("ALTDATA_CACHE", "./.altdata_cache"))
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 60 * 60 * 12  # 12h


def _cache(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def _is_buy(s: str) -> bool:
    return "buy" in (s or "").lower() or "purchase" in (s or "").lower()


def _fetch_quiver(symbol: str) -> list[dict]:
    key = os.getenv("QUIVER_API_KEY", "")
    if not key:
        return []
    import requests
    url = f"https://api.quiverquant.com/beta/historical/congresstrading/{symbol.upper()}"
    r = requests.get(url, headers={"Authorization": f"Bearer {key}",
                                   "Accept": "application/json"}, timeout=12)
    r.raise_for_status()
    rows = r.json() or []
    out = []
    for x in rows:
        out.append({
            "name": x.get("Representative") or x.get("Name") or "",
            "side": "buy" if _is_buy(x.get("Transaction", "")) else "sell",
            "amount": x.get("Range") or x.get("Amount") or "",
            "transaction_date": x.get("TransactionDate") or x.get("Date") or "",
            "disclosed_date": x.get("ReportDate") or x.get("Disclosure") or "",
        })
    return out


def _fetch_finnhub(symbol: str, lookback_days: int) -> list[dict]:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return []
    import requests
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    r = requests.get("https://finnhub.io/api/v1/stock/congressional-trading",
                     params={"symbol": symbol.upper(), "from": start.isoformat(),
                             "to": end.isoformat(), "token": key}, timeout=12)
    r.raise_for_status()
    rows = (r.json() or {}).get("data", []) or []
    out = []
    for x in rows:
        out.append({
            "name": x.get("name", ""),
            "side": "buy" if _is_buy(x.get("transactionType", "")) else "sell",
            "amount": f"{x.get('amountFrom','')}-{x.get('amountTo','')}".strip("-"),
            "transaction_date": x.get("transactionDate", ""),
            "disclosed_date": x.get("filingDate", ""),
        })
    return out


def _lag_days(trade: dict) -> int | None:
    try:
        t = dt.date.fromisoformat(trade["transaction_date"][:10])
        d = dt.date.fromisoformat(trade["disclosed_date"][:10])
        return (d - t).days
    except Exception:  # noqa: BLE001
        return None


def congress_activity(symbol: str, lookback_days: int = 90) -> dict:
    """Recent disclosed congressional trades + a net-buy score for `symbol`."""
    p = _cache(symbol)
    if p.exists() and (time.time() - p.stat().st_mtime < CACHE_TTL):
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass

    try:
        trades = _fetch_quiver(symbol) or _fetch_finnhub(symbol, lookback_days)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"alt-data fetch failed: {type(e).__name__}"}

    if not os.getenv("QUIVER_API_KEY") and not os.getenv("FINNHUB_API_KEY"):
        return {"available": False, "reason": "no QUIVER_API_KEY / FINNHUB_API_KEY set"}

    # keep within lookback window
    cutoff = dt.date.today() - dt.timedelta(days=lookback_days)
    recent = []
    for t in trades:
        try:
            td = dt.date.fromisoformat((t.get("transaction_date") or "")[:10])
        except Exception:  # noqa: BLE001
            td = None
        if td is None or td >= cutoff:
            t["lag_days"] = _lag_days(t)
            recent.append(t)

    buys = sum(1 for t in recent if t["side"] == "buy")
    sells = sum(1 for t in recent if t["side"] == "sell")
    out = {
        "available": True,
        "symbol": symbol.upper(),
        "buys": buys,
        "sells": sells,
        "net": buys - sells,                 # tiebreak score (never alters probability)
        "lookback_days": lookback_days,
        "trades": recent[:12],
        "caveat": ("Disclosure lag is often weeks; congressional-tracking ETF "
                   "evidence (NANC/KRUZ) is mixed. Context, not an edge."),
    }
    try:
        p.write_text(json.dumps(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def enabled() -> bool:
    return bool(os.getenv("QUIVER_API_KEY") or os.getenv("FINNHUB_API_KEY"))
