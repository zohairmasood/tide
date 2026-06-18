"""
Universe screening — pick the broad, cross-sector, tradable set to scan.

Built PER HISTORICAL DATE from grouped-daily membership (the names that
actually traded that day), NOT from today's active-ticker list. This avoids
survivorship bias: if you screen using only currently-listed names and then
pull their history, delisted/blown-up names vanish and both the pop and safe
base rates come out optimistically wrong.

The screen keeps liquid, reasonably-priced common stock and drops penny names,
illiquid names, and non-CS instruments (ETFs/warrants/units/rights).
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path

from history import HistoryProvider, Bar

CACHE_DIR = Path(os.getenv("UNIVERSE_CACHE", "./.universe_cache"))
CACHE_DIR.mkdir(exist_ok=True)

# Screening thresholds (env-overridable)
MIN_PRICE = float(os.getenv("SCREEN_MIN_PRICE", "3.0"))
MAX_PRICE = float(os.getenv("SCREEN_MAX_PRICE", "2000.0"))
MIN_DOLLAR_VOLUME = float(os.getenv("SCREEN_MIN_DOLLAR_VOLUME", "5000000"))  # $5M/day


def _cs_symbols(hist: HistoryProvider) -> set[str]:
    rows = hist.reference_tickers(type_="CS")
    return {r["ticker"] for r in rows if r.get("ticker")}


def screen_universe(
    date,
    hist: HistoryProvider,
    *,
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    restrict_to_cs: bool = True,
    cache: bool = True,
) -> list[str]:
    """Return the screened ticker list for `date` (a datetime.date).

    Screens on price band and dollar volume from that day's grouped-daily bars,
    optionally intersected with the active common-stock set. Cached per date.
    """
    import datetime as dt
    if isinstance(date, dt.datetime):
        date = date.date()
    iso = date.isoformat()
    # UNIVERSE_LIMIT caps the scan to the most-liquid N names (0 = no cap). This
    # bounds the cost of the live broad scan, which scores every name each tick.
    limit = int(os.getenv("UNIVERSE_LIMIT", "200"))  # sane default; set 0 to uncap
    cache_path = CACHE_DIR / f"{iso}_{limit}.json"
    if cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    grouped: dict[str, Bar] = hist.grouped_daily(date)
    cs = _cs_symbols(hist) if restrict_to_cs else None

    scored: list[tuple[str, float]] = []
    for sym, bar in grouped.items():
        if cs is not None and sym not in cs:
            continue
        if not (min_price <= bar.c <= max_price):
            continue
        dv = bar.c * bar.v
        if dv < min_dollar_volume:
            continue
        scored.append((sym, dv))
    if limit > 0:
        scored.sort(key=lambda r: -r[1])      # most liquid first
        scored = scored[:limit]
    out = sorted(s for s, _ in scored)
    if cache and out:
        cache_path.write_text(json.dumps(out))
    return out
