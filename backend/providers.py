"""
Market data providers. Swap implementations by setting DATA_PROVIDER in config.

Every provider returns the same Bar/Snapshot shape so the scoring engine never
cares where the data came from. Drop your API key in .env and flip the provider
name. The MockProvider runs with zero credentials so you can develop the whole
pipeline before paying for a feed.
"""
from __future__ import annotations
import os
import random
import time
import math
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Snapshot:
    """A point-in-time view of one symbol. All providers normalize to this."""
    symbol: str
    sector: str
    last: float                 # latest trade price
    prev_close: float           # prior session close
    open_: float                # today's open
    volume: int                 # cumulative volume today
    avg_volume: float           # average full-day volume (lookback)
    vwap: float                 # volume-weighted average price today
    intraday: list[float] = field(default_factory=list)  # recent prices, oldest->newest
    has_catalyst: bool = False  # earnings today/tomorrow or fresh news
    catalyst_note: str = ""
    ts: float = field(default_factory=time.time)


class DataProvider(Protocol):
    def universe(self) -> list[str]: ...
    def snapshot(self, symbols: list[str]) -> list[Snapshot]: ...


# ---------------------------------------------------------------------------
# Mock provider: deterministic-ish synthetic market so you can run end to end.
# ---------------------------------------------------------------------------
_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "TSLA": "Consumer Cyclical", "AMZN": "Consumer Cyclical",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
    "PFE": "Healthcare", "MRNA": "Healthcare", "LLY": "Healthcare",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "KO": "Consumer Defensive", "PG": "Consumer Defensive",
    "NEM": "Materials", "FCX": "Materials", "NUE": "Materials",
    "PLD": "Real Estate", "AMT": "Real Estate",
    "DUK": "Utilities", "SO": "Utilities",
    "T": "Communication", "DIS": "Communication", "NFLX": "Communication",
}


class MockProvider:
    """Synthetic feed. Each symbol carries persistent state so momentum and
    volume evolve smoothly between polls instead of jumping randomly."""

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        for sym, sector in _SECTORS.items():
            base = random.uniform(20, 600)
            self._state[sym] = {
                "sector": sector,
                "prev_close": base,
                "open": base * random.uniform(0.985, 1.015),
                "last": base,
                "drift": random.uniform(-0.0008, 0.0012),  # per-tick bias
                "vol_mult": random.uniform(0.6, 2.5),       # rel volume tendency
                "volume": 0,
                "avg_volume": random.uniform(5e6, 8e7),
                "intraday": [],
                "cum_pv": 0.0,
                "cum_v": 0.0,
            }
        # seed a couple of names with catalysts and strong moves
        for sym in random.sample(list(self._state), 4):
            self._state[sym]["drift"] += random.uniform(0.002, 0.006)
            self._state[sym]["vol_mult"] *= random.uniform(2.0, 4.0)
            self._state[sym]["catalyst"] = random.choice(
                ["Earnings after close", "Upgraded by analyst", "M&A rumor", "FDA decision pending"]
            )

    def universe(self) -> list[str]:
        return list(self._state)

    def snapshot(self, symbols: list[str]) -> list[Snapshot]:
        out: list[Snapshot] = []
        for sym in symbols:
            s = self._state[sym]
            # random walk with drift
            shock = random.gauss(0, 0.004)
            s["last"] = max(0.5, s["last"] * (1 + s["drift"] + shock))
            tick_vol = int(s["avg_volume"] / 390 * s["vol_mult"] * random.uniform(0.5, 1.8))
            s["volume"] += tick_vol
            s["cum_pv"] += s["last"] * tick_vol
            s["cum_v"] += tick_vol
            s["intraday"].append(round(s["last"], 2))
            s["intraday"] = s["intraday"][-60:]
            vwap = s["cum_pv"] / s["cum_v"] if s["cum_v"] else s["last"]
            out.append(Snapshot(
                symbol=sym, sector=s["sector"], last=round(s["last"], 2),
                prev_close=round(s["prev_close"], 2), open_=round(s["open"], 2),
                volume=s["volume"], avg_volume=s["avg_volume"], vwap=round(vwap, 2),
                intraday=list(s["intraday"]),
                has_catalyst="catalyst" in s, catalyst_note=s.get("catalyst", ""),
            ))
        return out


# ---------------------------------------------------------------------------
# Polygon provider skeleton. Fill from env; falls back to mock if no key.
# Snapshot endpoint: /v2/snapshot/locale/us/markets/stocks/tickers
# ---------------------------------------------------------------------------
class PolygonProvider:
    def __init__(self) -> None:
        self.key = os.getenv("POLYGON_API_KEY", "")
        if not self.key:
            raise RuntimeError("POLYGON_API_KEY not set")
        import requests  # noqa
        self._requests = requests
        self.base = "https://api.polygon.io"

    def universe(self) -> list[str]:
        # Broad, cross-sector screened universe (thousands of names), built from
        # the most recent completed trading day's grouped-daily membership.
        # Falls back to the curated set if screening is unavailable.
        try:
            import datetime as dt
            from history import HistoryProvider
            from universe import screen_universe
            hist = HistoryProvider()
            day = dt.date.today()
            for _ in range(6):  # walk back over weekends/holidays
                day = day - dt.timedelta(days=1)
                if day.weekday() >= 5:
                    continue
                syms = screen_universe(day, hist)
                if syms:
                    return syms
        except Exception as e:  # noqa: BLE001
            print(f"[providers] universe screen failed ({e}); using curated set")
        return list(_SECTORS)

    def snapshot(self, symbols: list[str]) -> list[Snapshot]:
        url = f"{self.base}/v2/snapshot/locale/us/markets/stocks/tickers"
        # batch the tickers param — a few thousand symbols in one query string
        # exceeds URL length limits (414) and silently breaks the broad scan.
        data = []
        BATCH = 100
        for i in range(0, len(symbols), BATCH):
            chunk = symbols[i:i + BATCH]
            r = self._requests.get(url, params={"tickers": ",".join(chunk), "apiKey": self.key}, timeout=15)
            r.raise_for_status()
            data.extend(r.json().get("tickers", []) or [])
        out: list[Snapshot] = []
        for t in data:
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            last = t.get("lastTrade", {}).get("p") or day.get("c") or prev.get("c", 0)
            out.append(Snapshot(
                symbol=t["ticker"], sector=_SECTORS.get(t["ticker"], "Unknown"),
                last=last, prev_close=prev.get("c", last), open_=day.get("o", last),
                volume=int(day.get("v", 0)), avg_volume=prev.get("v", 1) or 1,
                vwap=day.get("vw", last), intraday=[],  # build via aggregates if needed
            ))
        return out


def get_provider(name: str | None = None) -> DataProvider:
    name = (name or os.getenv("DATA_PROVIDER", "mock")).lower()
    if name == "polygon":
        try:
            return PolygonProvider()
        except RuntimeError:
            print("[providers] POLYGON_API_KEY missing, falling back to mock")
            return MockProvider()
    return MockProvider()
