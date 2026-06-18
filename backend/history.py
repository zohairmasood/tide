"""
Historical bar fetcher.

Wired for Polygon's aggregates endpoint. Returns a list of daily bars per
symbol, cached to disk so you don't re-pull (and re-bill) on every run.

The backtest layer (backtest.py) consumes these bars to compute empirical
forward-return frequencies. Without a key, raises clearly so you know exactly
what to set rather than silently producing fake history.

Polygon aggregates:
  GET /v2/aggs/ticker/{T}/range/1/day/{from}/{to}
Docs: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
"""
from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

CACHE_DIR = Path(os.getenv("HIST_CACHE", "./.hist_cache"))
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 60 * 60 * 12  # 12h; daily bars don't change intraday


@dataclass
class Bar:
    t: int       # ms epoch (session date)
    o: float
    h: float
    l: float
    c: float
    v: int
    vw: float    # volume-weighted avg price for the day


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_daily.json"


def _read_cache(symbol: str) -> list[Bar] | None:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > CACHE_TTL:
        return None
    try:
        raw = json.loads(p.read_text())
        return [Bar(**b) for b in raw]
    except Exception:
        return None


def _write_cache(symbol: str, bars: list[Bar]) -> None:
    _cache_path(symbol).write_text(json.dumps([asdict(b) for b in bars]))


class HistoryProvider:
    """Polygon-backed daily history. Swap the fetch method to use another feed;
    keep the Bar shape and the rest of the stack is unchanged."""

    def __init__(self) -> None:
        self.key = os.getenv("POLYGON_API_KEY", "")
        self.base = "https://api.polygon.io"
        self._requests = None

    _last_call = 0.0  # class-wide throttle clock (shared across instances)

    def _client(self):
        if self._requests is None:
            import requests
            self._requests = requests
        return self._requests

    def _request(self, url: str, params: dict) -> dict:
        """GET with optional throttle + 429/5xx backoff. Set POLYGON_MIN_INTERVAL
        (seconds between calls) on a rate-limited (free) tier, e.g. 13 for the
        5-req/min limit. Returns parsed JSON or raises after retries."""
        import time as _t
        interval = float(os.getenv("POLYGON_MIN_INTERVAL", "0"))
        resp = None
        for _ in range(6):
            if interval:
                wait = interval - (_t.time() - HistoryProvider._last_call)
                if wait > 0:
                    _t.sleep(wait)
            HistoryProvider._last_call = _t.time()
            resp = self._client().get(url, params=params, timeout=30)
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                _t.sleep(float(ra) if ra else max(15.0, interval or 15.0))
                continue
            if resp.status_code >= 500:
                _t.sleep(2.0)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def daily(self, symbol: str, lookback_days: int = 730) -> list[Bar]:
        cached = _read_cache(symbol)
        if cached:
            return cached
        if not self.key:
            raise RuntimeError(
                "POLYGON_API_KEY not set. The forward-likelihood panel needs real "
                "historical bars to compute frequencies. Set the key in .env, or "
                "run backtest.py with --synthetic to see the mechanics on fake data."
            )
        import datetime as dt
        end = dt.date.today()
        start = end - dt.timedelta(days=lookback_days)
        url = (f"{self.base}/v2/aggs/ticker/{symbol.upper()}/range/1/day/"
               f"{start.isoformat()}/{end.isoformat()}")
        body = self._request(url, {"adjusted": "true", "sort": "asc",
                                   "limit": 50000, "apiKey": self.key})
        results = body.get("results", []) or []
        bars = [Bar(t=x["t"], o=x["o"], h=x["h"], l=x["l"], c=x["c"],
                    v=int(x["v"]), vw=x.get("vw", x["c"])) for x in results]
        if bars:
            _write_cache(symbol, bars)
        return bars

    # ------------------------------------------------------------------
    # Whole-market endpoints for the broad-universe model pipeline.
    # ------------------------------------------------------------------
    def grouped_daily(self, date) -> dict[str, Bar]:
        """One call returns OHLCV for the ENTIRE US stock market for `date`
        (a datetime.date). Keyed by ticker. This is what makes whole-market
        training feasible on the cheap tier: ~one call per trading day instead
        of one per (symbol, day). Cached per date — immutable once the day has
        closed, so the cache TTL is effectively infinite for past dates."""
        import datetime as dt
        if isinstance(date, dt.datetime):
            date = date.date()
        iso = date.isoformat()
        cache = CACHE_DIR / "grouped" / f"{iso}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            try:
                raw = json.loads(cache.read_text())
                return {k: Bar(**v) for k, v in raw.items()}
            except Exception:
                pass
        if not self.key:
            raise RuntimeError("POLYGON_API_KEY not set (grouped_daily needs real data)")
        url = (f"{self.base}/v2/aggs/grouped/locale/us/market/stocks/{iso}")
        body = self._request(url, {"adjusted": "true", "apiKey": self.key})
        results = body.get("results", []) or []
        out: dict[str, Bar] = {}
        for x in results:
            t = x.get("T")
            if not t:
                continue
            out[t] = Bar(t=x.get("t", 0), o=x["o"], h=x["h"], l=x["l"], c=x["c"],
                         v=int(x["v"]), vw=x.get("vw", x["c"]))
        # only persist closed-day pulls (a non-empty result means the day traded)
        if out:
            cache.write_text(json.dumps({k: asdict(v) for k, v in out.items()}))
        return out

    def ticker_meta(self, symbol: str) -> dict:
        """Company name + sector/industry for a ticker, from Polygon ticker
        details (sic_description). Disk-cached ~30d (it rarely changes). Returns
        {} without a key. Used to replace 'Unknown' sectors on displayed rows."""
        d = CACHE_DIR / "meta"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{symbol.upper()}.json"
        if p.exists() and (time.time() - p.stat().st_mtime < 60 * 60 * 24 * 30):
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        if not self.key:
            return {}
        try:
            body = self._request(f"{self.base}/v3/reference/tickers/{symbol.upper()}",
                                 {"apiKey": self.key})
            res = body.get("results", {}) or {}
            sic = (res.get("sic_description") or "").strip()
            meta = {"name": res.get("name", ""), "sector": sic.title() if sic else ""}
        except Exception:  # noqa: BLE001
            meta = {}
        # only cache a successful lookup — never let a transient failure poison
        # the 30-day cache and pin a ticker to "Unknown".
        if meta.get("sector"):
            try:
                p.write_text(json.dumps(meta))
            except Exception:
                pass
        return meta

    def reference_tickers(self, type_: str = "CS") -> list[dict]:
        """Active common-stock tickers (paged). Cached to disk (membership
        changes slowly). Returns the raw ticker dicts (ticker, name, primary_exchange,
        type, ...)."""
        cache = CACHE_DIR / f"reference_{type_}.json"
        if cache.exists() and (time.time() - cache.stat().st_mtime < 60 * 60 * 24):
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        if not self.key:
            raise RuntimeError("POLYGON_API_KEY not set (reference_tickers needs real data)")
        out: list[dict] = []
        url = f"{self.base}/v3/reference/tickers"
        params = {"market": "stocks", "type": type_, "active": "true",
                  "limit": 1000, "apiKey": self.key}
        for _ in range(50):  # hard page cap (50k tickers) — safety, not expected
            body = self._request(url, params)
            out.extend(body.get("results", []) or [])
            nxt = body.get("next_url")
            if not nxt:
                break
            url = nxt
            params = {"apiKey": self.key}  # next_url already carries the cursor
        cache.write_text(json.dumps(out))
        return out
