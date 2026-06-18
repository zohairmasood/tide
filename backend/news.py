"""
Catalyst / news retrieval (Polygon news endpoint).

The project author's core objection to a probability field was that the events
driving 10% moves (earnings, M&A, FDA) aren't in the price signals. This module
retrieves the missing context so the Claude analyst note can ground its
narrative in real catalysts — never to fabricate one. With no key (or no
results) it returns an empty list and the caller falls back to the snapshot's
catalyst flag; it never invents a headline.
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("NEWS_CACHE", "./.news_cache"))
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 60 * 60  # 1h


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def recent_news(symbol: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` recent news items: {title, publisher, published, url}."""
    p = _cache_path(symbol)
    if p.exists() and (time.time() - p.stat().st_mtime < CACHE_TTL):
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        return []
    try:
        import requests
        r = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={"ticker": symbol.upper(), "limit": limit, "apiKey": key},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", []) or []
    except Exception:
        return []
    items = [{
        "title": x.get("title", ""),
        "publisher": (x.get("publisher") or {}).get("name", ""),
        "published": x.get("published_utc", ""),
        "url": x.get("article_url", ""),
    } for x in results]
    try:
        p.write_text(json.dumps(items))
    except Exception:
        pass
    return items
