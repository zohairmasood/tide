"""
Labels — the two prediction targets, defined precisely against future bars.

Both labels use ONLY bars after `i` and return None when the forward window is
incomplete (so an unlabeled, in-progress setup is never trained on as if its
outcome were known). They mirror backtest._forward_outcome's logic and share
the target/horizon constants in config.py, so the predictive target equals the
question the empirical panel answers.

  pop  : did the high reach +POP_TARGET_PCT within POP_HORIZON trading days?
         (intraday high — a move that touches and fades still "popped", matching
         the empirical engine.)
  safe : did the price stay at/above entry (minus SAFE_BUFFER) for every day of
         the next SAFE_HORIZON days? (intraday LOW — "won't go lower than current
         price" is a path/drawdown property; a close-only test would call a name
         safe that gapped down hard intraday.)

The two are negatively correlated by construction in volatile names — the same
volatility that produces a +10% pop also tends to breach the entry low. That is
exactly why they are modeled as two separate calibrated classifiers.
"""
from __future__ import annotations
from dataclasses import dataclass

from history import Bar
from config import POP_TARGET_PCT, POP_HORIZON, SAFE_HORIZON, SAFE_BUFFER


@dataclass
class Labels:
    pop: int | None
    safe: int | None


def pop_label(bars: list[Bar], i: int) -> int | None:
    """1 if max intraday high over the next POP_HORIZON days >= +POP_TARGET_PCT."""
    if i + POP_HORIZON >= len(bars):
        return None
    entry = bars[i].c
    if not entry:
        return None
    window = bars[i + 1:i + 1 + POP_HORIZON]
    if not window:
        return None
    max_high = max(x.h for x in window)
    return 1 if (max_high - entry) / entry * 100 >= POP_TARGET_PCT else 0


def safe_label(bars: list[Bar], i: int, buffer: float | None = None) -> int | None:
    """1 if the intraday low never dropped below entry*(1-buffer) over the next
    SAFE_HORIZON days. buffer defaults to config.SAFE_BUFFER (fraction)."""
    buf = SAFE_BUFFER if buffer is None else buffer
    if i + SAFE_HORIZON >= len(bars):
        return None
    entry = bars[i].c
    if not entry:
        return None
    window = bars[i + 1:i + 1 + SAFE_HORIZON]
    if not window:
        return None
    floor = entry * (1.0 - buf)
    min_low = min(x.l for x in window)
    return 1 if min_low >= floor else 0


def labels_at(bars: list[Bar], i: int) -> Labels:
    return Labels(pop=pop_label(bars, i), safe=safe_label(bars, i))


def base_rates(values, buffer_alt: float = 0.01) -> dict:
    """Summarize positive fraction of an iterable of 0/1/None labels.

    Returns {"n": int, "positive": int, "rate": float | None}. None values
    (incomplete windows) are excluded. `buffer_alt` is accepted for API symmetry
    with callers that report strict-vs-buffered safe rates; it is not used here
    (the caller supplies the already-computed label list)."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    pos = sum(vals)
    return {"n": n, "positive": pos, "rate": (pos / n) if n else None}
