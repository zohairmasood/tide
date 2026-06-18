"""
Dataset assembly — pooled, cross-sectional, leakage-aware.

Turns per-symbol daily bars into one big training matrix shared by both label
heads. Pooling thousands of names on the same calendar dates is what makes the
rare positive class (a +10% move in 2 days) learnable AND calibratable — a
single name has far too few positives to fit, let alone hold out for honest
calibration.

Cross-sectional context (market / sector 5-day return on each date) is computed
here and fed into the feature vector; it is the biggest lift over per-symbol
features and the reason pooling wins.

Every row carries its date and symbol in `meta` so train.py can split by time
with an embargo gap (the safe-label window must not straddle the train/test
boundary — the subtlest leak).
"""
from __future__ import annotations
import statistics

from features import feature_at, FEATURE_NAMES, WARMUP
from labels import labels_at
from config import SAFE_HORIZON


def _date_iso(t_ms: int) -> str:
    import datetime as dt
    if not t_ms:
        return ""
    return dt.datetime.utcfromtimestamp(t_ms / 1000).date().isoformat()


def _context_by_date(bars_by_symbol: dict, sector_by_symbol: dict):
    """Compute market and per-sector 5-day return for each date key (bar.t).

    Returns (mkt[t] -> float, sector[(sector,t)] -> float)."""
    mkt_sum: dict[int, float] = {}
    mkt_cnt: dict[int, int] = {}
    sec_sum: dict[tuple, float] = {}
    sec_cnt: dict[tuple, int] = {}
    for sym, bars in bars_by_symbol.items():
        sector = sector_by_symbol.get(sym, "Unknown")
        for i in range(5, len(bars)):
            past = bars[i - 5].c
            if not past:
                continue
            r5 = (bars[i].c - past) / past
            t = bars[i].t
            mkt_sum[t] = mkt_sum.get(t, 0.0) + r5
            mkt_cnt[t] = mkt_cnt.get(t, 0) + 1
            key = (sector, t)
            sec_sum[key] = sec_sum.get(key, 0.0) + r5
            sec_cnt[key] = sec_cnt.get(key, 0) + 1
    mkt = {t: mkt_sum[t] / mkt_cnt[t] * 100 for t in mkt_sum}
    sec = {k: sec_sum[k] / sec_cnt[k] * 100 for k in sec_sum}
    return mkt, sec


def build_dataset(bars_by_symbol: dict, sector_by_symbol: dict):
    """Returns (X, y_pop, y_safe, meta) as numpy arrays + a list of {date,symbol}.

    Only rows where BOTH labels are known (full forward windows) are kept, so
    each row is trainable for both heads.
    """
    import numpy as np

    mkt, sec = _context_by_date(bars_by_symbol, sector_by_symbol)

    rows: list[list[float]] = []
    y_pop: list[int] = []
    y_safe: list[int] = []
    meta: list[dict] = []

    for sym, bars in bars_by_symbol.items():
        sector = sector_by_symbol.get(sym, "Unknown")
        # need full safe window ahead; pop window is shorter so it's covered
        last_usable = len(bars) - SAFE_HORIZON - 1
        for i in range(WARMUP, last_usable):
            lab = labels_at(bars, i)
            if lab.pop is None or lab.safe is None:
                continue
            t = bars[i].t
            fv = feature_at(
                bars, i,
                mkt_ret_5d=mkt.get(t),
                sector_ret_5d=sec.get((sector, t)),
            )
            if fv is None:
                continue
            rows.append(fv.to_array())
            y_pop.append(lab.pop)
            y_safe.append(lab.safe)
            meta.append({"date": _date_iso(t), "symbol": sym})

    X = np.array(rows, dtype=float) if rows else np.empty((0, len(FEATURE_NAMES)))
    return X, np.array(y_pop), np.array(y_safe), meta


def feature_stats(X) -> dict:
    """Per-feature mean/std over the (training) matrix, ignoring NaNs. Used for
    the UI 'why this pick' attribution."""
    import numpy as np
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            out[name] = {"mean": 0.0, "std": 0.0}
        else:
            out[name] = {"mean": float(np.mean(col)), "std": float(np.std(col) or 0.0)}
    return out
