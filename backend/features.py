"""
Feature engineering — the single, canonical feature vocabulary.

Every feature is computable from a list of daily `Bar`s using ONLY bars at or
before index `i` (no look-ahead). The model's training pipeline (dataset.py)
and the live serving path (server.py via `live_features`) both build their
inputs here, against the same `FEATURE_NAMES` ordering, so the model never sees
a differently-shaped or differently-ordered vector than it was trained on.

The first five features (rel_volume, momentum_5d, gap, above_vwap,
vol_expansion) intentionally mirror backtest._signal_at so the predictive model
and the honest empirical engine speak the same language.

Live-vs-training skew (the biggest serving risk): the model is trained on
SETTLED daily bars. The live Snapshot is a partial intraday view. `live_features`
therefore scores "as of the last completed daily bar" rather than feeding
partial-day numbers into a daily-trained model, and flags data sufficiency. See
the plan / CLAUDE.md "Feature parity" note.
"""
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, asdict, fields

from history import Bar

# Minimum bars of history required before any feature row is valid.
# SMA50 is the longest trailing window.
WARMUP = 50

# Canonical, ordered feature names. The model artifact persists this list and
# server-side loading validates it against this module — if they diverge, fail
# loudly rather than silently scoring a misaligned vector.
FEATURE_NAMES: list[str] = [
    "rel_volume",
    "rel_volume_5d",
    "dollar_volume_log",
    "momentum_5d",
    "momentum_10d",
    "momentum_21d",
    "dist_sma20",
    "dist_sma50",
    "pct_to_window_high",
    "dist_from_recent_high_20",
    "gap",
    "gap_held",
    "above_vwap",
    "close_loc",
    "range_pct",
    "atr_pct_14",
    "realized_vol_20d",
    "vol_expansion",
    "up_days_5",
    "consec_up",
    "mkt_ret_5d",
    "sector_ret_5d",
]


@dataclass
class FeatureVector:
    rel_volume: float
    rel_volume_5d: float
    dollar_volume_log: float
    momentum_5d: float
    momentum_10d: float
    momentum_21d: float
    dist_sma20: float
    dist_sma50: float
    pct_to_window_high: float
    dist_from_recent_high_20: float
    gap: float
    gap_held: float
    above_vwap: float
    close_loc: float
    range_pct: float
    atr_pct_14: float
    realized_vol_20d: float
    vol_expansion: float
    up_days_5: float
    consec_up: float
    mkt_ret_5d: float
    sector_ret_5d: float

    def to_array(self) -> "list[float]":
        d = asdict(self)
        return [d[name] for name in FEATURE_NAMES]

    def to_dict(self) -> dict:
        return asdict(self)


# sanity: the dataclass and FEATURE_NAMES must stay in lockstep
assert [f.name for f in fields(FeatureVector)] == FEATURE_NAMES, (
    "FeatureVector fields and FEATURE_NAMES are out of sync"
)


def _returns(bars: list[Bar]) -> list[float]:
    return [
        (bars[i].c - bars[i - 1].c) / bars[i - 1].c
        for i in range(1, len(bars))
        if bars[i - 1].c
    ]


def _nan() -> float:
    return float("nan")


def feature_at(
    bars: list[Bar],
    i: int,
    *,
    mkt_ret_5d: float | None = None,
    sector_ret_5d: float | None = None,
    intraday_bars: list | None = None,  # v2 slot — unused in v1
) -> FeatureVector | None:
    """Compute the full feature vector as it would have looked at the CLOSE of
    day `i`, using only bars[:i+1]. Returns None before warmup.

    `intraday_bars` is a deliberately-unused v2 hook: minute bars for the day
    would let us add opening-range / first-hour features for a truer 24-48h
    horizon. Left unfilled in v1 to keep the daily pipeline cheap and to avoid
    a third data representation widening the live-vs-train skew.
    """
    if i < WARMUP or i >= len(bars):
        return None
    b = bars[i]
    prev = bars[i - 1]

    # --- volume ---
    vol20 = [x.v for x in bars[i - 20:i]]
    avg_vol = statistics.mean(vol20) if vol20 else 0.0
    rel_volume = (b.v / avg_vol) if avg_vol else 1.0
    avg_vol5 = statistics.mean([x.v for x in bars[i - 5:i]]) if i >= 5 else avg_vol
    rel_volume_5d = (avg_vol5 / avg_vol) if avg_vol else 1.0
    dollar_volume_log = math.log(b.c * b.v) if (b.c > 0 and b.v > 0) else 0.0

    # --- momentum (multi-horizon) ---
    def mom(n: int) -> float:
        past = bars[i - n].c
        return ((b.c - past) / past * 100) if past else 0.0

    momentum_5d = mom(5)
    momentum_10d = mom(10)
    momentum_21d = mom(21)

    # --- trend / position ---
    sma20 = statistics.mean([x.c for x in bars[i - 20:i]]) if i >= 20 else b.c
    sma50 = statistics.mean([x.c for x in bars[i - 50:i]]) if i >= 50 else b.c
    dist_sma20 = ((b.c - sma20) / sma20 * 100) if sma20 else 0.0
    dist_sma50 = ((b.c - sma50) / sma50 * 100) if sma50 else 0.0

    window = bars[max(0, i - 252):i + 1]  # up to ~1y
    win_high = max(x.h for x in window)
    win_low = min(x.l for x in window)
    pct_to_window_high = ((b.c - win_high) / win_high * 100) if win_high else 0.0

    high20 = max(x.h for x in bars[i - 20:i + 1])
    dist_from_recent_high_20 = ((b.c - high20) / high20 * 100) if high20 else 0.0

    # --- gap / intraday position ---
    gap = ((b.o - prev.c) / prev.c * 100) if prev.c else 0.0
    gap_held = ((b.c - b.o) / b.o * 100) if b.o else 0.0
    above_vwap = ((b.c - b.vw) / b.vw * 100) if b.vw else 0.0
    rng = (b.h - b.l)
    close_loc = ((b.c - b.l) / rng) if rng else 0.5
    range_pct = (rng / b.c * 100) if b.c else 0.0

    # --- volatility ---
    trs = []
    for j in range(i - 14, i):
        if j < 1:
            continue
        hi, lo, pc = bars[j].h, bars[j].l, bars[j - 1].c
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    atr = statistics.mean(trs) if trs else 0.0
    atr_pct_14 = (atr / b.c * 100) if b.c else 0.0

    r20 = _returns(bars[i - 20:i + 1])
    realized_vol_20d = (statistics.pstdev(r20) * 100) if len(r20) > 1 else 0.0
    r5 = _returns(bars[i - 5:i + 1])
    sd5 = statistics.pstdev(r5) if len(r5) > 1 else 0.0
    sd20 = statistics.pstdev(r20) if len(r20) > 1 else 0.0
    vol_expansion = (sd5 / sd20) if sd20 else 1.0

    # --- streaks ---
    up_days_5 = float(sum(1 for j in range(i - 4, i + 1) if j >= 1 and bars[j].c > bars[j - 1].c))
    consec_up = 0.0
    j = i
    while j >= 1 and bars[j].c > bars[j - 1].c:
        consec_up += 1
        j -= 1

    return FeatureVector(
        rel_volume=rel_volume,
        rel_volume_5d=rel_volume_5d,
        dollar_volume_log=dollar_volume_log,
        momentum_5d=momentum_5d,
        momentum_10d=momentum_10d,
        momentum_21d=momentum_21d,
        dist_sma20=dist_sma20,
        dist_sma50=dist_sma50,
        pct_to_window_high=pct_to_window_high,
        dist_from_recent_high_20=dist_from_recent_high_20,
        gap=gap,
        gap_held=gap_held,
        above_vwap=above_vwap,
        close_loc=close_loc,
        range_pct=range_pct,
        atr_pct_14=atr_pct_14,
        realized_vol_20d=realized_vol_20d,
        vol_expansion=vol_expansion,
        up_days_5=up_days_5,
        consec_up=consec_up,
        mkt_ret_5d=mkt_ret_5d if mkt_ret_5d is not None else _nan(),
        sector_ret_5d=sector_ret_5d if sector_ret_5d is not None else _nan(),
    )


def live_features(
    symbol: str,
    bars: list[Bar],
    *,
    mkt_ret_5d: float | None = None,
    sector_ret_5d: float | None = None,
) -> tuple[dict | None, bool, str]:
    """Build the live feature dict for scoring, computed AS OF THE LAST COMPLETED
    DAILY BAR (not partial-intraday) to keep parity with how the model was
    trained. Returns (features_dict | None, data_sufficient, note).

    v1 deliberately scores at the last close. This means the leaderboard
    reflects completed-session activity, not tick-by-tick — documented as a
    known limitation. The empirical panel and /api/calibration are the live
    honesty checks that the daily-trained model still calibrates.
    """
    if not bars or len(bars) <= WARMUP:
        return None, False, "insufficient daily history for model features"
    fv = feature_at(bars, len(bars) - 1, mkt_ret_5d=mkt_ret_5d, sector_ret_5d=sector_ret_5d)
    if fv is None:
        return None, False, "feature computation failed (warmup)"
    return fv.to_dict(), True, ""
