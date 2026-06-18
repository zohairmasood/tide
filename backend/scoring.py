"""
Scoring engine.

Turns a Snapshot into a 0-100 composite momentum score plus the component
breakdown, so every leaderboard row is explainable. No probabilities, no
forecasts, no downside guarantees: just observable signals normalized and
weighted. The number means "how much is happening right now," not "this will
go up."

Each component returns 0-100. The composite is a weighted average. Tune
WEIGHTS to taste.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import statistics
from providers import Snapshot

WEIGHTS = {
    "rel_volume": 0.30,   # is unusual volume flowing in
    "momentum": 0.25,     # short-horizon price thrust
    "vwap_dist": 0.15,    # trading above the day's fair value
    "volatility": 0.15,   # range expanding vs its own baseline
    "gap_hold": 0.10,     # opened up and holding the gap
    "catalyst": 0.05,     # known event flag
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _rel_volume_score(s: Snapshot) -> float:
    if s.avg_volume <= 0:
        return 0.0
    # ratio of cumulative volume to average; >1 means heavier than usual.
    # Scaled so 1x -> ~40, 2x -> ~70, 3x+ -> ~100.
    ratio = s.volume / s.avg_volume
    return _clamp(40 * ratio)


def _momentum_score(s: Snapshot) -> float:
    if len(s.intraday) < 6:
        # fall back to session change off prev close
        chg = (s.last - s.prev_close) / s.prev_close * 100 if s.prev_close else 0
        return _clamp(50 + chg * 5)
    recent = s.intraday[-1]
    past = s.intraday[-6]  # ~6 ticks back
    pct = (recent - past) / past * 100 if past else 0
    # 0% -> 50 (neutral), +2% -> ~90, -2% -> ~10
    return _clamp(50 + pct * 20)


def _vwap_dist_score(s: Snapshot) -> float:
    if s.vwap <= 0:
        return 50.0
    pct = (s.last - s.vwap) / s.vwap * 100
    return _clamp(50 + pct * 25)


def _volatility_score(s: Snapshot) -> float:
    if len(s.intraday) < 10:
        return 50.0
    # realized vol of recent returns vs a flat baseline expectation
    rets = [
        (s.intraday[i] - s.intraday[i - 1]) / s.intraday[i - 1]
        for i in range(1, len(s.intraday)) if s.intraday[i - 1]
    ]
    if len(rets) < 2:
        return 50.0
    vol = statistics.pstdev(rets)
    # ~0.4% per-tick stdev -> ~100. Expansion reads high.
    return _clamp(vol / 0.004 * 100)


def _gap_hold_score(s: Snapshot) -> float:
    if s.prev_close <= 0:
        return 50.0
    gap = (s.open_ - s.prev_close) / s.prev_close * 100
    if gap <= 0:
        return _clamp(50 + gap * 10)  # gap down penalized
    # gapped up: reward if still trading above the open (holding it)
    hold = (s.last - s.open_) / s.open_ * 100 if s.open_ else 0
    return _clamp(60 + gap * 4 + hold * 8)


def _catalyst_score(s: Snapshot) -> float:
    return 100.0 if s.has_catalyst else 30.0


_COMPONENTS = {
    "rel_volume": _rel_volume_score,
    "momentum": _momentum_score,
    "vwap_dist": _vwap_dist_score,
    "volatility": _volatility_score,
    "gap_hold": _gap_hold_score,
    "catalyst": _catalyst_score,
}


@dataclass
class Scored:
    symbol: str
    sector: str
    last: float
    session_pct: float          # % change vs prev close
    composite: float            # 0-100
    components: dict             # each sub-score
    catalyst_note: str
    sparkline: list[float]

    def to_row(self) -> dict:
        d = asdict(self)
        d["composite"] = round(self.composite, 1)
        d["session_pct"] = round(self.session_pct, 2)
        d["components"] = {k: round(v, 1) for k, v in self.components.items()}
        return d


def score(s: Snapshot) -> Scored:
    comps = {name: fn(s) for name, fn in _COMPONENTS.items()}
    composite = sum(comps[name] * w for name, w in WEIGHTS.items())
    session_pct = (s.last - s.prev_close) / s.prev_close * 100 if s.prev_close else 0
    return Scored(
        symbol=s.symbol, sector=s.sector, last=s.last,
        session_pct=session_pct, composite=composite, components=comps,
        catalyst_note=s.catalyst_note, sparkline=s.intraday[-30:],
    )


def leaderboard(snaps: list[Snapshot], top: int | None = None) -> list[dict]:
    rows = sorted((score(s) for s in snaps), key=lambda r: r.composite, reverse=True)
    if top:
        rows = rows[:top]
    return [r.to_row() for r in rows]
