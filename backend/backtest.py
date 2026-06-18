"""
Backtest / forward-likelihood engine.

This is the honest core of the "chance it pops" feature. Instead of inventing a
confidence number, it walks a symbol's real history, computes the same signals
at each past day, and records what ACTUALLY happened over the next 1-2 days.

For a given current setup it answers, from real data:
  - of past days that looked like this, what fraction rose >= TARGET_PCT
    within HORIZON_DAYS?
  - what was the median forward return?
  - how often did price round-trip back to the entry within RECOVER_DAYS
    after first dropping? (the closest honest read on the "back to my entry"
    question, with no guarantee implied)
  - n = sample size, shown always, because a frequency on n=4 is noise.

Signals computed per historical day (mirrors scoring.py intent on daily bars):
  - rel_volume:   day volume / trailing 20d avg volume
  - momentum_5d:  5-day price change %
  - gap:          open vs prior close %
  - above_vwap:   close vs day vwap %
  - vol_expansion: 5d realized vol / 20d realized vol

"Similar setup" = current signal vector within tolerance bands of a past day.
Tolerances are deliberately loose so samples aren't trivially tiny; you can
tighten them and watch n shrink, which is itself informative.
"""
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass
from history import Bar, HistoryProvider

# Label/target definitions live in config.py so the empirical engine here and
# the predictive model's labels (labels.py) can never drift apart.
from config import TARGET_PCT, HORIZON_DAYS, RECOVER_DAYS  # noqa: F401


@dataclass
class SignalVector:
    rel_volume: float
    momentum_5d: float
    gap: float
    above_vwap: float
    vol_expansion: float


@dataclass
class Likelihood:
    symbol: str
    n: int                       # sample size of similar past setups
    freq_target: float | None    # fraction that hit +TARGET_PCT within horizon (0-1)
    median_fwd: float | None     # median forward return % over horizon
    p25_fwd: float | None
    p75_fwd: float | None
    recover_rate: float | None   # of setups that dropped, fraction back to entry in RECOVER_DAYS
    note: str = ""

    def to_dict(self) -> dict:
        def pct(x): return None if x is None else round(x * 100, 1)
        def r(x): return None if x is None else round(x, 2)
        return {
            "symbol": self.symbol, "n": self.n,
            "freq_target": pct(self.freq_target),
            "median_fwd": r(self.median_fwd),
            "p25_fwd": r(self.p25_fwd), "p75_fwd": r(self.p75_fwd),
            "recover_rate": pct(self.recover_rate),
            "note": self.note,
            "target_pct": TARGET_PCT, "horizon_days": HORIZON_DAYS,
        }


def _returns(bars: list[Bar]) -> list[float]:
    return [(bars[i].c - bars[i-1].c) / bars[i-1].c for i in range(1, len(bars)) if bars[i-1].c]


def _signal_at(bars: list[Bar], i: int) -> SignalVector | None:
    """Compute the signal vector as it would have looked at the close of day i."""
    if i < 20:
        return None
    b = bars[i]
    avg_vol = statistics.mean(x.v for x in bars[i-20:i]) or 1
    rel_volume = b.v / avg_vol
    momentum_5d = (b.c - bars[i-5].c) / bars[i-5].c * 100 if bars[i-5].c else 0
    gap = (b.o - bars[i-1].c) / bars[i-1].c * 100 if bars[i-1].c else 0
    above_vwap = (b.c - b.vw) / b.vw * 100 if b.vw else 0
    r5 = _returns(bars[i-5:i+1]); r20 = _returns(bars[i-20:i+1])
    sd5 = statistics.pstdev(r5) if len(r5) > 1 else 0
    sd20 = statistics.pstdev(r20) if len(r20) > 1 else 0
    vol_expansion = (sd5 / sd20) if sd20 else 1.0
    return SignalVector(rel_volume, momentum_5d, gap, above_vwap, vol_expansion)


def _similar(a: SignalVector, b: SignalVector) -> bool:
    return (
        abs(a.rel_volume - b.rel_volume) <= max(0.5, 0.4 * a.rel_volume) and
        abs(a.momentum_5d - b.momentum_5d) <= 4.0 and
        abs(a.gap - b.gap) <= 2.5 and
        abs(a.above_vwap - b.above_vwap) <= 2.0 and
        abs(a.vol_expansion - b.vol_expansion) <= 0.6
    )


def _forward_outcome(bars: list[Bar], i: int) -> tuple[float, bool, bool, bool] | None:
    """Returns (fwd_return_pct, hit_target, dropped, recovered) for setup at day i."""
    if i + HORIZON_DAYS >= len(bars):
        return None
    entry = bars[i].c
    if not entry:
        return None
    window = bars[i+1:i+1+HORIZON_DAYS]
    max_high = max(x.h for x in window)
    fwd_return = (window[-1].c - entry) / entry * 100
    hit_target = (max_high - entry) / entry * 100 >= TARGET_PCT
    # round-trip question: did it dip below entry then come back within RECOVER_DAYS?
    rec_window = bars[i+1:i+1+RECOVER_DAYS]
    dropped = False
    recovered = False
    first_drop_idx = None
    for j, x in enumerate(rec_window):
        if not dropped and x.l < entry:
            dropped = True
            first_drop_idx = j
        elif dropped and j > first_drop_idx and x.h >= entry:
            recovered = True
            break
    return fwd_return, hit_target, dropped, recovered


def likelihood_for(symbol: str, bars: list[Bar]) -> Likelihood:
    if len(bars) < 60:
        return Likelihood(symbol, 0, None, None, None, None, None,
                          note="insufficient history")
    current = _signal_at(bars, len(bars) - 1)
    if current is None:
        return Likelihood(symbol, 0, None, None, None, None, None, note="no current signal")

    fwd_returns: list[float] = []
    hits = drops = recovers = 0
    # scan all but the most recent HORIZON days (those have no known outcome yet)
    for i in range(20, len(bars) - HORIZON_DAYS - 1):
        past = _signal_at(bars, i)
        if past is None or not _similar(current, past):
            continue
        out = _forward_outcome(bars, i)
        if out is None:
            continue
        fwd, hit, dropped, recovered = out
        fwd_returns.append(fwd)
        hits += hit
        if dropped:
            drops += 1
            recovers += recovered

    n = len(fwd_returns)
    if n == 0:
        return Likelihood(symbol, 0, None, None, None, None, None,
                          note="no comparable setups in history")
    fwd_returns.sort()
    def q(p):
        k = max(0, min(n - 1, int(p * (n - 1))))
        return fwd_returns[k]
    return Likelihood(
        symbol=symbol, n=n,
        freq_target=hits / n,
        median_fwd=q(0.5), p25_fwd=q(0.25), p75_fwd=q(0.75),
        recover_rate=(recovers / drops) if drops else None,
        note="thin sample, treat as noise" if n < 12 else "",
    )


# convenience for the server
_hist = HistoryProvider()

def likelihood(symbol: str) -> dict:
    try:
        bars = _hist.daily(symbol)
    except RuntimeError as e:
        return {"symbol": symbol, "n": 0, "error": str(e)}
    return likelihood_for(symbol, bars).to_dict()


if __name__ == "__main__":
    import sys, json
    if "--synthetic" in sys.argv:
        # demonstrate mechanics on a synthetic random walk so you can see output shape
        import random
        random.seed(7)
        bars, price = [], 100.0
        for d in range(500):
            drift = random.uniform(-0.01, 0.012)
            price = max(1, price * (1 + drift + random.gauss(0, 0.02)))
            o = price * random.uniform(0.99, 1.01)
            h = max(o, price) * random.uniform(1.0, 1.03)
            l = min(o, price) * random.uniform(0.97, 1.0)
            bars.append(Bar(t=d, o=round(o, 2), h=round(h, 2), l=round(l, 2),
                            c=round(price, 2), v=int(random.uniform(1e6, 5e6)),
                            vw=round(price, 2)))
        print(json.dumps(likelihood_for("SYNTH", bars).to_dict(), indent=2))
    else:
        sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        print(json.dumps(likelihood(sym), indent=2))
