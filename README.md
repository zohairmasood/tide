# Tide — Predictive Momentum Leaderboard

Scans a broad cross-sector universe and surfaces a **top 5** ranked by a
**calibrated model estimate** of a +10% move within 1–2 days **P(pop)**, gated
by a downside-safety estimate **P(safe)** (won't drop below the current price
for ~3 weeks). Each pick shows the model's supporting stats, a Claude analyst
note explaining the numbers, and the real empirical history. Inspired by
FinRobot; built to stay honest. Not investment advice.

## Run
    cd backend
    pip install -r requirements.txt
    cp .env.example .env          # mock feed works with no key
    python train.py --synthetic   # builds a demo model artifact (no key needed)
    uvicorn server:app --port 8000
Then open ../frontend/index.html (or serve it). On the synthetic artifact the UI
shows a "synthetic" banner; with no artifact it ranks by the activity score.

## Go live
Set in `.env`:
    DATA_PROVIDER=polygon
    POLYGON_API_KEY=your_key
    ANTHROPIC_API_KEY=your_key      # optional — enables analyst notes
Then train on real data (Polygon Stocks Starter ~$30/mo, 2y history):
    python train.py --start 2024-06-01 --end 2026-06-01
This writes the model artifact and a `report.json` with the held-out validation
metrics. Inspect those before trusting the leaderboard.

## Model-backed leaderboard
A two-tier loop drives the board: an expensive broad scan every `SCREEN_SECONDS`
keeps a ~50-name candidate pool (ranked by `P(pop)`, gated by `P(safe)`), and a
cheap re-rank every `REFRESH_SECONDS` publishes the top 5. Each row carries a
`prediction` block with `p_pop`, `p_safe`, the safe-gate result, and the top
feature drivers ("why this pick").

## Calibration & trust
The model is calibrated on a strictly-later held-out slice and walk-forward
validated with an embargo gap, so it can't peek at the future. `/api/calibration`
exposes the validation **Brier score**, **base rate**, and **reliability bins**;
the trust badge surfaces them. Probabilities are shown **only** when the model
clears a calibration-quality bar AND the live data is sufficient — otherwise the
UI falls back to the explainable activity score and the empirical history.

## What it is not — and why the old prohibition was lifted
Earlier versions refused to show *any* "X% chance" because an uncalibrated number
can't be made true. That's still the rule for any unvalidated number. What's new
is that the shown probability is **calibrated, validated, gated, and base-rate
contextualized** — a number that *can* be made to mean what it says, with the
evidence one click away. It is still an estimate, not a guarantee: the events
that drive large moves are only partly in the signals, and big fast moves are
rare, so an honest list is often short.

## Forward-likelihood panel (the ground truth, unchanged)
Expand any card. `/api/likelihood/{symbol}`:
  1. pulls real daily history (Polygon aggregates, disk-cached)
  2. computes the same signals at every past day (`backtest.py`)
  3. finds past days similar to now
  4. reports what ACTUALLY happened next: frequency of a +10% move within
     1–2 days, median forward return + IQR, round-trip-to-entry rate
  5. shows sample size (n) on every figure
The model estimate is checked against this. Run `python backtest.py --synthetic`
to see its shape with no key.

## Analyst note
On expand, `/api/analyst/{symbol}` calls Claude (`claude-opus-4-8`) with ONLY the
pre-computed stats + retrieved news; it explains the numbers and the catalyst
context and is instructed never to invent or adjust a probability. A post-check
flags any percentage in the note that isn't in the source stats. Disabled (and
hidden) when `ANTHROPIC_API_KEY` is unset.

## Paper trading (simulated, no brokerage)
Tide has no external broker — it ships an **internal simulated paper broker**
(`broker.py`). Click **"Arm paper trade"** on a pick to open a simulated bracket
(target +10%, stop −5%) at the live price; the refresh loop marks it each tick
and fires the stop/target. The header **paper** button opens a portfolio drawer
with the equity curve, open positions (live P&L), and closed-trade log. Fills are
modeled vs live prices ± a slippage assumption and are clearly badged SIMULATED —
this is the honest forward-validation of the model, not real trading. The `Broker`
protocol lets a real broker drop in later. Tide never trades on its own.

## Smart money (optional)
With a `QUIVER_API_KEY` or `FINNHUB_API_KEY`, cards show recent **disclosed US
congressional trades** for the ticker (who, side, size, disclosure lag) plus a
"net congress buys" badge. It's a labeled side-signal and at most a rank
tiebreak — it never changes the calibrated probability. Hidden without a key.

## Charts
Expanding a card embeds a free TradingView candlestick chart for the symbol
(frontend-only, no key).
