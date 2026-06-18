# Tide — Predictive Momentum Leaderboard

Live cross-sector dashboard that scans a broad screened universe and surfaces a
**top 5** ranked by a **calibrated model estimate** of a near-term +10% move,
gated by a downside-safety estimate. Every pick shows its supporting stats, a
Claude analyst note, and (on click) the real empirical forward-return record.

## What this is NOT (the contract, revised)
The earlier contract forbade any "X% chance of +10%" field outright. That has
been **deliberately revised** — not discarded. A probability *is* now shown,
but only under conditions that make it honest:

1. It is **trained and calibrated** on held-out future data (sigmoid/Platt over
   a strictly-later slice — never random folds).
2. It is **walk-forward validated** with an embargo gap, and the validation
   Brier / base rate / reliability bins are persisted in the artifact and shown
   in the UI (the trust badge).
3. It is **gated**: probabilities are displayed only when the model clears a
   calibration-quality bar (`MAX_CALIB_BRIER`, `MIN_CALIB_N`) AND the live
   features are sufficient (`data_sufficient`). Otherwise the UI falls back to
   the explainable activity score + empirical history.
4. It is always shown **with its base rate** so "18%" reads against a ~2% base.

What remains forbidden: an **uncalibrated, unvalidated, or unlabeled**
probability — a number that can't be checked against real outcomes. The
original intent ("a number you can't make true") is preserved; a *validated*
number can be made to mean what it says, and the UI proves it. Do not bypass
the gates. The empirical engine (`backtest.py`) stays as the ground truth.

## Architecture
- `backend/config.py` — single source of truth for target/label definitions
  (POP_TARGET_PCT, POP_HORIZON, SAFE_HORIZON, SAFE_BUFFER) + a minimal `.env`
  loader. `backtest.py` and `labels.py` both import these so they never drift.
- `backend/providers.py` — live snapshots. MockProvider (no key) / PolygonProvider
  (`universe()` delegates to the broad screener). `Snapshot` is the live-vs-train
  parity surface.
- `backend/scoring.py` — the explainable 0-100 activity composite (kept, pure).
- `backend/features.py` — the canonical feature vocabulary (`FEATURE_NAMES`,
  `feature_at`, `live_features`). No look-ahead. Shares the original 5 signals
  with `backtest._signal_at`.
- `backend/labels.py` — the two prediction targets (`pop`, `safe`).
- `backend/history.py` — daily bars + `grouped_daily` (whole-market/day) +
  `reference_tickers`, all disk-cached.
- `backend/universe.py` — per-date screened universe (survivorship-safe).
- `backend/dataset.py` — pooled cross-sectional matrix + market/sector context.
- `backend/train.py` — offline training/calibration/validation → joblib artifact
  + report.json. `--synthetic` runs with no key.
- `backend/model.py` — `MomentumModel` artifact loader/inference; the seam between
  training and serving. Degrades to no-model mode if absent.
- `backend/backtest.py` — empirical forward-likelihood engine (unchanged logic).
- `backend/news.py` — Polygon news retrieval for catalyst context.
- `backend/analyst.py` — Claude analyst note (claude-opus-4-8), structured output,
  explains the real numbers, never invents them. Cached/lazy.
- `backend/broker.py` — SIMULATED paper broker behind a `Broker` protocol. No
  external brokerage (user is not a US taxholder); fills modeled vs live prices
  ± `PAPER_SLIPPAGE_BPS`, persisted to `.paper/`. A real broker can drop in later.
- `backend/altdata.py` — congressional "smart-money" overlay (Quiver/Finnhub),
  disk-cached, graceful without a key. A labeled side-signal + tiebreak only.
- `backend/server.py` — FastAPI two-tier refresh loop + endpoints.
- `frontend/index.html` — self-contained top-5 dashboard. No build step.

## Paper execution (simulated) — the forward-validation layer
`broker.py` is an internal `PaperBroker` (no Alpaca/IBKR — the user can't open a
US brokerage). The refresh loop calls `broker.mark(prices)` each tick to re-mark
open positions and fire stop/target exits; the equity curve is the honest
forward record of the model. Execution is **human-in-the-loop**: the only thing
that opens a position is the dashboard "Arm paper trade" button (`POST
/api/paper/arm`), enabled only when the pick's gauges are shown (gates pass +
model trustworthy). Arming places exactly the model's thesis: target = +`POP_TARGET_PCT`%,
stop = −`STOP_PCT`. The `Broker` protocol is the seam for a future real broker.
Everything is badged SIMULATED in the UI. Do not add autonomous trading.

## Smart-money overlay (optional)
`altdata.py` surfaces recent disclosed US-congressional trades per ticker as raw
facts (who/side/size/disclosure-lag), shown like the empirical panel. It is a
labeled side-signal and at most a rank **tiebreak among already-gated picks** — it
**never** alters the calibrated `P(pop)`. Needs `QUIVER_API_KEY` or `FINNHUB_API_KEY`
(no brokerage/citizenship requirement); hidden gracefully without one.

## Feature parity / train-serve skew (known limitation)
The model trains on **settled daily bars**; the live `Snapshot` is partial
intraday. `live_features` therefore scores **as of the last completed daily
bar** rather than feeding partial-day numbers into a daily-trained model, and
sets `data_sufficient=False` when history is missing. The leaderboard thus
reflects completed-session activity, not tick-by-tick. `/api/calibration` is the
live check that the daily-trained model still calibrates.

## Honesty guardrails (binding for any UI work)
1. Every probability is tagged "model estimate (calibrated)" — never a bare %.
2. Trust (Brier, base rate, reliability) is shown inline + in the trust modal.
3. `data_sufficient=false` → "insufficient live data to estimate" + empirical only.
4. `trustworthy=false` (Brier above ceiling / n below floor) → probabilities
   suppressed app-wide; server enforces by withholding.
5. Ranking is by `P(pop)` only among names passing `P(safe) ≥ SAFE_THRESHOLD`.
6. The empirical panel stays on every card.
7. The disclaimer frames the estimate honestly (validated, gated, not a guarantee).

## Run
    cd backend
    python -m venv .venv && .venv\Scripts\activate    # Windows; or source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env                               # mock feed works with no key
    python train.py --synthetic                        # builds a demo model artifact, no key
    uvicorn server:app --reload --port 8000
Then open ../frontend/index.html (or serve it). With the synthetic artifact the
UI shows a "synthetic" banner; with no artifact it ranks by the activity score.

## Go live with real data
Set in `.env`: `DATA_PROVIDER=polygon`, `POLYGON_API_KEY=...`, optionally
`ANTHROPIC_API_KEY=...` for analyst notes. Then:
    python train.py --start 2024-06-01 --end 2026-06-01   # trains on 2y, writes report.json
Polygon Stocks Starter (~$30/mo, unlimited calls, 15-min delay, 2y history)
covers grouped-daily training, live snapshots, and news.

## Endpoints
- GET  /api/health                liveness + model status
- GET  /api/leaderboard           top-N rows with a `prediction` block
- WS   /ws/leaderboard            pushes the leaderboard
- GET  /api/likelihood/{symbol}   empirical forward-return frequencies (unchanged)
- GET  /api/calibration           model trust metrics (Brier, base rate, reliability)
- GET  /api/analyst/{symbol}      Claude analyst note (lazy, cached)
- POST /api/paper/arm             open a simulated bracket trade {symbol,stop_pct?,target_pct?,notional?}
- POST /api/paper/close/{symbol}  manual exit
- GET  /api/paper/portfolio       simulated portfolio state (equity, positions, closed trades)
- POST /api/paper/reset           reset paper account to starting cash
- GET  /api/smartmoney/{symbol}   disclosed congressional trades (needs Quiver/Finnhub key)

## Test without a key
    python backtest.py --synthetic   # empirical engine output shape
    python train.py --synthetic      # full train→calibrate→validate→artifact

## Good next steps
- Intraday minute bars (the `intraday_bars` slot in `feature_at`) for a truer
  24-48h horizon — the single biggest v2 lift, deferred for cost/parity reasons.
- SHAP attributions in place of the lightweight z×importance contribution proxy.
- Pre-warm analyst notes for the current top 5 at the end of each scan.
- Per-sector model heads / regime conditioning.
