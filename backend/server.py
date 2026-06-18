"""
FastAPI backend.

Two-tier refresh:
  - _screen_loop : every SCREEN_SECONDS, score the full screened universe,
    gate by P(safe) >= SAFE_THRESHOLD, rank by P(pop), keep the top
    CANDIDATE_POOL names as the working pool. Expensive.
  - _refresh_loop: every REFRESH_SECONDS, re-score only the pool and publish
    the top TOP_N. Cheap (daily-bar features are disk-cached).

Probabilities come from the calibrated model artifact (model.py). If no artifact
is present, or live features are insufficient, the server degrades gracefully:
the leaderboard ranks by the explainable composite and predictions report
unavailable — preserving the no-credentials dev path.

Endpoints:
  GET  /api/health                liveness + model status
  GET  /api/leaderboard           ranked top-N snapshot (with predictions)
  WS   /ws/leaderboard            pushes the snapshot every REFRESH_SECONDS
  GET  /api/likelihood/{symbol}   empirical forward-return frequencies (unchanged)
  GET  /api/calibration           model trust metrics (Brier, base rate, reliability)
  GET  /api/analyst/{symbol}      Claude analyst note (see analyst.py)
"""
from __future__ import annotations
import os
import time
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from providers import get_provider
from scoring import score as score_snap
from backtest import likelihood as compute_likelihood
from config import MODEL_PATH
from model import MomentumModel
from features import live_features
from history import HistoryProvider
import analyst
import altdata
from news import recent_news
from broker import PaperBroker

REFRESH_SECONDS = float(os.getenv("REFRESH_SECONDS", "3"))
SCREEN_SECONDS = float(os.getenv("SCREEN_SECONDS", "900"))
TOP_N = int(os.getenv("TOP_N", "5"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "50"))
SAFE_THRESHOLD = float(os.getenv("SAFE_THRESHOLD", "0.55"))

provider = get_provider()
_universe = provider.universe()
_hist = HistoryProvider()


def _load_model() -> MomentumModel | None:
    m = MomentumModel.load(MODEL_PATH)
    if m is None:
        # dev convenience: fall back to the synthetic artifact if present
        m = MomentumModel.load(MODEL_PATH.replace(".joblib", "_synthetic.joblib"))
    return m


MODEL = _load_model()
print(f"[server] model loaded: {bool(MODEL)} "
      f"({MODEL.metadata.get('version') if MODEL else 'none'})")

# shared state
_latest: dict = {"rows": [], "ts": 0.0}
_pool: list[str] = list(_universe)[:CANDIDATE_POOL]
_last_screen_ts: float = 0.0
_last_prices: dict[str, float] = {}
_lock = asyncio.Lock()

# simulated paper broker (no external brokerage; fills modeled vs live prices)
broker = PaperBroker()

# per-symbol live daily-bar cache (in addition to history.py's disk cache)
_feat_cache: dict[str, tuple] = {}


def _live_feats(symbol: str):
    """(features_dict|None, data_sufficient, note) for one symbol, as of last close."""
    try:
        bars = _hist.daily(symbol)
    except Exception as e:  # noqa: BLE001 — no key / fetch error
        return None, False, f"no daily history ({type(e).__name__})"
    return live_features(symbol, bars)


def _build_rows(snaps):
    """Return list of (Scored, row_dict, Prediction|None)."""
    scored = [score_snap(s) for s in snaps]
    preds = {}
    if MODEL is not None:
        items = []
        for sc in scored:
            feats, ok, note = _live_feats(sc.symbol)
            items.append((sc.symbol, feats, ok, note))
        for p in MODEL.predict_batch(items):
            preds[p.symbol] = p
    out = []
    for sc in scored:
        row = sc.to_row()
        p = preds.get(sc.symbol)
        if p is not None:
            pred = p.to_dict()
        else:
            pred = {"available": False, "p_pop": None, "p_safe": None,
                    "data_sufficient": False, "contributions": [],
                    "note": "model unavailable"}
        pred.setdefault("safe_gate_passed", False)
        row["prediction"] = pred
        out.append((sc, row, p))
    return out


def _rank(rows, top: int, gate: bool = True):
    have_preds = any(r[2] and r[2].p_pop is not None for r in rows)
    if have_preds:
        elig = []
        for sc, row, p in rows:
            if not (p and p.p_pop is not None):
                continue
            passed = (p.p_safe or 0.0) >= SAFE_THRESHOLD
            row["prediction"]["safe_gate_passed"] = passed
            if gate and not passed:
                continue
            elig.append((sc, row, p))
        elig.sort(key=lambda r: r[2].p_pop, reverse=True)
        return elig[:top]
    # fallback: rank by the explainable composite
    rows = sorted(rows, key=lambda r: r[0].composite, reverse=True)
    return rows[:top]


async def _screen_loop():
    global _pool, _last_screen_ts
    while True:
        try:
            snaps = await asyncio.to_thread(provider.snapshot, _universe)
            rows = await asyncio.to_thread(_build_rows, snaps)
            pool = _rank(rows, CANDIDATE_POOL, gate=True)
            async with _lock:
                _pool = [sc.symbol for sc, _, _ in pool] or list(_universe)[:CANDIDATE_POOL]
                _last_screen_ts = time.time()
            print(f"[screen] scanned {len(_universe)} -> pool {len(_pool)}")
        except Exception as e:  # noqa: BLE001
            print(f"[screen] error: {e}")
        await asyncio.sleep(SCREEN_SECONDS)


async def _refresh_loop():
    while True:
        try:
            async with _lock:
                pool = list(_pool)
            # also snapshot any held paper positions that dropped out of the pool,
            # so the broker can mark them and fire stops/targets.
            held = [s for s in broker.positions if s not in pool]
            snaps = await asyncio.to_thread(provider.snapshot, pool + held)
            prices = {s.symbol: s.last for s in snaps}
            await asyncio.to_thread(broker.mark, prices)
            poolset = set(pool)
            rows = await asyncio.to_thread(_build_rows, [s for s in snaps if s.symbol in poolset])
            top = _rank(rows, TOP_N, gate=True)
            payload_rows = [row for _, row, _ in top]
            # smart-money OVERLAY (top-N only, cached ~12h): a labeled side-signal +
            # tiebreak. Never alters the calibrated probability.
            if altdata.enabled():
                for row in payload_rows:
                    sm = await asyncio.to_thread(altdata.congress_activity, row["symbol"])
                    row["smart_money"] = {"available": sm.get("available", False),
                                          "net": sm.get("net"), "buys": sm.get("buys"),
                                          "sells": sm.get("sells")}
                payload_rows.sort(key=lambda r: (-(r["prediction"].get("p_pop") or 0),
                                                 -((r.get("smart_money") or {}).get("net") or 0)))
            async with _lock:
                _latest["rows"] = payload_rows
                _latest["ts"] = time.time()
                _last_prices.update(prices)
        except Exception as e:  # noqa: BLE001
            print(f"[refresh] error: {e}")
        await asyncio.sleep(REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # kick one screen immediately so the first board isn't empty
    screen = asyncio.create_task(_screen_loop())
    refresh = asyncio.create_task(_refresh_loop())
    yield
    screen.cancel()
    refresh.cancel()


app = FastAPI(title="Tide — Predictive Momentum Leaderboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _leaderboard_payload() -> dict:
    return {
        "rows": _latest["rows"],
        "ts": _latest["ts"],
        "refresh": REFRESH_SECONDS,
        "last_screen_ts": _last_screen_ts,
        "safe_threshold": SAFE_THRESHOLD,
        "model_loaded": bool(MODEL),
        "top_n": TOP_N,
    }


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "provider": type(provider).__name__,
        "universe": len(_universe),
        "model_loaded": bool(MODEL),
        "model_version": MODEL.metadata.get("version") if MODEL else None,
        "last_screen_ts": _last_screen_ts,
    }


@app.get("/api/leaderboard")
async def get_leaderboard():
    async with _lock:
        return _leaderboard_payload()


@app.get("/api/calibration")
async def get_calibration():
    if MODEL is None:
        return {"available": False, "reason": "no model artifact loaded"}
    cal = dict(MODEL.metadata.get("calibration", {}))
    cal["available"] = True
    cal["model_version"] = MODEL.metadata.get("version")
    cal["trained_at"] = MODEL.metadata.get("trained_at")
    cal["synthetic"] = MODEL.metadata.get("synthetic", False)
    return cal


@app.get("/api/likelihood/{symbol}")
async def get_likelihood(symbol: str):
    """Empirical forward-return frequencies for one symbol, from real history.
    Unchanged — this is the honest ground-truth panel the model estimate is
    checked against."""
    return await asyncio.to_thread(compute_likelihood, symbol.upper())


def _analyst_payload(symbol: str) -> dict:
    """Assemble ONLY real, pre-computed stats for the analyst note."""
    feats, ok, note = _live_feats(symbol)
    prediction = None
    if MODEL is not None:
        p = MODEL.predict(symbol, feats, ok, note)
        prediction = {
            "p_pop_pct": None if p.p_pop is None else round(p.p_pop * 100, 1),
            "p_safe_pct": None if p.p_safe is None else round(p.p_safe * 100, 1),
            "data_sufficient": p.data_sufficient,
            "top_drivers": p.to_dict().get("contributions", []),
        }
    cal_meta = MODEL.metadata.get("calibration", {}) if MODEL else {}
    val = cal_meta.get("validation", {}).get("pop", {}) if cal_meta else {}
    calibration = {
        "model_version": MODEL.metadata.get("version") if MODEL else None,
        "trustworthy": cal_meta.get("trustworthy"),
        "base_rate_pct": None if val.get("base_rate") is None else round(val["base_rate"] * 100, 1),
        "brier_pop": val.get("brier"),
        "validation_n": val.get("n"),
        "synthetic": MODEL.metadata.get("synthetic") if MODEL else None,
    }
    try:
        empirical = compute_likelihood(symbol.upper())
    except Exception:  # noqa: BLE001
        empirical = {"n": 0}
    return {
        "prediction": prediction,
        "calibration": calibration,
        "empirical": {k: empirical.get(k) for k in
                      ("n", "freq_target", "median_fwd", "recover_rate", "target_pct", "horizon_days")},
        "news": recent_news(symbol),
    }


@app.get("/api/analyst/{symbol}")
async def get_analyst(symbol: str):
    sym = symbol.upper()
    payload = await asyncio.to_thread(_analyst_payload, sym)
    return await asyncio.to_thread(analyst.generate_note, sym, payload)


# ---------------------------------------------------------------------------
# Paper execution (SIMULATED — no external brokerage). Human-in-the-loop: the
# frontend "Arm paper trade" button is the only thing that opens a position.
# ---------------------------------------------------------------------------
def _price_for(symbol: str) -> float:
    p = _last_prices.get(symbol.upper())
    if p:
        return p
    try:
        snaps = provider.snapshot([symbol.upper()])
        return snaps[0].last if snaps else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


@app.post("/api/paper/arm")
async def paper_arm(body: dict):
    symbol = (body.get("symbol") or "").upper()
    if not symbol:
        return {"ok": False, "error": "symbol required"}
    price = await asyncio.to_thread(_price_for, symbol)
    return await asyncio.to_thread(
        lambda: broker.arm(symbol, price, notional=body.get("notional"),
                           stop_pct=body.get("stop_pct"), target_pct=body.get("target_pct"))
    )


@app.post("/api/paper/close/{symbol}")
async def paper_close(symbol: str):
    price = await asyncio.to_thread(_price_for, symbol)
    return await asyncio.to_thread(broker.close, symbol.upper(), price)


@app.get("/api/paper/portfolio")
async def paper_portfolio():
    return broker.state()


@app.get("/api/smartmoney/{symbol}")
async def get_smartmoney(symbol: str):
    """Full disclosed congressional trades for one symbol (raw facts panel)."""
    if not altdata.enabled():
        return {"available": False, "reason": "no QUIVER_API_KEY / FINNHUB_API_KEY set"}
    return await asyncio.to_thread(altdata.congress_activity, symbol.upper())


@app.post("/api/paper/reset")
async def paper_reset():
    return broker.reset()


@app.websocket("/ws/leaderboard")
async def ws_leaderboard(ws: WebSocket):
    await ws.accept()
    try:
        last_sent = -1.0
        while True:
            async with _lock:
                payload = _leaderboard_payload()
            if payload["ts"] != last_sent and payload["rows"]:
                await ws.send_text(json.dumps(payload))
                last_sent = payload["ts"]
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Serve the wired dashboard from the backend itself (so a single hosted process
# serves both the API and the real UI). Mounted LAST so /api/* and /ws/* above
# take precedence. html=True serves frontend/index.html at "/".
# ---------------------------------------------------------------------------
_frontend = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

