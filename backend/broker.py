"""
Execution layer — an internal SIMULATED paper broker.

Why simulated (not Alpaca/IBKR): the user is not a US taxholder and can't open a
US brokerage. So fills are modeled here against live prices from the data
provider, with an explicit slippage assumption, and the whole thing lives behind
a `Broker` protocol — a real `AlpacaBroker` could drop in later without touching
the server. This keeps the citizenship blocker from killing the execution/P&L
layer and keeps fills under our control for an honesty-first tool.

Honesty: everything here is labeled SIMULATED in the UI. Fills are at the last
trade price ± `PAPER_SLIPPAGE_BPS`, which is a model of a real fill, not a real
fill. Arming a pick places exactly the bet the model described (+TARGET% target,
−STOP% stop), so the paper P&L is the honest forward-validation of the signal.
"""
from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Protocol

from config import POP_TARGET_PCT

PAPER_DIR = Path(os.getenv("PAPER_DIR", "./.paper"))
PAPER_CASH = float(os.getenv("PAPER_CASH", "100000"))
PAPER_SLIPPAGE_BPS = float(os.getenv("PAPER_SLIPPAGE_BPS", "5"))
DEFAULT_STOP_PCT = float(os.getenv("STOP_PCT", "0.05"))
DEFAULT_TARGET_PCT = POP_TARGET_PCT / 100.0  # the +10% the model predicted
DEFAULT_NOTIONAL = float(os.getenv("PAPER_NOTIONAL", "10000"))
_EQUITY_MIN_INTERVAL = 30.0  # seconds between equity-curve points
_EQUITY_CAP = 1500


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: float
    stop_price: float
    target_price: float
    last: float = 0.0  # live mark, not persisted-authoritative


@dataclass
class Trade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: float
    exit_ts: float
    pnl: float
    pnl_pct: float
    reason: str  # "target" | "stop" | "manual"


@dataclass
class EquityPoint:
    ts: float
    equity: float


class Broker(Protocol):
    def arm(self, symbol: str, price: float, *, notional: float | None = None,
            stop_pct: float | None = None, target_pct: float | None = None) -> dict: ...
    def close(self, symbol: str, price: float, reason: str = "manual") -> dict: ...
    def mark(self, prices: dict[str, float]) -> None: ...
    def state(self) -> dict: ...
    def reset(self) -> dict: ...


class PaperBroker:
    """Simulated broker persisted to PAPER_DIR/state.json."""

    def __init__(self) -> None:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        self.path = PAPER_DIR / "state.json"
        self.slip = PAPER_SLIPPAGE_BPS / 10000.0
        self.cash = PAPER_CASH
        self.positions: dict[str, Position] = {}
        self.closed: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self._last_equity_ts = 0.0
        self._load()

    # ----- persistence -----
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self.cash = raw.get("cash", PAPER_CASH)
            self.positions = {k: Position(**v) for k, v in raw.get("positions", {}).items()}
            self.closed = [Trade(**t) for t in raw.get("closed", [])]
            self.equity_curve = [EquityPoint(**e) for e in raw.get("equity_curve", [])]
        except Exception as e:  # noqa: BLE001
            print(f"[broker] failed to load state: {e}")

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "cash": self.cash,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "closed": [asdict(t) for t in self.closed],
                "equity_curve": [asdict(e) for e in self.equity_curve[-_EQUITY_CAP:]],
            }))
        except Exception as e:  # noqa: BLE001
            print(f"[broker] failed to save state: {e}")

    # ----- actions -----
    def arm(self, symbol: str, price: float, *, notional: float | None = None,
            stop_pct: float | None = None, target_pct: float | None = None) -> dict:
        symbol = symbol.upper()
        if price <= 0:
            return {"ok": False, "error": "no live price for symbol"}
        if symbol in self.positions:
            return {"ok": False, "error": f"already holding {symbol}"}
        notional = float(notional or DEFAULT_NOTIONAL)
        stop_pct = DEFAULT_STOP_PCT if stop_pct is None else float(stop_pct)
        target_pct = DEFAULT_TARGET_PCT if target_pct is None else float(target_pct)
        entry = price * (1 + self.slip)  # buy fills slightly worse than last
        if notional > self.cash:
            return {"ok": False, "error": "insufficient paper cash"}
        qty = notional / entry
        pos = Position(
            symbol=symbol, qty=qty, entry_price=entry, entry_ts=time.time(),
            stop_price=entry * (1 - stop_pct), target_price=entry * (1 + target_pct),
            last=price,
        )
        self.cash -= qty * entry
        self.positions[symbol] = pos
        self._save()
        return {"ok": True, "position": asdict(pos), "slippage_bps": PAPER_SLIPPAGE_BPS}

    def _exit(self, symbol: str, fill: float, reason: str) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * fill
        self.cash += proceeds
        pnl = proceeds - pos.qty * pos.entry_price
        pnl_pct = (fill - pos.entry_price) / pos.entry_price * 100
        self.closed.append(Trade(
            symbol=symbol, qty=pos.qty, entry_price=pos.entry_price, exit_price=fill,
            entry_ts=pos.entry_ts, exit_ts=time.time(), pnl=pnl, pnl_pct=pnl_pct,
            reason=reason,
        ))

    def close(self, symbol: str, price: float, reason: str = "manual") -> dict:
        symbol = symbol.upper()
        if symbol not in self.positions:
            return {"ok": False, "error": f"no open position in {symbol}"}
        if price <= 0:
            return {"ok": False, "error": "no live price"}
        fill = price * (1 - self.slip)  # sell fills slightly worse than last
        self._exit(symbol, fill, reason)
        self._save()
        return {"ok": True}

    def mark(self, prices: dict[str, float]) -> None:
        """Update live marks and fire stop/target exits. Called each refresh tick."""
        changed = False
        for symbol in list(self.positions):
            last = prices.get(symbol)
            if not last or last <= 0:
                continue
            pos = self.positions[symbol]
            pos.last = last
            if last >= pos.target_price:
                self._exit(symbol, pos.target_price, "target")  # limit fills at target
                changed = True
            elif last <= pos.stop_price:
                self._exit(symbol, pos.stop_price * (1 - self.slip), "stop")  # stop slips
                changed = True
        # equity-curve point (throttled)
        now = time.time()
        if now - self._last_equity_ts >= _EQUITY_MIN_INTERVAL or changed:
            self.equity_curve.append(EquityPoint(ts=now, equity=self._equity()))
            self.equity_curve = self.equity_curve[-_EQUITY_CAP:]
            self._last_equity_ts = now
            changed = True
        if changed:
            self._save()

    def reset(self) -> dict:
        self.cash = PAPER_CASH
        self.positions = {}
        self.closed = []
        self.equity_curve = []
        self._last_equity_ts = 0.0
        self._save()
        return {"ok": True}

    # ----- reporting -----
    def _equity(self) -> float:
        return self.cash + sum(p.qty * (p.last or p.entry_price) for p in self.positions.values())

    def state(self) -> dict:
        unrealized = sum(((p.last or p.entry_price) - p.entry_price) * p.qty
                         for p in self.positions.values())
        realized = sum(t.pnl for t in self.closed)
        equity = self._equity()
        wins = sum(1 for t in self.closed if t.pnl > 0)
        n_closed = len(self.closed)
        open_rows = []
        for p in self.positions.values():
            last = p.last or p.entry_price
            open_rows.append({
                "symbol": p.symbol,
                "qty": round(p.qty, 4),
                "entry_price": round(p.entry_price, 2),
                "last": round(last, 2),
                "stop_price": round(p.stop_price, 2),
                "target_price": round(p.target_price, 2),
                "unrealized_pnl": round((last - p.entry_price) * p.qty, 2),
                "unrealized_pct": round((last - p.entry_price) / p.entry_price * 100, 2),
                "dist_to_stop_pct": round((last - p.stop_price) / last * 100, 2),
                "dist_to_target_pct": round((p.target_price - last) / last * 100, 2),
            })
        return {
            "simulated": True,
            "starting_cash": PAPER_CASH,
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "return_pct": round((equity - PAPER_CASH) / PAPER_CASH * 100, 2),
            "win_rate": round(wins / n_closed * 100, 1) if n_closed else None,
            "n_closed": n_closed,
            "slippage_bps": PAPER_SLIPPAGE_BPS,
            "default_target_pct": round(DEFAULT_TARGET_PCT * 100, 1),
            "default_stop_pct": round(DEFAULT_STOP_PCT * 100, 1),
            "default_notional": DEFAULT_NOTIONAL,
            "open_positions": open_rows,
            "closed_trades": [
                {**asdict(t), "pnl": round(t.pnl, 2), "pnl_pct": round(t.pnl_pct, 2),
                 "entry_price": round(t.entry_price, 2), "exit_price": round(t.exit_price, 2)}
                for t in self.closed[-50:]
            ],
            "equity_curve": [round(e.equity, 2) for e in self.equity_curve[-200:]],
        }
