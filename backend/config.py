"""
Shared constants — the single source of truth for label/target definitions.

Both the honest empirical engine (backtest.py) and the predictive model's
labels (labels.py) MUST read these from here. If the "+10% in 1-2 days" target
ever drifts between the two, the model's calibration would be measured against
a different question than the empirical panel reports — a silent correctness
bug. Keeping the numbers in one place prevents that.
"""
from __future__ import annotations
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Loads KEY=VALUE lines from the
    backend/.env next to this file into os.environ without overriding values
    already set in the real environment. uvicorn does not load .env on its own."""
    path = Path(__file__).with_name(".env")
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_load_dotenv()

# --- The "pop" event we are measuring / predicting ---------------------------
POP_TARGET_PCT = float(os.getenv("POP_TARGET_PCT", "10.0"))  # the move size, %
POP_HORIZON = int(os.getenv("POP_HORIZON", "2"))             # within N trading days

# --- The downside-safety question --------------------------------------------
SAFE_HORIZON = int(os.getenv("SAFE_HORIZON", "15"))          # ~3 trading weeks
# A literal zero-tolerance "never traded below entry" label is extremely strict
# and rare. SAFE_BUFFER allows a small drawdown tolerance (fraction, e.g. 0.01
# = 1%). The intent ("won't go lower than current price") is well served by a
# small buffer; train.py reports the base rate at both 0 and the buffer so the
# choice is made knowingly.
SAFE_BUFFER = float(os.getenv("SAFE_BUFFER", "0.05"))

# --- Artifact / cache locations ----------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "./.model/model.joblib")

# Backwards-compatible aliases for the names backtest.py used inline before the
# refactor. Keep both so existing imports/readers stay valid.
TARGET_PCT = POP_TARGET_PCT
HORIZON_DAYS = POP_HORIZON
RECOVER_DAYS = SAFE_HORIZON
