"""
Model artifact: the seam between the offline training pipeline and live serving.

`train.py` builds a `MomentumModel` (two calibrated classifiers + metadata) and
persists it with joblib. The server loads it via `MomentumModel.load(path)` and
calls `predict_batch`. The server never imports training code — only this.

Degrades gracefully: if the artifact is missing/unreadable, `load` returns None
and the server runs in "no-model" mode (the composite activity leaderboard still
works; probabilities show as unavailable). This preserves the no-credentials dev
path the project started with.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from features import FEATURE_NAMES


@dataclass
class Prediction:
    symbol: str
    p_pop: float | None          # calibrated P(+TARGET% within HORIZON days), 0..1
    p_safe: float | None         # calibrated P(no breach of entry over SAFE_HORIZON), 0..1
    contributions: dict          # feature_name -> signed contribution to p_pop
    feature_vector: dict         # the live features actually fed to the model
    data_sufficient: bool        # were features computable from this snapshot?
    note: str = ""
    potential_growth_pct: float | None = None  # est. favorable 48h move (volatility-based)

    def to_dict(self, top_contrib: int = 3) -> dict:
        contrib = sorted(
            self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:top_contrib]
        return {
            "available": self.data_sufficient and self.p_pop is not None,
            "p_pop": None if self.p_pop is None else round(self.p_pop, 4),
            "p_safe": None if self.p_safe is None else round(self.p_safe, 4),
            "potential_growth_pct": self.potential_growth_pct,
            "data_sufficient": self.data_sufficient,
            "contributions": [{"feature": k, "value": round(v, 4)} for k, v in contrib],
            "note": self.note,
        }


@dataclass
class MomentumModel:
    pop_clf: object              # calibrated classifier for the pop target
    safe_clf: object             # calibrated classifier for the safe target
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)

    # ----- inference -----
    def _vectorize(self, feats: dict):
        import numpy as np
        return np.array([[feats.get(name, np.nan) for name in self.feature_names]], dtype=float)

    def _proba(self, clf, X):
        import numpy as np
        try:
            p = clf.predict_proba(X)
            # column for class "1"
            classes = list(getattr(clf, "classes_", [0, 1]))
            idx = classes.index(1) if 1 in classes else (len(classes) - 1)
            return float(p[0][idx])
        except Exception:
            return None

    def predict(self, symbol: str, feats: dict | None, data_sufficient: bool,
                note: str = "") -> Prediction:
        if not data_sufficient or feats is None:
            return Prediction(symbol, None, None, {}, feats or {}, False, note or "insufficient data")
        X = self._vectorize(feats)
        p_pop = self._proba(self.pop_clf, X)
        p_safe = self._proba(self.safe_clf, X)
        contrib = self._contributions(feats, p_pop)
        # "potential growth over the next ~48h" = expected favorable 2-day move
        # from recent volatility: daily stdev (%) scaled to 2 trading days (×√2).
        rv = feats.get("realized_vol_20d")
        pg = rv * (2 ** 0.5) if (rv and rv == rv and rv > 0) else None
        # guard against data artifacts (unadjusted splits / illiquid prints) that
        # produce absurd volatility — a >35% expected 2-day move isn't real signal.
        pg = round(pg, 1) if (pg is not None and pg <= 35) else None
        return Prediction(symbol, p_pop, p_safe, contrib, feats, True, note,
                          potential_growth_pct=pg)

    def predict_batch(self, items: list[tuple]) -> list[Prediction]:
        """items: list of (symbol, feats|None, data_sufficient, note)."""
        return [self.predict(*it) for it in items]

    def _contributions(self, feats: dict, p_pop: float | None) -> dict:
        """Lightweight per-feature attribution for the pop head. Uses the base
        estimator's feature_importances_ weighted by the feature's signed
        z-deviation from the training mean, if available; otherwise empty.
        This is for UI explainability ("why this pick"), not a formal SHAP."""
        import numpy as np
        stats = self.metadata.get("feature_stats") or {}
        importances = self.metadata.get("pop_importances") or {}
        if not stats or not importances:
            return {}
        out = {}
        for name in self.feature_names:
            mean = stats.get(name, {}).get("mean")
            std = stats.get(name, {}).get("std")
            imp = importances.get(name)
            val = feats.get(name)
            if mean is None or std in (None, 0) or imp is None or val is None or np.isnan(val):
                continue
            z = (val - mean) / std
            out[name] = float(imp * z)
        return out

    # ----- persistence -----
    def save(self, path: str) -> None:
        import os
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MomentumModel | None":
        import os
        if not path or not os.path.exists(path):
            return None
        try:
            import joblib
            m = joblib.load(path)
        except Exception as e:  # noqa: BLE001
            print(f"[model] failed to load artifact at {path}: {e}")
            return None
        # Fail loud on feature-name drift between artifact and code.
        if list(getattr(m, "feature_names", [])) != FEATURE_NAMES:
            print("[model] WARNING: artifact feature_names differ from features.FEATURE_NAMES; "
                  "refusing to serve a misaligned model.")
            return None
        return m
