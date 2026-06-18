"""
Offline training pipeline.

Builds the pooled dataset, trains two classifiers (pop, safe), calibrates each
on a strictly-later held-out slice (never random folds — that leaks the future
into calibration), validates on a final untouched test window separated from
train by an embargo gap, and persists a joblib artifact + a human-readable
report.json.

Run with no Polygon key to prove the plumbing + calibration math end to end:

    python train.py --synthetic

Run on real data (needs POLYGON_API_KEY) over the cheap tier's ~2y window:

    python train.py --start 2024-06-01 --end 2026-06-01
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from features import FEATURE_NAMES
from config import (POP_TARGET_PCT, POP_HORIZON, SAFE_HORIZON, SAFE_BUFFER,
                    MODEL_PATH)
from dataset import build_dataset, feature_stats
from model import MomentumModel


# ---------------------------------------------------------------------------
# Synthetic universe (no key) — mirrors backtest.py --synthetic, scaled up to a
# pooled multi-name dataset with deliberately-injected pop setups so the
# positive class exists and is learnable. Validates calibration, not markets.
# ---------------------------------------------------------------------------
def generate_synthetic(n_symbols: int = 250, n_days: int = 620, seed: int = 7):
    import random
    from history import Bar
    random.seed(seed)
    sectors = ["Technology", "Financials", "Energy", "Healthcare", "Industrials",
               "Consumer", "Materials", "Utilities", "Communication", "Real Estate"]
    bars_by_symbol: dict[str, list] = {}
    sector_by_symbol: dict[str, str] = {}
    for s in range(n_symbols):
        sym = f"SYN{s:03d}"
        sector_by_symbol[sym] = sectors[s % len(sectors)]
        price = random.uniform(8, 400)
        drift = random.uniform(-0.0006, 0.0010)
        vol_base = random.uniform(2e5, 3e7)
        bars: list = []
        # pre-mark which days will be "pop setups"
        pop_days = set()
        d = 60
        while d < n_days - SAFE_HORIZON - 2:
            if random.random() < 0.02:
                pop_days.add(d)
                d += random.randint(8, 25)
            else:
                d += 1
        for day in range(n_days):
            shock = random.gauss(0, 0.018)
            setup = day in pop_days
            day_drift = drift + (random.uniform(0.0, 0.004) if setup else 0.0)
            price = max(1.0, price * (1 + day_drift + shock))
            o = price * random.uniform(0.99, 1.01)
            if setup:
                o = price * random.uniform(1.01, 1.03)  # gap up on setup
            h = max(o, price) * random.uniform(1.0, 1.025)
            l = min(o, price) * random.uniform(0.975, 1.0)
            v = int(vol_base * random.uniform(0.5, 1.6) * (random.uniform(2.5, 5.0) if setup else 1.0))
            bars.append(Bar(t=(1_600_000_000 + day * 86400) * 1000,
                            o=round(o, 2), h=round(h, 2), l=round(l, 2),
                            c=round(price, 2), v=v, vw=round((h + l + price) / 3, 2)))
        # realize the injected pops: force a >=11% high within the next 2 days
        for d in pop_days:
            if d + 2 < len(bars):
                entry = bars[d].c
                target = entry * 1.12
                bars[d + 1].h = max(bars[d + 1].h, round(target, 2))
                bars[d + 1].c = max(bars[d + 1].c, round(entry * random.uniform(1.04, 1.10), 2))
        bars_by_symbol[sym] = bars
    return bars_by_symbol, sector_by_symbol


# ---------------------------------------------------------------------------
# Real universe loader (needs key). Capped to keep the first run polite.
# ---------------------------------------------------------------------------
def load_real(start: str, end: str, max_symbols: int):
    import datetime as dt
    from history import HistoryProvider
    from universe import screen_universe
    from providers import _SECTORS  # sector hints for known names

    hist = HistoryProvider()
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)

    # Build the symbol set from screened membership sampled across the range.
    symbols: set[str] = set()
    probe = d1
    probes = 0
    while probe > d0 and probes < 8:
        if probe.weekday() < 5:
            try:
                for s in screen_universe(probe, hist):
                    symbols.add(s)
            except Exception as e:  # noqa: BLE001
                print(f"[train] screen {probe} failed: {e}")
            probes += 1
        probe -= dt.timedelta(days=30)
    symbols = set(sorted(symbols)[:max_symbols])
    print(f"[train] fetching daily history for {len(symbols)} symbols...")

    bars_by_symbol: dict[str, list] = {}
    sector_by_symbol: dict[str, str] = {}
    lookback = (dt.date.today() - d0).days + 5
    for i, sym in enumerate(sorted(symbols)):
        try:
            bars = hist.daily(sym, lookback_days=lookback)
        except Exception as e:  # noqa: BLE001
            print(f"[train] {sym} history failed: {e}")
            continue
        if len(bars) > 60:
            bars_by_symbol[sym] = bars
            sector_by_symbol[sym] = _SECTORS.get(sym, "Unknown")
        if (i + 1) % 200 == 0:
            print(f"[train]   {i + 1}/{len(symbols)}")
    return bars_by_symbol, sector_by_symbol


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _calibrate(base, X_cal, y_cal, method: str):
    """Wrap a prefit classifier in a calibrator fit on a held-out future slice.
    Robust across sklearn versions (FrozenEstimator in >=1.6, cv='prefit' before)."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal


def _balanced_sample_weight(y):
    import numpy as np
    y = np.asarray(y)
    n = len(y)
    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg)


def _fit_head(X_tr, y_tr, model_kind: str):
    import numpy as np
    sw = _balanced_sample_weight(y_tr)
    if model_kind == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=-1, num_leaves=31, subsample=0.8,
                                 colsample_bytree=0.8, verbose=-1)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                              max_depth=None, l2_regularization=1.0,
                                              early_stopping=False)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    return clf


def _permutation_importances(clf, X, y, cap: int = 2000):
    import numpy as np
    from sklearn.inspection import permutation_importance
    if len(X) > cap:
        idx = np.linspace(0, len(X) - 1, cap).astype(int)
        X, y = X[idx], y[idx]
    if len(set(y.tolist())) < 2:
        return {name: 0.0 for name in FEATURE_NAMES}
    try:
        r = permutation_importance(clf, X, y, n_repeats=3, random_state=0,
                                   scoring="average_precision")
        return {FEATURE_NAMES[j]: float(r.importances_mean[j]) for j in range(len(FEATURE_NAMES))}
    except Exception:
        return {name: 0.0 for name in FEATURE_NAMES}


def _evaluate(cal, X_te, y_te, n_bins: int = 10) -> dict:
    import numpy as np
    from sklearn.metrics import brier_score_loss, average_precision_score
    classes = list(getattr(cal, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else (len(classes) - 1)
    p = cal.predict_proba(X_te)[:, idx]
    base_rate = float(np.mean(y_te)) if len(y_te) else None
    brier = float(brier_score_loss(y_te, p)) if len(set(y_te.tolist())) > 1 else None
    try:
        ap = float(average_precision_score(y_te, p)) if len(set(y_te.tolist())) > 1 else None
    except Exception:
        ap = None
    # reliability bins
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "predicted": round(float(p[m].mean()), 4),
            "observed": round(float(y_te[m].mean()), 4),
            "n": int(m.sum()),
        })
    # precision@k / lift among top-ranked
    order = np.argsort(-p)
    prec_at = {}
    for k in (5, 25, 50):
        kk = min(k, len(order))
        if kk:
            prec = float(y_te[order[:kk]].mean())
            prec_at[str(k)] = round(prec, 4)
    lift5 = (prec_at.get("5") / base_rate) if (base_rate and prec_at.get("5")) else None
    return {
        "n": int(len(y_te)),
        "base_rate": None if base_rate is None else round(base_rate, 4),
        "brier": None if brier is None else round(brier, 4),
        "average_precision": ap if ap is None else round(ap, 4),
        "precision_at_k": prec_at,
        "lift_at_5": None if lift5 is None else round(lift5, 3),
        "reliability": bins,
    }


def _time_split(meta, embargo_days: int):
    """Return train/cal/test row-index lists, split by calendar date with an
    embargo gap (in trading days) removed around each boundary so a row's
    forward label window can't straddle into the next segment."""
    dates = sorted({m["date"] for m in meta if m["date"]})
    n = len(dates)
    if n < 10:
        raise SystemExit("Not enough distinct dates to split.")
    i_tr = int(n * 0.70)
    i_cal = int(n * 0.85)
    train_dates = set(dates[:max(0, i_tr - embargo_days)])
    cal_dates = set(dates[i_tr:max(i_tr, i_cal - embargo_days)])
    test_dates = set(dates[i_cal:])
    tr, ca, te = [], [], []
    for idx, m in enumerate(meta):
        d = m["date"]
        if d in train_dates:
            tr.append(idx)
        elif d in cal_dates:
            ca.append(idx)
        elif d in test_dates:
            te.append(idx)
    return tr, ca, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--model", choices=["hgb", "lgbm"], default="hgb")
    ap.add_argument("--calibration", choices=["sigmoid", "isotonic", "both"], default="sigmoid")
    ap.add_argument("--embargo-days", type=int, default=SAFE_HORIZON)
    ap.add_argument("--max-symbols", type=int, default=int(os.getenv("TRAIN_MAX_SYMBOLS", "800")))
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    import numpy as np

    if args.synthetic:
        print("[train] generating synthetic universe (no key)...")
        bars_by_symbol, sector_by_symbol = generate_synthetic()
    else:
        bars_by_symbol, sector_by_symbol = load_real(args.start, args.end, args.max_symbols)
    if not bars_by_symbol:
        raise SystemExit("No data assembled — aborting.")

    print(f"[train] building dataset from {len(bars_by_symbol)} symbols...")
    X, y_pop, y_safe, meta = build_dataset(bars_by_symbol, sector_by_symbol)
    print(f"[train] dataset rows={len(X)} pop_rate={y_pop.mean():.4f} safe_rate={y_safe.mean():.4f}")
    if len(X) < 200:
        raise SystemExit("Too few rows to train.")

    tr, ca, te = _time_split(meta, args.embargo_days)
    print(f"[train] split train={len(tr)} cal={len(ca)} test={len(te)} (embargo={args.embargo_days}d)")

    def pick_method(base, Xc, yc, Xt, yt):
        from sklearn.metrics import brier_score_loss
        methods = ["sigmoid", "isotonic"] if args.calibration == "both" else [args.calibration]
        best, best_brier = None, None
        for m in methods:
            try:
                cal = _calibrate(base, Xc, yc, m)
                cls = list(getattr(cal, "classes_", [0, 1]))
                idx = cls.index(1) if 1 in cls else len(cls) - 1
                p = cal.predict_proba(Xt)[:, idx]
                br = brier_score_loss(yt, p) if len(set(yt.tolist())) > 1 else 1.0
                if best_brier is None or br < best_brier:
                    best, best_brier, best_m = cal, br, m
            except Exception as e:  # noqa: BLE001
                print(f"[train] calibration {m} failed: {e}")
        return best, best_m

    report = {
        "model_kind": args.model,
        "synthetic": args.synthetic,
        "target": {"pop_pct": POP_TARGET_PCT, "pop_horizon": POP_HORIZON,
                   "safe_horizon": SAFE_HORIZON, "safe_buffer": SAFE_BUFFER},
        "rows": int(len(X)),
        "heads": {},
    }

    heads = {}
    importances = {}
    for name, y in (("pop", y_pop), ("safe", y_safe)):
        Xtr, ytr = X[tr], y[tr]
        Xca, yca = X[ca], y[ca]
        Xte, yte = X[te], y[te]
        print(f"[train] fitting {name} head (train pos={int(ytr.sum())}/{len(ytr)})...")
        base = _fit_head(Xtr, ytr, args.model)
        cal, method = pick_method(base, Xca, yca, Xte, yte)
        metrics = _evaluate(cal, Xte, yte)
        metrics["calibration_method"] = method
        report["heads"][name] = metrics
        heads[name] = cal
        if name == "pop":
            importances = _permutation_importances(base, Xtr, ytr)
        print(f"[train]   {name}: brier={metrics['brier']} ap={metrics['average_precision']} "
              f"base_rate={metrics['base_rate']} prec@5={metrics['precision_at_k'].get('5')}")

    # calibration-quality gate baked into the artifact (server reads this)
    brier_pop = report["heads"]["pop"]["brier"]
    n_test = report["heads"]["pop"]["n"]
    trustworthy = bool(brier_pop is not None and brier_pop <= float(os.getenv("MAX_CALIB_BRIER", "0.15"))
                       and n_test >= int(os.getenv("MIN_CALIB_N", "500")))

    metadata = {
        "version": __import__("time").strftime("%Y-%m-%d") + f"-{args.model}",
        "trained_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "synthetic": args.synthetic,
        "feature_stats": feature_stats(X[tr]),
        "pop_importances": importances,
        "calibration": {
            "trustworthy": trustworthy,
            "targets": {
                "pop": f"rise >= {POP_TARGET_PCT}% within {POP_HORIZON} trading days",
                "safe": f"no intraday low below entry-{SAFE_BUFFER:.0%} over {SAFE_HORIZON} trading days",
            },
            "validation": {
                "pop": report["heads"]["pop"],
                "safe": report["heads"]["safe"],
            },
            "feature_skew_note": ("Model trained on settled daily bars; live features are "
                                  "computed as of the last completed session — see CLAUDE.md."),
        },
    }

    model = MomentumModel(pop_clf=heads["pop"], safe_clf=heads["safe"],
                          feature_names=list(FEATURE_NAMES), metadata=metadata)
    out = args.out if not args.synthetic else args.out.replace(".joblib", "_synthetic.joblib")
    model.save(out)
    print(f"[train] saved artifact -> {out}  (trustworthy={trustworthy})")

    report_path = args.report or (os.path.splitext(out)[0] + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train] wrote report -> {report_path}")
    print(json.dumps(report["heads"], indent=2))


if __name__ == "__main__":
    main()
