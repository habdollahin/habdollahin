"""Experiment driver: hold-out split + nested cross-validation over every
(feature selector x classifier) combination.

For each combination:
  outer CV (repeated)  -> unbiased out-of-fold (OOF) probabilities on the training set
    inner CV           -> hyper-parameter search (grid / random / halving / optuna)
  refit on full train  -> final model -> held-out test probabilities
The operating threshold is chosen on OOF predictions and applied unchanged to the test set.
Combinations are ranked by OOF AUC (never by test AUC).
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .metrics import bootstrap_metrics, choose_threshold, delong_test
from .models import build_pipeline, make_search

log = logging.getLogger(__name__)


def _cv(n_splits, seed, groups):
    return (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed) if groups is not None
            else StratifiedKFold(n_splits, shuffle=True, random_state=seed))


def _proba(est, X, multi):
    P = est.predict_proba(X)
    return P if multi else P[:, 1]


def unwrap(est):
    """Return the underlying (fitted) Pipeline from a search / calibration wrapper."""
    if hasattr(est, "best_estimator_"):
        est = est.best_estimator_
    if isinstance(est, CalibratedClassifierCV):
        est = est.calibrated_classifiers_[0].estimator
    return est


def selected_features(pipe, names) -> list[str]:
    names = np.asarray(names)
    for step in ("constant", "decorrelate", "selector"):
        st = pipe.named_steps.get(step)
        if st is None or st == "passthrough" or not hasattr(st, "support_"):
            continue
        names = names[st.support_]
    return list(names)


def split_data(data: dict, cfg: dict):
    X, y, groups = data["X"], data["y"], data["groups"]
    n = len(y)
    if data.get("split") is not None:
        te = data["split"] == "test"
    elif cfg["modeling"].get("test_size", 0) > 0:
        idx = np.arange(n)
        ts = cfg["modeling"]["test_size"]
        if groups is not None:
            sgk = StratifiedGroupKFold(int(round(1 / ts)), shuffle=True, random_state=cfg["seed"])
            _, te_idx = next(sgk.split(X, y, groups))
        else:
            _, te_idx = train_test_split(idx, test_size=ts, stratify=y, random_state=cfg["seed"])
        te = np.isin(idx, te_idx)
    else:
        te = np.zeros(n, bool)
    return ~te, te


def run_combo(selector, model, Xtr, ytr, gtr, Xte, yte, cfg, multi, names) -> dict:
    m = cfg["modeling"]
    seed = cfg["seed"]
    scoring = m.get("scoring", "roc_auc")
    if multi and scoring == "roc_auc":
        scoring = "roc_auc_ovr"
    t0 = time.time()
    n_cls = len(np.unique(ytr))
    oof = np.zeros((len(ytr), n_cls)) if multi else np.zeros(len(ytr))
    counts = np.zeros(len(ytr))
    fold_scores, params, feats = [], [], []
    for r in range(int(m.get("outer_repeats", 1))):
        outer = _cv(int(m["outer_folds"]), seed + r, gtr)
        for f, (tr, va) in enumerate(outer.split(Xtr, ytr, gtr)):
            pipe, space = build_pipeline(selector, model, cfg, ytr[tr])
            inner = _cv(int(m["inner_folds"]), seed + 100 + r * 10 + f, gtr)
            search = make_search(pipe, space, cfg, inner, scoring, len(tr))
            fit_kw = {"groups": gtr[tr]} if gtr is not None else {}
            search.fit(Xtr[tr], ytr[tr], **fit_kw)
            est = search.best_estimator_
            if m.get("calibration"):
                est = CalibratedClassifierCV(clone(est), method=m["calibration"], cv=int(m["inner_folds"]))
                est.fit(Xtr[tr], ytr[tr])
            p = _proba(est, Xtr[va], multi)
            oof[va] += p
            counts[va] += 1
            try:
                fold_scores.append(roc_auc_score(ytr[va], p, multi_class="ovr") if multi
                                   else roc_auc_score(ytr[va], p))
            except ValueError:
                fold_scores.append(np.nan)
            params.append({k: (str(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()})
            feats.append(selected_features(unwrap(search), names))
    oof = oof / counts[:, None] if multi else oof / counts

    # final model on the whole training set
    pipe, space = build_pipeline(selector, model, cfg, ytr)
    search = make_search(pipe, space, cfg, _cv(int(m["inner_folds"]), seed, gtr), scoring, len(ytr))
    search.fit(Xtr, ytr, **({"groups": gtr} if gtr is not None else {}))
    final = search.best_estimator_
    if m.get("calibration"):
        final = CalibratedClassifierCV(clone(final), method=m["calibration"], cv=int(m["inner_folds"])).fit(Xtr, ytr)
    p_te = _proba(final, Xte, multi) if len(yte) else None
    return {
        "selector": selector, "model": model, "name": f"{selector}+{model}",
        "oof": oof, "test": p_te, "fold_auc": fold_scores, "fold_params": params, "fold_features": feats,
        "final": final, "final_params": search.best_params_,
        "final_features": selected_features(unwrap(final), names),
        "inner_score": float(search.best_score_), "seconds": time.time() - t0,
    }


def _safe_combo(*a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return run_combo(*a, **kw)
        except Exception as e:  # one failing combo must not kill the experiment
            import traceback

            log.error("combo %s+%s failed: %s", a[0], a[1], e)
            return {"selector": a[0], "model": a[1], "name": f"{a[0]}+{a[1]}", "error": str(e),
                    "trace": traceback.format_exc()}


def apply_robustness_filter(X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rob = cfg["features"].get("robustness", {})
    icc_csv = Path(cfg["output_dir"]) / "features" / "robustness_icc.csv"
    if not rob.get("enabled") or not icc_csv.exists():
        return X
    icc = pd.read_csv(icc_csv, index_col=0)["icc"]
    keep = [c for c in X.columns if c.startswith("clinical_") or icc.get(c, 0) >= rob.get("icc_threshold", 0.75)]
    log.info("robustness filter: kept %d/%d features (ICC >= %.2f)", len(keep), X.shape[1], rob.get("icc_threshold", 0.75))
    return X[keep]


def run_experiment(data: dict, cfg: dict) -> dict:
    from joblib import Parallel, delayed

    m = cfg["modeling"]
    X = apply_robustness_filter(data["X"], cfg)
    X = X.loc[:, X.notna().mean() > 0.5]  # drop features missing in > 50% of patients
    names = list(X.columns)
    y = np.asarray(data["y"])
    multi = len(np.unique(y)) > 2
    tr, te = split_data(data, cfg)
    Xv = X.to_numpy(float)
    g = data["groups"]
    gtr = g[tr] if g is not None else None
    log.info("train n=%d (classes %s), test n=%d, features=%d, multiclass=%s", tr.sum(),
             np.bincount(y[tr]).tolist(), te.sum(), len(names), multi)

    combos = [(s, mo) for s in m["selectors"] for mo in m["models"]]
    log.info("running %d selector x model combinations with nested CV (%dx%d outer, %d inner, search=%s)",
             len(combos), m.get("outer_repeats", 1), m["outer_folds"], m["inner_folds"], m["search"])
    res = Parallel(n_jobs=int(cfg.get("n_jobs", 1)), verbose=0)(
        delayed(_safe_combo)(s, mo, Xv[tr], y[tr], gtr, Xv[te], y[te], cfg, multi, names) for s, mo in combos)
    failed = [r for r in res if "error" in r]
    res = [r for r in res if "error" not in r]
    if not res:
        raise RuntimeError("all combinations failed: " + "; ".join(f"{r['name']}: {r['error']}" for r in failed))

    # threshold from OOF predictions + metrics
    nb = int(m.get("n_bootstrap", 1000))
    for r in res:
        r["threshold"] = None if multi else choose_threshold(y[tr], r["oof"], m.get("threshold", "youden"))
        r["metrics_cv"] = bootstrap_metrics(y[tr], r["oof"], r["threshold"], nb, cfg["seed"])
        r["metrics_test"] = (bootstrap_metrics(y[te], r["test"], r["threshold"], nb, cfg["seed"])
                             if te.any() else None)
    res.sort(key=lambda r: -r["metrics_cv"].loc["AUC", "value"])

    # soft-voting ensemble of the top-N combinations (by OOF AUC)
    top_n = int(m.get("ensemble_top_n", 0) or 0)
    if top_n >= 2 and len(res) >= 2:
        top = res[:top_n]
        ens = {"selector": "ensemble", "model": f"top{len(top)}", "name": f"ensemble_top{len(top)}",
               "oof": np.mean([r["oof"] for r in top], 0),
               "test": np.mean([r["test"] for r in top], 0) if te.any() else None,
               "fold_auc": [], "fold_params": [], "fold_features": [],
               "final": [r["final"] for r in top], "members": [r["name"] for r in top],
               "final_params": {"members": [r["name"] for r in top]},
               "final_features": sorted({f for r in top for f in r["final_features"]}),
               "inner_score": np.nan, "seconds": sum(r["seconds"] for r in top)}
        ens["threshold"] = None if multi else choose_threshold(y[tr], ens["oof"], m.get("threshold", "youden"))
        ens["metrics_cv"] = bootstrap_metrics(y[tr], ens["oof"], ens["threshold"], nb, cfg["seed"])
        ens["metrics_test"] = (bootstrap_metrics(y[te], ens["test"], ens["threshold"], nb, cfg["seed"])
                               if te.any() else None)
        res.append(ens)
        res.sort(key=lambda r: -r["metrics_cv"].loc["AUC", "value"])

    exp = {"results": res, "failed": failed, "X": X, "y": y, "train": tr, "test": te, "names": names,
           "classes": data["classes"], "ids": np.asarray(data["ids"]), "multi": multi}
    save_experiment(exp, cfg)
    return exp


def summary_table(exp: dict, which: str = "cv") -> pd.DataFrame:
    rows = []
    for r in exp["results"]:
        mt = r[f"metrics_{which}"]
        if mt is None:
            continue
        row = {"combination": r["name"], "selector": r["selector"], "model": r["model"]}
        for k in mt.index:
            row[k] = mt.loc[k, "value"]
            if not np.isnan(mt.loc[k, "ci_low"]):
                row[f"{k}_CI"] = f"[{mt.loc[k, 'ci_low']:.3f}, {mt.loc[k, 'ci_high']:.3f}]"
        if which == "cv" and r["fold_auc"]:
            row["fold_AUC_mean"] = np.nanmean(r["fold_auc"])
            row["fold_AUC_sd"] = np.nanstd(r["fold_auc"])
        row["n_features"] = len(r["final_features"])
        row["seconds"] = r["seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def delong_matrix(exp: dict, which: str = "test") -> pd.DataFrame | None:
    if exp["multi"]:
        return None
    mask = exp["test"] if which == "test" else exp["train"]
    if not mask.any():
        return None
    y = exp["y"][mask]
    res = exp["results"]
    key = "test" if which == "test" else "oof"
    names = [r["name"] for r in res]
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(res):
        for j, b in enumerate(res):
            if i < j:
                _, p = delong_test(y, a[key], b[key])
                M.iloc[i, j] = M.iloc[j, i] = p
    return M


def feature_stability(exp: dict) -> pd.DataFrame:
    """How often each feature was selected across outer folds (per combination)."""
    rows = []
    for r in exp["results"]:
        if not r["fold_features"]:
            continue
        nf = len(r["fold_features"])
        cnt = pd.Series([f for fs in r["fold_features"] for f in fs]).value_counts()
        for f, c in cnt.items():
            rows.append({"combination": r["name"], "feature": f, "frequency": c / nf})
    return pd.DataFrame(rows)


def save_experiment(exp: dict, cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    ids, y, tr, te = exp["ids"], exp["y"], exp["train"], exp["test"]

    cv_tab = summary_table(exp, "cv")
    te_tab = summary_table(exp, "test")
    stab = feature_stability(exp)
    with pd.ExcelWriter(out / "results.xlsx") as xw:
        cv_tab.to_excel(xw, sheet_name="nested_cv_train", index=False)
        if len(te_tab):
            te_tab.to_excel(xw, sheet_name="test", index=False)
        params = pd.DataFrame([{"combination": r["name"], "final_params": str(r["final_params"]),
                                "threshold": r["threshold"], "features": ", ".join(r["final_features"])}
                               for r in exp["results"]])
        params.to_excel(xw, sheet_name="params_features", index=False)
        if len(stab):
            stab.to_excel(xw, sheet_name="feature_stability", index=False)
        for which in ("test", "train"):
            dm = delong_matrix(exp, which)
            if dm is not None:
                dm.to_excel(xw, sheet_name=f"delong_p_{which}")
        if exp["failed"]:
            pd.DataFrame([{"combination": r["name"], "error": r["error"]} for r in exp["failed"]]) \
                .to_excel(xw, sheet_name="failed", index=False)
    cv_tab.to_csv(out / "metrics_nested_cv.csv", index=False)
    if len(te_tab):
        te_tab.to_csv(out / "metrics_test.csv", index=False)

    for r in exp["results"]:
        df = pd.DataFrame({"PatientID": ids[tr], "y_true": y[tr], "set": "train_oof"})
        P = np.atleast_2d(r["oof"].T).T
        for k in range(P.shape[1]):
            df[f"prob_{k}" if P.shape[1] > 1 else "prob"] = P[:, k]
        if r["test"] is not None:
            dt = pd.DataFrame({"PatientID": ids[te], "y_true": y[te], "set": "test"})
            Pt = np.atleast_2d(r["test"].T).T
            for k in range(Pt.shape[1]):
                dt[f"prob_{k}" if Pt.shape[1] > 1 else "prob"] = Pt[:, k]
            df = pd.concat([df, dt])
        df.to_csv(out / "predictions" / f"{r['name']}.csv", index=False)

    best = exp["results"][0]
    joblib.dump({"model": best["final"], "name": best["name"], "features": exp["names"],
                 "threshold": best["threshold"], "classes": exp["classes"], "config": cfg,
                 "members": best.get("members")}, out / "models" / "best_model.joblib")
    for r in exp["results"]:
        if r["selector"] != "ensemble":
            joblib.dump({"model": r["final"], "name": r["name"], "features": exp["names"],
                         "threshold": r["threshold"], "classes": exp["classes"]},
                        out / "models" / f"{r['name']}.joblib")
    log.info("best (by nested-CV AUC): %s  CV AUC=%.3f%s", best["name"], best["metrics_cv"].loc["AUC", "value"],
             f"  test AUC={best['metrics_test'].loc['AUC', 'value']:.3f}" if best["metrics_test"] is not None else "")


def predict_proba_saved(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    Xv = X.reindex(columns=bundle["features"]).to_numpy(float)
    models = bundle["model"] if isinstance(bundle["model"], list) else [bundle["model"]]
    P = np.mean([m.predict_proba(Xv) for m in models], 0)
    return P[:, 1] if P.shape[1] == 2 else P
