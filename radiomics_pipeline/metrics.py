"""Classification metrics with bootstrap confidence intervals, DeLong AUC CI/test,
calibration statistics (Brier, ECE, intercept/slope, Hosmer-Lemeshow), threshold
selection and decision-curve analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

# --------------------------------------------------------------------------- thresholds


def choose_threshold(y, p, rule="youden") -> float:
    """Pick an operating point on *training* (out-of-fold) predictions."""
    if isinstance(rule, (int, float)) or str(rule).replace(".", "", 1).isdigit():
        return float(rule)
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    if rule == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if rule == "f1":
        cands = np.unique(p)
        f1s = [f1_score(y, p >= t, zero_division=0) for t in cands]
        return float(cands[int(np.argmax(f1s))])
    if rule == "sens90":
        ok = np.flatnonzero(tpr >= 0.9)
        return float(thr[ok[0]]) if len(ok) else 0.5
    if rule == "spec90":
        ok = np.flatnonzero(1 - fpr >= 0.9)
        return float(thr[ok[-1]]) if len(ok) else 0.5
    raise ValueError(f"unknown threshold rule {rule!r}")


# --------------------------------------------------------------------------- calibration
def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal a=0, b=1.
    The intercept is 'calibration-in-the-large' estimated with the slope fixed at 1."""
    import statsmodels.api as sm  # optional dependency

    import warnings

    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        warnings.filterwarnings("ignore", module="statsmodels")
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        return float(inter), float(slope)
    except Exception:
        return np.nan, np.nan


def _cal_int_slope(y, p):
    try:
        return calibration_intercept_slope(y, p)
    except ImportError:  # fall back to scikit-learn (unpenalised) if statsmodels is missing
        from sklearn.linear_model import LogisticRegression

        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
            slope = float(lr.coef_[0, 0])
        except Exception:
            slope = np.nan
        # intercept with slope fixed at 1: solve mean(sigmoid(a + lp)) = mean(y)
        from scipy.optimize import brentq

        f = lambda a: (1 / (1 + np.exp(-(a + lp.ravel())))).mean() - np.mean(y)  # noqa: E731
        try:
            inter = brentq(f, -20, 20)
        except ValueError:
            inter = np.nan
        return float(inter), slope


def hosmer_lemeshow(y, p, groups=10) -> tuple[float, float]:
    df = pd.DataFrame({"y": y, "p": p})
    try:
        df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    except ValueError:
        return np.nan, np.nan
    g = df.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    pbar = g["exp"] / g["n"]
    den = g["n"] * pbar * (1 - pbar)
    hl = float((((g["obs"] - g["exp"]) ** 2) / den.replace(0, np.nan)).sum())
    dof = max(1, len(g) - 2)
    return hl, float(stats.chi2.sf(hl, dof))


def expected_calibration_error(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(np.mean(y[m]) - np.mean(p[m]))
    return float(ece)


# --------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _delong_cov(y, preds):
    """Fast DeLong (Sun & Xu 2014). preds: (k, n). Returns aucs (k,), cov (k,k)."""
    y = np.asarray(y)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.atleast_2d(np.cov(v10)) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    return aucs, sx / m + sy / n


def delong_ci(y, p, alpha=0.95) -> tuple[float, float, float]:
    auc, cov = _delong_cov(y, np.asarray(p)[None])
    se = np.sqrt(max(cov[0, 0], 1e-12))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return float(auc[0]), float(max(0, auc[0] - z * se)), float(min(1, auc[0] + z * se))


def delong_test(y, p1, p2) -> tuple[float, float]:
    """Two-sided DeLong test for correlated ROC curves. Returns (z, p-value)."""
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- point metrics
def binary_metrics(y, p, thr) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else np.nan  # noqa: E731
    sens, spec = sd(tp, tp + fn), sd(tn, tn + fp)
    out = {
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "PR_AUC": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "Accuracy": accuracy_score(y, yhat),
        "Balanced_Accuracy": balanced_accuracy_score(y, yhat),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": sd(tp, tp + fp),
        "NPV": sd(tn, tn + fn),
        "F1": f1_score(y, yhat, zero_division=0),
        "MCC": matthews_corrcoef(y, yhat),
        "Kappa": cohen_kappa_score(y, yhat),
        "LR+": sd(sens, 1 - spec),
        "LR-": sd(1 - sens, spec),
        "Youden_J": sens + spec - 1,
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "ECE": expected_calibration_error(y, p),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Threshold": thr,
    }
    return out


def binary_metrics_full(y, p, thr) -> dict:
    """Point metrics + calibration statistics (not bootstrapped, slower)."""
    out = binary_metrics(y, p, thr)
    out["Cal_Intercept"], out["Cal_Slope"] = _cal_int_slope(np.asarray(y), np.asarray(p))
    out["HL_chi2"], out["HL_p"] = hosmer_lemeshow(np.asarray(y), np.asarray(p))
    return out


def multiclass_metrics(y, P) -> dict:
    y = np.asarray(y)
    P = np.asarray(P, float)
    K = P.shape[1]
    yhat = P.argmax(1)
    cm = confusion_matrix(y, yhat, labels=list(range(K)))
    spec = []
    for k in range(K):
        tn = cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if tn + fp else np.nan)
    onehot = np.eye(K)[y]
    return {
        "AUC": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K))),
        "AUC_weighted": roc_auc_score(y, P, multi_class="ovr", average="weighted", labels=list(range(K))),
        "Accuracy": accuracy_score(y, yhat),
        "Balanced_Accuracy": balanced_accuracy_score(y, yhat),
        "Sensitivity": recall_score(y, yhat, average="macro", zero_division=0),
        "Specificity": float(np.nanmean(spec)),
        "PPV": precision_score(y, yhat, average="macro", zero_division=0),
        "F1": f1_score(y, yhat, average="macro", zero_division=0),
        "F1_weighted": f1_score(y, yhat, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, yhat),
        "Kappa": cohen_kappa_score(y, yhat),
        "Brier": float(((P - onehot) ** 2).sum(1).mean()),
        "LogLoss": log_loss(y, np.clip(P, 1e-7, 1) / np.clip(P, 1e-7, 1).sum(1, keepdims=True), labels=list(range(K))),
    }


def bootstrap_metrics(y, p, thr=None, n_boot=1000, seed=0, alpha=0.95) -> pd.DataFrame:
    """Point estimate + percentile bootstrap CI (stratified resampling) for every metric."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    multi = p.ndim == 2
    point = (multiclass_metrics(y, p) if multi else binary_metrics_full(y, p, thr))
    fn = (lambda yy, pp: multiclass_metrics(yy, pp)) if multi else (lambda yy, pp: binary_metrics(yy, pp, thr))
    rng = np.random.default_rng(seed)
    idx_by_c = [np.flatnonzero(y == c) for c in np.unique(y)]
    boots = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_c])
        try:
            boots.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    B = pd.DataFrame(boots)
    lo_q, hi_q = (1 - alpha) / 2, 1 - (1 - alpha) / 2
    rows = []
    for k, v in point.items():
        if k in B and k not in ("TP", "FP", "TN", "FN", "Threshold"):
            col = pd.to_numeric(B[k], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = (col.quantile(lo_q), col.quantile(hi_q)) if len(col) else (np.nan, np.nan)
        else:
            lo = hi = np.nan
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows).set_index("metric")
    if not multi:
        _, lo, hi = delong_ci(y, p, alpha)
        df.loc["AUC_DeLong_CI"] = [df.loc["AUC", "value"], lo, hi]
    return df


# --------------------------------------------------------------------------- decision curves
def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = t / (1 - t)
        rows.append({"threshold": t, "model": tp / n - fp / n * w, "treat_all": prev - (1 - prev) * w,
                     "treat_none": 0.0})
    return pd.DataFrame(rows)
