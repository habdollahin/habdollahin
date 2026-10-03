"""Figures: ROC (with bootstrap CI band), precision-recall, calibration, decision curves,
confusion matrices, model x selector heatmap, feature stability, importance/SHAP,
correlation of selected features and class-wise probability distributions."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from .metrics import decision_curve  # noqa: E402

log = logging.getLogger(__name__)

# validated categorical palette (fixed order, never cycled); extra series fold to grey
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b5b4ae"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = "Blues"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _subset(exp, which):
    mask = exp["test"] if which == "test" else exp["train"]
    key = "test" if which == "test" else "oof"
    return exp["y"][mask], key


def roc_overlay(exp, which, path, max_series=8):
    y, key = _subset(exp, which)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    res = exp["results"]
    for i, r in enumerate(res[::-1]):  # draw the best last (on top)
        rank = len(res) - 1 - i
        fpr, tpr, _ = roc_curve(y, r[key])
        a = auc(fpr, tpr)
        if rank < max_series:
            ax.plot(fpr, tpr, color=SERIES[rank], lw=2.2 if rank == 0 else 1.6,
                    label=f"{r['name']}  AUC {a:.3f}", zorder=10 - rank)
        else:
            ax.plot(fpr, tpr, color=OTHER, lw=0.8, alpha=0.6, zorder=0)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="lower right", fontsize=8)
    ax.set(xlabel="1 - Specificity (FPR)", ylabel="Sensitivity (TPR)", xlim=(0, 1), ylim=(0, 1.01),
           title=f"ROC - {'held-out test' if which == 'test' else 'nested-CV (train OOF)'}")
    if len(res) > max_series:
        ax.text(0.98, 0.02 + 0.045 * (max_series + 0.3), f"grey: {len(res) - max_series} other combinations",
                transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    return _save(fig, path)


def roc_with_ci(y, p, path, title, n_boot=1000, seed=0, thr=None):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 201)
    fpr, tpr, thrs = roc_curve(y, p)
    tprs = []
    idx_c = [np.flatnonzero(y == c) for c in (0, 1)]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_c])
        f, t, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, f, t))
    lo, hi = np.percentile(tprs, [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.fill_between(grid, lo, hi, color=SERIES[0], alpha=0.18, lw=0, label="95% bootstrap CI")
    ax.plot(fpr, tpr, color=SERIES[0], label=f"AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    if thr is not None:
        yhat = p >= thr
        sens = (yhat & (y == 1)).sum() / max(1, (y == 1).sum())
        spec = (~yhat & (y == 0)).sum() / max(1, (y == 0).sum())
        ax.scatter([1 - spec], [sens], s=70, color=SERIES[1], edgecolor="#fcfcfb", lw=2, zorder=5,
                   label=f"threshold {thr:.2f}: sens {sens:.2f}, spec {spec:.2f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title=title)
    return _save(fig, path)


def pr_overlay(exp, which, path, max_series=8):
    y, key = _subset(exp, which)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.axhline(y.mean(), ls="--", lw=1, color=INK2, label=f"prevalence {y.mean():.2f}")
    for rank, r in enumerate(exp["results"][:max_series]):
        pr, rc, _ = precision_recall_curve(y, r[key])
        ax.plot(rc, pr, color=SERIES[rank], lw=1.6, label=f"{r['name']}  AP {auc(rc, pr):.3f}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Precision-recall - {'test' if which == 'test' else 'nested-CV'}")
    return _save(fig, path)


def calibration_plot(exp, which, path, top=4, bins=None):
    y, key = _subset(exp, which)
    bins = bins or int(np.clip(len(y) // 15, 3, 10))  # >= ~15 patients per bin
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="perfect calibration")
    for rank, r in enumerate(exp["results"][:top]):
        p = r[key]
        frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        mt = r[f"metrics_{'test' if which == 'test' else 'cv'}"]
        lab = (f"{r['name']}  Brier {mt.loc['Brier', 'value']:.3f}, "
               f"slope {mt.loc['Cal_Slope', 'value']:.2f}" if "Cal_Slope" in mt.index else r["name"])
        ax.plot(mean, frac, marker="o", ms=5, color=SERIES[rank], lw=1.6, label=lab,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        axh.hist(p, bins=20, range=(0, 1), histtype="step", color=SERIES[rank], lw=1.4)
    ax.set(ylabel="Observed fraction positive", xlim=(0, 1), ylim=(0, 1),
           title=f"Calibration - {'test' if which == 'test' else 'nested-CV'}")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#fcfcfb", edgecolor=GRID, framealpha=0.92)
    ax.text(0.99, 0.02, f"{bins} quantile bins", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    axh.set(xlabel="Predicted probability", ylabel="count")
    return _save(fig, path)


def decision_curve_plot(exp, which, path, top=4):
    y, key = _subset(exp, which)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    d0 = decision_curve(y, exp["results"][0][key])
    ax.plot(d0["threshold"], d0["treat_all"], color=INK2, lw=1, ls="--", label="treat all")
    ax.plot(d0["threshold"], d0["treat_none"], color=INK, lw=1, label="treat none")
    for rank, r in enumerate(exp["results"][:top]):
        d = decision_curve(y, r[key])
        ax.plot(d["threshold"], d["model"], color=SERIES[rank], lw=1.8, label=r["name"])
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0, 1),
           ylim=(min(-0.05, d0["model"].min()), max(0.05, y.mean() * 1.15)),
           title=f"Decision curve analysis - {'test' if which == 'test' else 'nested-CV'}")
    ax.legend(fontsize=8)
    return _save(fig, path)


def confusion_plot(y, yhat, labels, path, title):
    cm = confusion_matrix(y, yhat, labels=list(range(len(labels))))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4 + 0.4 * len(labels), 4 + 0.4 * len(labels)))
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="#ffffff" if cmn[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(len(labels)), [str(c) for c in labels])
    ax.set_yticks(range(len(labels)), [str(c) for c in labels])
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    return _save(fig, path)


def heatmap_auc(exp, path, which="cv"):
    rows = [(r["selector"], r["model"], r[f"metrics_{which}"].loc["AUC", "value"])
            for r in exp["results"] if r["selector"] != "ensemble" and r[f"metrics_{which}"] is not None]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["selector", "model", "AUC"]).pivot(index="selector", columns="model", values="AUC")
    fig, ax = plt.subplots(figsize=(1.3 + 1.05 * df.shape[1], 1.2 + 0.55 * df.shape[0]))
    vmin = max(0.5, np.nanmin(df.values) - 0.02)
    ax.imshow(df.values, cmap=SEQ, vmin=vmin, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if (v - vmin) / (1 - vmin + 1e-9) > 0.6 else INK)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=35, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.set_title(f"AUC by selector x model ({'nested-CV' if which == 'cv' else 'test'})")
    return _save(fig, path)


def metric_bars(exp, path, which="cv", metrics=("AUC", "Accuracy", "Sensitivity", "Specificity", "F1")):
    res = [r for r in exp["results"] if r[f"metrics_{which}"] is not None][:8]
    if not res:
        return None
    metrics = [m for m in metrics if m in res[0][f"metrics_{which}"].index]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(res)
    x = np.arange(len(metrics))
    for i, r in enumerate(res):
        mt = r[f"metrics_{which}"]
        v = mt.loc[metrics, "value"].to_numpy(float)
        lo = v - mt.loc[metrics, "ci_low"].to_numpy(float)
        hi = mt.loc[metrics, "ci_high"].to_numpy(float) - v
        ax.bar(x + i * w - 0.4 + w / 2, v, w * 0.9, color=SERIES[i], label=r["name"],
               yerr=np.vstack([np.nan_to_num(lo), np.nan_to_num(hi)]), error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_xticks(x, metrics)
    ax.set(ylim=(0, 1.05), ylabel="value (95% CI)",
           title=f"Top combinations - {'nested-CV' if which == 'cv' else 'test'}")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, path)


def stability_plot(stab: pd.DataFrame, combo: str, path, top=20):
    s = stab[stab["combination"] == combo].sort_values("frequency", ascending=False).head(top)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.28 * len(s)))
    ax.barh(range(len(s))[::-1], s["frequency"], color=SERIES[0], height=0.7)
    ax.set_yticks(range(len(s))[::-1], [_short(f) for f in s["feature"]], fontsize=8)
    ax.set(xlim=(0, 1), xlabel="selection frequency across outer folds", title=f"Feature stability - {combo}")
    return _save(fig, path)


def _short(name: str, n=55) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def importance_plot(imp: pd.DataFrame, path, title):
    imp = imp.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.3 * len(imp)))
    ax.barh(range(len(imp)), imp["importance"], xerr=imp.get("std"), color=SERIES[0], height=0.7,
            error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_yticks(range(len(imp)), [_short(f) for f in imp.index], fontsize=8)
    ax.set(xlabel="mean decrease in AUC (permutation)", title=title)
    return _save(fig, path)


def correlation_plot(X: pd.DataFrame, path, title="Spearman correlation - selected features"):
    if X.shape[1] < 2:
        return None
    C = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1.5 + 0.35 * len(C), 1.2 + 0.35 * len(C)))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(C)), [_short(c, 30) for c in C.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(C)), [_short(c, 30) for c in C.columns], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    return _save(fig, path)


def probability_distribution(y, p, labels, path, thr=None, title="Predicted probability by class"):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    data = [p[y == k] for k in range(len(labels))]
    parts = ax.violinplot([d for d in data if len(d)], showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(SERIES[i])
        b.set_alpha(0.35)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(INK2)
        parts[k].set_linewidth(1)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.08, 0.08, len(d)), d, s=12, color=SERIES[i], alpha=0.8,
                   edgecolor="#fcfcfb", lw=0.5)
    if thr is not None:
        ax.axhline(thr, ls="--", lw=1, color=INK2)
        ax.text(len(labels) + 0.45, thr, f"threshold {thr:.2f}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(1, len(labels) + 1), [str(c) for c in labels])
    ax.set(ylabel="P(positive)", ylim=(-0.02, 1.02), title=title)
    return _save(fig, path)


def multiclass_roc(y, P, labels, path, title):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    for k, lab in enumerate(labels[:8]):
        f, t, _ = roc_curve(y == k, P[:, k])
        ax.plot(f, t, color=SERIES[k], lw=1.8, label=f"{lab} vs rest  AUC {auc(f, t):.3f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=title, xlim=(0, 1), ylim=(0, 1.01))
    return _save(fig, path)


def shap_summary(shap_values, Xs: pd.DataFrame, path):
    import shap

    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False, max_display=20, plot_size=(9, 1.6 + 0.42 * min(20, Xs.shape[1])))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary - best model")
    return _save(fig, path)
