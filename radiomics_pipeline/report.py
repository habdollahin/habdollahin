"""Make all figures, permutation importance / SHAP for the best model, and a self-contained
HTML report (``report.html``) in the output directory."""
from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import plots
from .training import delong_matrix, feature_stability, summary_table, unwrap

log = logging.getLogger(__name__)

KEY_METRICS = ["AUC", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1",
               "MCC", "Kappa", "Brier", "ECE", "Cal_Intercept", "Cal_Slope", "HL_p", "PR_AUC", "LogLoss"]


def _importance(exp, cfg, figdir):
    best = exp["results"][0]
    if best["selector"] == "ensemble":
        cand = [r for r in exp["results"] if r["selector"] != "ensemble"]
        if not cand:
            return None, None
        best = cand[0]
    X, y = exp["X"], exp["y"]
    mask = exp["test"] if exp["test"].any() else exp["train"]
    feats = best["final_features"]
    cols = [exp["names"].index(f) for f in feats]
    scoring = "roc_auc_ovr" if exp["multi"] else "roc_auc"
    out = {}
    try:
        pi = permutation_importance(best["final"], X.to_numpy(float)[mask], y[mask], scoring=scoring,
                                    n_repeats=20, random_state=cfg["seed"], n_jobs=1)
        imp = pd.DataFrame({"importance": pi.importances_mean[cols], "std": pi.importances_std[cols]}, index=feats)
        imp.to_csv(Path(cfg["output_dir"]) / "feature_importance_permutation.csv")
        out["importance"] = plots.importance_plot(
            imp, figdir / "importance_permutation.png",
            f"Permutation importance - {best['name']} ({'test' if exp['test'].any() else 'train'})")
    except Exception as e:
        log.warning("permutation importance failed: %s", e)
    if cfg["modeling"].get("shap", True) and not exp["multi"]:
        try:
            import shap

            pipe = unwrap(best["final"])
            Xt = X.to_numpy(float)
            for name, step in pipe.steps[:-1]:
                if step == "passthrough" or name == "sampler":
                    continue
                Xt = step.transform(Xt)
            Xs = pd.DataFrame(Xt, columns=feats)
            clf = pipe.steps[-1][1]
            bg = shap.sample(Xs, min(50, len(Xs)), random_state=cfg["seed"])
            explainer = shap.Explainer(lambda a: clf.predict_proba(a)[:, 1], bg)
            sv = explainer(Xs.iloc[: min(200, len(Xs))])
            out["shap"] = plots.shap_summary(sv, Xs.iloc[: min(200, len(Xs))], figdir / "shap_summary.png")
        except Exception as e:
            log.warning("SHAP failed (optional): %s", e)
    try:
        out["corr"] = plots.correlation_plot(X[feats], figdir / "selected_feature_correlation.png")
    except Exception as e:
        log.warning("correlation plot failed: %s", e)
    return out, best


def make_figures(exp, cfg) -> dict:
    figdir = Path(cfg["output_dir"]) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    F = {}
    best = exp["results"][0]
    y, tr, te = exp["y"], exp["train"], exp["test"]
    labels = exp["classes"]
    sets = ["cv"] + (["test"] if te.any() else [])
    for s in sets:
        which = "test" if s == "test" else "train"
        mask = te if s == "test" else tr
        key = "test" if s == "test" else "oof"
        if not exp["multi"]:
            F[f"roc_{s}"] = plots.roc_overlay(exp, which, figdir / f"roc_all_{s}.png")
            F[f"roc_best_{s}"] = plots.roc_with_ci(
                y[mask], best[key], figdir / f"roc_best_{s}.png",
                f"ROC - {best['name']} ({'test' if s == 'test' else 'nested-CV'})",
                n_boot=int(cfg["modeling"].get("n_bootstrap", 1000)), seed=cfg["seed"], thr=best["threshold"])
            F[f"pr_{s}"] = plots.pr_overlay(exp, which, figdir / f"pr_{s}.png")
            F[f"cal_{s}"] = plots.calibration_plot(exp, which, figdir / f"calibration_{s}.png")
            F[f"dca_{s}"] = plots.decision_curve_plot(exp, which, figdir / f"decision_curve_{s}.png")
            F[f"dist_{s}"] = plots.probability_distribution(
                y[mask], best[key], labels, figdir / f"probability_by_class_{s}.png", best["threshold"],
                f"Predicted probability - {best['name']} ({'test' if s == 'test' else 'nested-CV'})")
            yhat = (best[key] >= best["threshold"]).astype(int)
        else:
            F[f"roc_{s}"] = plots.multiclass_roc(y[mask], best[key], labels, figdir / f"roc_ovr_{s}.png",
                                                 f"One-vs-rest ROC - {best['name']} ({s})")
            yhat = best[key].argmax(1)
        F[f"cm_{s}"] = plots.confusion_plot(y[mask], yhat, labels, figdir / f"confusion_{s}.png",
                                            f"Confusion - {best['name']} ({'test' if s == 'test' else 'nested-CV'})")
        F[f"heat_{s}"] = plots.heatmap_auc(exp, figdir / f"auc_heatmap_{s}.png", s)
        F[f"bars_{s}"] = plots.metric_bars(exp, figdir / f"metric_bars_{s}.png", s)
    stab = feature_stability(exp)
    if len(stab):
        first = next((r for r in exp["results"] if r["fold_features"]), None)
        if first:
            F["stability"] = plots.stability_plot(stab, first["name"], figdir / "feature_stability.png")
    imp, _ = _importance(exp, cfg, figdir)
    F.update({k: v for k, v in (imp or {}).items() if v})
    return {k: v for k, v in F.items() if v}


def _img(path: Path, inline: bool) -> str:
    if inline:
        b = base64.b64encode(Path(path).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b}" alt="{html.escape(Path(path).stem)}">'
    return f'<img src="figures/{Path(path).name}" alt="{html.escape(Path(path).stem)}">'


def _table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _metric_ci_table(mt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in [m for m in KEY_METRICS if m in mt.index]:
        v, lo, hi = mt.loc[k, ["value", "ci_low", "ci_high"]]
        ci = "" if np.isnan(lo) else f"{lo:.3f} – {hi:.3f}"
        rows.append({"Metric": k, "Value": v, "95% CI": ci})
    if "AUC_DeLong_CI" in mt.index:
        lo, hi = mt.loc["AUC_DeLong_CI", ["ci_low", "ci_high"]]
        rows.append({"Metric": "AUC (DeLong CI)", "Value": mt.loc["AUC", "value"], "95% CI": f"{lo:.3f} – {hi:.3f}"})
    return pd.DataFrame(rows)


def write_report(exp, cfg, figures: dict, inline: bool = True) -> Path:
    out = Path(cfg["output_dir"])
    best = exp["results"][0]
    te = exp["test"].any()
    cv = summary_table(exp, "cv")
    cols = ["combination"] + [c for c in ["AUC", "AUC_CI", "Accuracy", "Sensitivity", "Specificity", "F1", "MCC",
                                          "Brier", "fold_AUC_mean", "fold_AUC_sd", "n_features"] if c in cv]
    sec = []
    m = cfg["modeling"]
    class_counts = ", ".join(f"{html.escape(str(c))} = {int((exp['y'] == i).sum())}" for i, c in enumerate(exp["classes"]))
    n_tr, n_te = int(exp["train"].sum()), int(exp["test"].sum())
    calib = ("→ " + html.escape(str(m["calibration"])) + " calibration") if m.get("calibration") else ""
    sec.append(f"""<section><h2>Study design</h2><ul>
<li>Patients: <b>{len(exp['y'])}</b> (train {n_tr}, test {n_te}); classes: {class_counts}</li>
<li>Candidate features after robustness / missingness filter: <b>{len(exp['names'])}</b></li>
<li>Nested CV: {m.get('outer_repeats', 1)} × {m['outer_folds']}-fold outer, {m['inner_folds']}-fold inner;
hyper-parameter search <b>{html.escape(str(m['search']))}</b> ({m.get('n_iter')} iterations), scoring {html.escape(str(m['scoring']))}</li>
<li>Pipeline per fold: median imputation → constant-feature removal → {html.escape(m.get('scaler', 'standard'))} scaling →
correlation filter (|ρ| &gt; {m.get('correlation_threshold')}) → feature selector → {html.escape(str(m.get('imbalance')))} → classifier
{calib}</li>
<li>Selectors: {', '.join(m['selectors'])}; models: {', '.join(m['models'])}; operating threshold: {html.escape(str(m.get('threshold')))} on OOF predictions</li>
<li>Ranking is by nested-CV AUC; the test set is used only once, for the final report.</li></ul></section>""")
    cards = [("Best combination", html.escape(best["name"])),
             ("Nested-CV AUC", f"{best['metrics_cv'].loc['AUC', 'value']:.3f}")]
    if te:
        cards.append(("Test AUC", f"{best['metrics_test'].loc['AUC', 'value']:.3f}"))
        cards.append(("Test accuracy", f"{best['metrics_test'].loc['Accuracy', 'value']:.3f}"))
    sec.insert(0, '<div class="cards">' + "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                                                   for k, v in cards) + "</div>")
    sec.append(f"<section><h2>All combinations (nested-CV on training set)</h2>{_table(cv[cols])}</section>")
    if te:
        tt = summary_table(exp, "test")
        tcols = ["combination"] + [c for c in ["AUC", "AUC_CI", "Accuracy", "Sensitivity", "Specificity", "PPV",
                                               "NPV", "F1", "MCC", "Brier"] if c in tt]
        sec.append(f"<section><h2>Held-out test set</h2>{_table(tt[tcols])}</section>")
    sec.append("<section><h2>Best model: " + html.escape(best["name"]) + "</h2><div class='two'>"
               + f"<div><h3>Nested-CV</h3>{_table(_metric_ci_table(best['metrics_cv']))}</div>"
               + (f"<div><h3>Test</h3>{_table(_metric_ci_table(best['metrics_test']))}</div>" if te else "")
               + "</div>"
               + f"<p>Hyper-parameters: <code>{html.escape(str(best['final_params']))}</code></p>"
               + f"<p>Selected features ({len(best['final_features'])}): "
               + ", ".join(f"<code>{html.escape(f)}</code>" for f in best["final_features"]) + "</p></section>")

    def figs(keys, title):
        imgs = [f"<figure>{_img(figures[k], inline)}</figure>" for k in keys if k in figures]
        return f"<section><h2>{title}</h2><div class='grid'>{''.join(imgs)}</div></section>" if imgs else ""

    s = ["test", "cv"] if te else ["cv"]
    sec.append(figs([f"roc_best_{x}" for x in s] + [f"roc_{x}" for x in s], "ROC curves"))
    sec.append(figs([f"bars_{x}" for x in s] + [f"heat_{x}" for x in s], "Model comparison"))
    sec.append(figs([f"cal_{x}" for x in s] + [f"dca_{x}" for x in s], "Calibration and clinical utility"))
    sec.append(figs([f"pr_{x}" for x in s] + [f"cm_{x}" for x in s] + [f"dist_{x}" for x in s],
                    "Precision-recall, confusion matrices, probability distributions"))
    sec.append(figs(["importance", "shap", "stability", "corr"], "Interpretability and feature stability"))
    dm = delong_matrix(exp, "test" if te else "train")
    if dm is not None and len(dm) > 1:
        d = dm.iloc[:, :1].rename(columns={dm.columns[0]: f"p vs {dm.columns[0]}"}).iloc[1:].reset_index()
        d.columns = ["combination", d.columns[1]]
        sec.append(f"<section><h2>DeLong test vs best ({'test' if te else 'nested-CV'})</h2>{_table(d, '{:.4f}')}</section>")
    if exp["failed"]:
        f = pd.DataFrame([{"combination": r["name"], "error": r["error"]} for r in exp["failed"]])
        sec.append(f"<section><h2>Failed combinations</h2>{_table(f)}</section>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Radiomics Report</title>
<style>
:root{{--bg:#f6f6f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111110;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}}}}
:root[data-theme="dark"]{{--bg:#111110;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:0 0 12px}} h3{{font-size:14px;color:var(--ink2)}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}}
.k{{color:var(--ink2);font-size:12px}} .v{{font-size:22px;font-weight:600;word-break:break-word}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}} th{{color:var(--ink2);font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
figure{{margin:0}} img{{max-width:100%;height:auto;border-radius:6px;background:#fcfcfb}}
code{{font-size:12px;word-break:break-all}} .muted{{color:var(--ink2)}}
</style></head><body><main>
<h1>Radiomics modelling report</h1>
<p class="muted">Output directory: {html.escape(str(out.resolve()))}. Research use only.</p>
{''.join(sec)}
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    log.info("report written to %s", path)
    return path
