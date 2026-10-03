"""Tests for radiomics_pipeline (skipped when its optional dependencies are missing)."""
import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("sklearn")

from radiomics_pipeline import io as rio  # noqa: E402
from radiomics_pipeline.config import load_config  # noqa: E402
from radiomics_pipeline.labels import encode_labels, normalize_id  # noqa: E402
from radiomics_pipeline.metrics import (binary_metrics, bootstrap_metrics, choose_threshold, delong_ci,  # noqa: E402
                                        delong_test, hosmer_lemeshow)
from radiomics_pipeline.preprocess import denoise, normalize, resample, resample_mask  # noqa: E402
from radiomics_pipeline.registration import align_mask, dice  # noqa: E402
from radiomics_pipeline.selection import (MRMR, Boruta, CorrelationFilter, ReliefF,  # noqa: E402
                                          UnivariateSelector, benjamini_hochberg, make_selector)


def _blob(shape=(30, 48, 48), center=(15, 24, 24), r=8, spacing=(1.0, 1.0, 2.0)):
    z, y, x = np.indices(shape)
    m = ((z - center[0]) * spacing[2]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= (r * 1.5) ** 2
    img = np.where(m, 100.0, 0.0) + np.random.default_rng(0).normal(0, 5, shape)
    im = sitk.GetImageFromArray(img.astype(np.float32))
    im.SetSpacing(spacing)
    mk = sitk.GetImageFromArray(m.astype(np.uint8))
    mk.CopyInformation(im)
    return im, mk


def test_resample_keeps_extent_and_mask_volume():
    im, mk = _blob()
    r = resample(im, (1.0, 1.0, 1.0))
    assert r.GetSize() == (48, 48, 60)
    rm = resample_mask(mk, r)
    v0 = sitk.GetArrayFromImage(mk).sum() * 2.0
    v1 = sitk.GetArrayFromImage(rm).sum() * 1.0
    assert abs(v1 - v0) / v0 < 0.05


def test_denoise_and_normalize():
    im, mk = _blob()
    for method in ["gaussian", "median", "curvature_flow", "anisotropic", "bilateral"]:
        out = denoise(im, {"method": method})
        assert out.GetSize() == im.GetSize()
        assert sitk.GetArrayFromImage(out).std() <= sitk.GetArrayFromImage(im).std() + 1e-3
    z = sitk.GetArrayFromImage(normalize(im, mk, {"method": "zscore_roi"}))
    roi = z[sitk.GetArrayFromImage(mk) > 0]
    assert abs(roi.mean()) < 1e-3 and abs(roi.std() - 1) < 1e-3
    p = sitk.GetArrayFromImage(normalize(im, None, {"method": "percentile"}))
    assert p.min() >= 0 and p.max() <= 1


def test_registration_recovers_shift():
    cfg = load_config(overrides={"registration": {"method": "rigid", "iterations": 150}})
    fixed, fixed_mask = _blob()
    # the mask was drawn on a reference image shifted by 6 mm in x
    tx = sitk.TranslationTransform(3, (6.0, 0.0, 0.0))
    ref = sitk.Resample(fixed, fixed, tx, sitk.sitkLinear, 0.0)
    ref_mask = sitk.Resample(fixed_mask, fixed, tx, sitk.sitkNearestNeighbor, 0)
    naive = resample_mask(ref_mask, fixed)
    aligned = align_mask(fixed, ref_mask, cfg, mask_reference=ref)
    assert dice(aligned, fixed_mask) > 0.9 > dice(naive, fixed_mask)


def test_align_mask_physical_resampling():
    cfg = load_config()
    im, mk = _blob()
    fine = resample(sitk.Cast(mk, sitk.sitkFloat32), (0.5, 0.5, 1.0), "linear", 0.0) >= 0.5
    out = align_mask(im, sitk.Cast(fine, sitk.sitkUInt8), cfg)
    assert out.GetSize() == im.GetSize()
    assert dice(out, mk) > 0.95


def test_rtstruct_roundtrip(tmp_path):
    from radiomics_pipeline.synthetic import make_volume, write_ct_series, write_rtstruct

    img, mask = make_volume(1, np.random.default_rng(3))
    info = write_ct_series(img, tmp_path / "P1" / "CT", "P1")
    write_rtstruct(mask, info, tmp_path / "P1" / "RS_P1.dcm", "P1")
    cfg = load_config(overrides={"data": {"root": str(tmp_path), "roi_name_regex": "GTV"}})
    (case,) = rio.discover_patients(cfg)
    assert case.mask_format == "rtstruct" and len(case.image_files) == img.shape[0]
    ref = rio.read_image(case.image_files)
    assert np.abs(sitk.GetArrayFromImage(ref) - img).max() <= 0.51
    m = sitk.GetArrayFromImage(rio.read_mask(case, ref, "GTV"))
    assert 2 * (m & mask).sum() / (m.sum() + mask.sum()) > 0.98


def test_id_normalisation_and_labels():
    assert normalize_id(" PAT0012 ", ["strip", "lower", "nozeros"]) == "pat12"
    assert normalize_id(12.0, ["strip"]) == "12"
    assert normalize_id("ID_ab-01", ["alnum", "lower"]) == "idab01"
    y, cls = encode_labels(pd.Series(["benign", "malignant", "benign"]))
    assert cls == ["benign", "malignant"] and list(y) == [0, 1, 0]
    y, cls = encode_labels(pd.Series(["a", "b", "c", "a"]))
    assert cls == ["a", "b", "c"] and list(y) == [0, 1, 2, 0]


def test_delong_matches_sklearn_and_detects_difference():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(0, 0.6, 300)
    bad = y + rng.normal(0, 3.0, 300)
    auc, lo, hi = delong_ci(y, good)
    assert abs(auc - roc_auc_score(y, good)) < 1e-9 and lo < auc < hi
    _, p = delong_test(y, good, bad)
    assert p < 0.001
    _, p_same = delong_test(y, good, good)
    assert p_same == 1.0


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.5 * y + rng.uniform(0, 0.5, 200), 0, 1)
    thr = choose_threshold(y, p, "youden")
    m = binary_metrics(y, p, thr)
    assert m["TP"] + m["FN"] == y.sum()
    assert abs(m["Sensitivity"] - m["TP"] / (m["TP"] + m["FN"])) < 1e-12
    b = bootstrap_metrics(y, p, thr, n_boot=100)
    for k in ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Brier", "Cal_Slope", "HL_p"]:
        assert k in b.index
    assert b.loc["AUC", "ci_low"] <= b.loc["AUC", "value"] <= b.loc["AUC", "ci_high"]
    _, hl_p = hosmer_lemeshow(y, np.clip(y * 0.8 + 0.1, 0, 1))
    assert 0 <= hl_p <= 1


def test_selectors_find_informative_features():
    rng = np.random.default_rng(0)
    n, p = 120, 40
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)
    X[:, 2] = X[:, 0] + rng.normal(0, 0.01, n)  # redundant copy of feature 0
    cf = CorrelationFilter(0.9).fit(X, y)
    assert cf.support_[[0, 2]].sum() == 1
    for sel in [UnivariateSelector("anova", k=3), UnivariateSelector("mannwhitney", k=3), MRMR(k=3),
                ReliefF(k=4), Boruta(n_iter=15, k=4)]:
        s = sel.fit(X, y).support_
        assert s[1] and (s[0] or s[2]), type(sel).__name__
    for name in ["lasso", "elasticnet", "rfe_svm", "rf_importance", "extratrees", "fdr"]:
        s = make_selector(name, k=4).fit(X, y).support_
        assert s[1], name
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= [0.01, 0.04, 0.03, 0.5]) and q[3] == 0.5


def _toy_data(n=90, p=30, k=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = rng.normal(size=(n, p))
    X[:, :3] += y[:, None] * 1.2
    ids = np.array([f"P{i}" for i in range(n)])
    return {"X": pd.DataFrame(X, index=ids, columns=[f"f{i}" for i in range(p)]), "y": y,
            "classes": [f"c{i}" for i in range(k)], "ids": ids, "split": None, "groups": None}


@pytest.mark.parametrize("search,k", [("random", 2), ("optuna", 2), ("grid", 3)])
def test_experiment_runs(tmp_path, search, k):
    pytest.importorskip("optuna") if search == "optuna" else None
    from radiomics_pipeline.report import make_figures, write_report
    from radiomics_pipeline.training import predict_proba_saved, run_experiment

    cfg = load_config(overrides={
        "output_dir": str(tmp_path), "features": {"robustness": {"enabled": False}},
        "modeling": {"outer_folds": 3, "inner_folds": 2, "search": search, "n_iter": 2, "n_bootstrap": 50,
                     "selectors": ["anova", "mrmr"], "selector_k": [3, 5], "models": ["logistic", "knn"],
                     "ensemble_top_n": 2, "shap": False}})
    data = _toy_data(k=k)
    exp = run_experiment(data, cfg)
    best = exp["results"][0]
    assert best["metrics_cv"].loc["AUC", "value"] > 0.75
    assert (tmp_path / "results.xlsx").exists() and (tmp_path / "models" / "best_model.joblib").exists()
    html = write_report(exp, cfg, make_figures(exp, cfg))
    assert html.exists() and "ROC" in html.read_text()
    import joblib

    bundle = joblib.load(tmp_path / "models" / "best_model.joblib")
    P = predict_proba_saved(bundle, data["X"].iloc[:5])
    assert P.shape[0] == 5


def test_end_to_end_synthetic(tmp_path):
    pytest.importorskip("radiomics")
    from radiomics_pipeline.cli import main
    from radiomics_pipeline.synthetic import make_cohort

    root = make_cohort(tmp_path / "data", n=24, seed=1)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
output_dir: {tmp_path / 'run'}
data: {{root: {root}, roi_name_regex: GTV}}
preprocess: {{resample: {{spacing: [2.0, 2.0, 2.0]}}, denoise: {{method: gaussian, sigma: 0.5}}}}
features: {{bin_width: 10, image_types: {{Original: {{}}}}, robustness: {{enabled: true}}}}
labels: {{file: {root / 'labels.xlsx'}, id_column: Patient ID, label_column: Outcome, positive_class: positive}}
modeling: {{test_size: 0.25, outer_folds: 3, inner_folds: 2, n_iter: 2, n_bootstrap: 50,
           selectors: [anova], selector_k: [3], models: [logistic], ensemble_top_n: 0, shap: false}}
""")
    main(["run", "--config", str(cfg_path)])
    feats = pd.read_csv(tmp_path / "run" / "features" / "features.csv", index_col=0)
    assert feats.shape[0] == 24 and feats.shape[1] > 50
    assert (tmp_path / "run" / "report.html").exists()
    assert (tmp_path / "run" / "features" / "robustness_icc.csv").exists()
    main(["predict", "--model", str(tmp_path / "run" / "models" / "best_model.joblib"), "--root", str(root),
          "--out", str(tmp_path / "pred")])
    pred = pd.read_csv(tmp_path / "pred" / "predictions.csv")
    assert len(pred) == 24 and pred.filter(like="P(").shape[1] == 1
