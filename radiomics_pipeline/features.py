"""Per-patient processing (read -> preprocess -> align mask -> extract) and the PyRadiomics
feature extraction driver with caching, parallelism, QC overlays and an optional
mask-perturbation robustness (ICC) analysis.
"""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from . import io as rio
from .preprocess import clean_mask, normalize, perturb_mask, preprocess_image
from .registration import align_mask

log = logging.getLogger(__name__)


def radiomics_settings(cfg: dict) -> dict:
    f = cfg["features"]
    s = {
        "normalize": False,               # done by our own preprocessing
        "resampledPixelSpacing": None,    # done by our own preprocessing
        "correctMask": True,
        "geometryTolerance": 1e-3,
        "force2D": bool(f.get("force_2d", False)),
        "label": 1,
        "minimumROIDimensions": 2,
        "minimumROISize": int(cfg["mask"].get("min_voxels", 10)),
    }
    if f.get("bin_width"):
        s["binWidth"] = float(f["bin_width"])
    else:
        s["binCount"] = int(f.get("bin_count", 32))
    if f.get("resegment_range"):
        s["resegmentRange"] = list(f["resegment_range"])
        s["resegmentShape"] = True
    s.update(f.get("extra_settings") or {})
    return s


def build_extractor(cfg: dict):
    from radiomics import featureextractor, setVerbosity

    setVerbosity(40)
    ext = featureextractor.RadiomicsFeatureExtractor(**radiomics_settings(cfg))
    ext.disableAllImageTypes()
    for name, params in (cfg["features"]["image_types"] or {"Original": {}}).items():
        ext.enableImageTypeByName(name, customArgs=params or {})
    ext.disableAllFeatures()
    for fc in cfg["features"]["feature_classes"]:
        ext.enableFeatureClassByName(fc)
    return ext


def load_case(case: rio.PatientCase, cfg: dict) -> tuple[sitk.Image, sitk.Image, dict]:
    """Return the preprocessed image, the aligned + cleaned mask and QC info."""
    d = cfg["data"]
    raw = rio.read_image(case.image_files)
    if d.get("suv") and case.image_modality == "PT" and case.header is not None:
        raw = raw * float(rio.suv_factor(case.header))
    raw_mask = rio.read_mask(case, raw, d.get("roi_name_regex"), d.get("mask_label"))
    ref = rio.read_image(case.mask_reference) if case.mask_reference is not None else None

    img = preprocess_image(raw, cfg, case.image_modality)
    mask = align_mask(img, raw_mask, cfg, ref)
    mask = clean_mask(mask, cfg["mask"])
    img = normalize(img, mask, cfg["preprocess"].get("normalize"))

    nvox = int(sitk.GetArrayViewFromImage(mask).sum())
    if nvox < int(cfg["mask"].get("min_voxels", 10)):
        raise ValueError(f"mask has only {nvox} voxels after alignment")
    qc = {
        "modality": case.image_modality,
        "mask_format": case.mask_format,
        "raw_spacing": list(raw.GetSpacing()),
        "raw_size": list(raw.GetSize()),
        "spacing": list(img.GetSpacing()),
        "size": list(img.GetSize()),
        "mask_voxels": nvox,
        "mask_volume_ml": nvox * float(np.prod(img.GetSpacing())) / 1000.0,
        "registered": case.mask_reference is not None,
    }
    return img, mask, qc


def qc_overlay(img: sitk.Image, mask: sitk.Image, out_png: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = sitk.GetArrayFromImage(img)
    m = sitk.GetArrayFromImage(mask)
    z = int(np.argmax(m.sum(axis=(1, 2))))
    y = int(np.argmax(m.sum(axis=(0, 2))))
    x = int(np.argmax(m.sum(axis=(0, 1))))
    sp = img.GetSpacing()
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    views = [(a[z], m[z], sp[1] / sp[0], "axial"), (a[:, y], m[:, y], sp[2] / sp[0], "coronal"),
             (a[:, :, x], m[:, :, x], sp[2] / sp[1], "sagittal")]
    roi = a[m > 0]
    lo, hi = (np.percentile(roi, [1, 99]) if roi.size else np.percentile(a, [1, 99]))
    pad = (hi - lo) * 1.5
    for axi, (im, mm, asp, name) in zip(ax, views):
        axi.imshow(im, cmap="gray", vmin=lo - pad, vmax=hi + pad, aspect=asp, origin="upper" if name == "axial" else "lower")
        if mm.any():
            axi.contour(mm, levels=[0.5], colors="#e8590c", linewidths=1.2, origin="upper" if name == "axial" else "lower")
        axi.set_title(name)
        axi.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def _extract(extractor, img, mask) -> tuple[dict, dict]:
    res = extractor.execute(img, mask, label=1)
    feats, diag = {}, {}
    for k, v in res.items():
        if k.startswith("diagnostics_"):
            diag[k] = v if isinstance(v, (str, int, float)) else str(v)
        else:
            try:
                feats[k] = float(v)
            except (TypeError, ValueError):
                pass
    return feats, diag


def process_case(case: rio.PatientCase, cfg: dict, out_dir: str) -> dict:
    """Worker: returns {'id', 'features', 'qc', 'perturbed', 'error'}."""
    out = Path(out_dir)
    try:
        img, mask, qc = load_case(case, cfg)
        ext = build_extractor(cfg)
        feats, _ = _extract(ext, img, mask)
        perturbed = {}
        rob = cfg["features"].get("robustness", {})
        if rob.get("enabled"):
            for r in rob.get("perturbations", [-1, 1]):
                perturbed[int(r)] = _extract(ext, img, perturb_mask(mask, int(r)))[0]
        if cfg["preprocess"].get("qc_png", True):
            (out / "qc").mkdir(parents=True, exist_ok=True)
            qc_overlay(img, mask, out / "qc" / f"{case.patient_id}.png",
                       f"{case.patient_id}  {qc['modality']}  {qc['mask_volume_ml']:.1f} ml")
        if cfg["preprocess"].get("save_preprocessed", False):
            (out / "preprocessed").mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(out / "preprocessed" / f"{case.patient_id}_image.nii.gz"))
            sitk.WriteImage(mask, str(out / "preprocessed" / f"{case.patient_id}_mask.nii.gz"))
        return {"id": case.patient_id, "features": feats, "qc": qc, "perturbed": perturbed, "error": None}
    except Exception as e:
        return {"id": case.patient_id, "features": None, "qc": None, "perturbed": {},
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def icc_3_1(x: np.ndarray) -> float:
    """ICC(3,1), two-way mixed, consistency, single measurement. x: (subjects, raters)."""
    n, k = x.shape
    if n < 2:
        return np.nan
    gm = x.mean()
    ss_r = k * ((x.mean(1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(0) - gm) ** 2).sum()
    ss_e = ((x - gm) ** 2).sum() - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e
    return float((ms_r - ms_e) / den) if den > 0 else np.nan


def robustness_table(base: pd.DataFrame, perturbed: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    ids = base.index
    for col in base.columns:
        mats = [base[col].reindex(ids)] + [p[col].reindex(ids) for p in perturbed.values() if col in p]
        x = np.column_stack(mats).astype(float)
        x = x[~np.isnan(x).any(1)]
        rows.append({"feature": col, "icc": icc_3_1(x) if len(x) > 1 else np.nan})
    return pd.DataFrame(rows).set_index("feature")


def extract_all(cfg: dict, cases: list[rio.PatientCase] | None = None, resume: bool = True) -> pd.DataFrame:
    """Extract features for every discovered patient. Writes features.csv, qc.csv, failures.csv,
    and robustness_icc.csv (if enabled) into <output_dir>/features/."""
    from joblib import Parallel, delayed

    out = Path(cfg["output_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    feat_csv = out / "features.csv"
    cases = cases if cases is not None else rio.discover_patients(cfg)

    done = pd.DataFrame()
    if resume and feat_csv.exists() and not cfg["features"].get("robustness", {}).get("enabled"):
        done = pd.read_csv(feat_csv, index_col=0)
        done.index = done.index.astype(str)
        todo = [c for c in cases if c.patient_id not in set(done.index)]
        log.info("resume: %d already extracted, %d to go", len(done), len(todo))
    else:
        todo = cases

    results = Parallel(n_jobs=int(cfg.get("n_jobs", 1)), verbose=0)(
        delayed(process_case)(c, cfg, str(out)) for c in todo)

    ok = [r for r in results if r["error"] is None]
    bad = [r for r in results if r["error"] is not None]
    for r in bad:
        log.error("patient %s failed: %s", r["id"], r["error"])
    if bad:
        pd.DataFrame([{"id": r["id"], "error": r["error"]} for r in bad]).to_csv(out / "failures.csv", index=False)
        (out / "failures_traceback.txt").write_text("\n\n".join(f"== {r['id']}\n{r['trace']}" for r in bad))

    new = pd.DataFrame({r["id"]: r["features"] for r in ok}).T
    feats = pd.concat([done, new]) if len(done) else new
    feats.index.name = "PatientID"
    feats.to_csv(feat_csv)

    qc = pd.DataFrame({r["id"]: {k: json.dumps(v) if isinstance(v, list) else v for k, v in r["qc"].items()} for r in ok}).T
    if len(qc):
        qc_csv = out / "qc.csv"
        if resume and qc_csv.exists():
            old = pd.read_csv(qc_csv, index_col=0)
            old.index = old.index.astype(str)
            qc = pd.concat([old[~old.index.isin(qc.index)], qc])
        qc.index.name = "PatientID"
        qc.to_csv(qc_csv)

    if cfg["features"].get("robustness", {}).get("enabled") and ok:
        radii = sorted({k for r in ok for k in r["perturbed"]})
        pert = {k: pd.DataFrame({r["id"]: r["perturbed"][k] for r in ok if k in r["perturbed"]}).T for k in radii}
        icc = robustness_table(new, pert)
        icc.to_csv(out / "robustness_icc.csv")
        thr = cfg["features"]["robustness"].get("icc_threshold", 0.75)
        log.info("robustness: %d/%d features with ICC >= %.2f", int((icc["icc"] >= thr).sum()), len(icc), thr)

    log.info("features: %d patients x %d features (%d failed)", feats.shape[0], feats.shape[1], len(bad))
    return feats
