"""Clinical evaluation metrics for predicted dose distributions (all in Gy)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def dose_score(pred: np.ndarray, ref: np.ndarray, body: np.ndarray) -> float:
    """OpenKBP dose score: mean absolute error inside the body (Gy)."""
    m = body.astype(bool)
    return float(np.abs(pred[m] - ref[m]).mean()) if m.any() else float("nan")


def dvh_metrics(dose: np.ndarray, mask: np.ndarray, voxel_cc: float, is_target: bool) -> Dict[str, float]:
    v = dose[mask.astype(bool)]
    if v.size == 0:
        return {}
    if is_target:
        return {"D99": float(np.percentile(v, 1)), "D95": float(np.percentile(v, 5)),
                "D1": float(np.percentile(v, 99)), "Dmean": float(v.mean())}
    n01 = max(1, int(round(0.1 / voxel_cc)))  # voxels in 0.1 cc
    d01 = float(np.sort(v)[-min(n01, v.size)])
    return {"Dmean": float(v.mean()), "D0.1cc": d01}


def dvh_score(pred, ref, structures: Dict[str, np.ndarray], targets: List[str], voxel_cc: float):
    """OpenKBP DVH score: mean |ΔDVH-metric| over all structures/metrics."""
    errs, table = [], {}
    for name, m in structures.items():
        is_t = name in targets
        a = dvh_metrics(pred, m, voxel_cc, is_t)
        b = dvh_metrics(ref, m, voxel_cc, is_t)
        for k in a:
            table[f"{name}/{k}"] = {"pred": a[k], "ref": b[k], "diff": a[k] - b[k]}
            errs.append(abs(a[k] - b[k]))
    return (float(np.mean(errs)) if errs else float("nan")), table


def gamma_pass_rate(pred: np.ndarray, ref: np.ndarray, spacing, dd: float = 0.03, dta: float = 3.0,
                    threshold: float = 0.1, mask: Optional[np.ndarray] = None) -> float:
    """Global 3-D gamma pass rate (brute-force search on the voxel grid).

    ``dd`` is the dose-difference criterion as a fraction of max(ref), ``dta``
    the distance-to-agreement in mm. Evaluation is restricted to voxels with
    ref dose ≥ ``threshold`` × max(ref) (and to ``mask`` if given). Search is
    on the voxel lattice only (no sub-voxel interpolation), which slightly
    underestimates the pass rate for coarse grids.
    """
    sx, sy, sz = (float(s) for s in spacing)
    dmax = float(ref.max())
    if dmax <= 0:
        return float("nan")
    eval_m = ref >= threshold * dmax
    if mask is not None:
        eval_m &= mask.astype(bool)
    if not eval_m.any():
        return float("nan")
    rz, ry, rx = (int(np.ceil(dta / s)) for s in (sz, sy, sx))
    best = np.full(ref.shape, np.inf, dtype=np.float32)
    dtol = dd * dmax
    pad = np.pad(pred, ((rz, rz), (ry, ry), (rx, rx)), mode="edge")
    Z, Y, X = ref.shape
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                dist2 = (dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2
                if dist2 > dta ** 2:
                    continue
                shifted = pad[rz + dz: rz + dz + Z, ry + dy: ry + dy + Y, rx + dx: rx + dx + X]
                g2 = ((shifted - ref) / dtol) ** 2 + dist2 / dta ** 2
                np.minimum(best, g2, out=best)
    return float((best[eval_m] <= 1.0).mean())


def evaluate_case(pred_gy, ref_gy, sample: dict) -> Dict:
    """All metrics for one preprocessed sample (dict from ``load_sample``)."""
    geom = sample["geometry"]
    structures = {"PTV": sample["ptv"]}
    for name, m in zip(sample["oar_names"], sample["oars"]):
        if m.any():
            structures[name] = m
    ds = dose_score(pred_gy, ref_gy, sample["body"])
    dvh, table = dvh_score(pred_gy, ref_gy, structures, ["PTV"], geom.voxel_volume_cc)
    gamma = gamma_pass_rate(pred_gy, ref_gy, geom.spacing, mask=sample["body"])
    return {"dose_score": ds, "dvh_score": dvh, "gamma_3%3mm": gamma, "dvh": table}
