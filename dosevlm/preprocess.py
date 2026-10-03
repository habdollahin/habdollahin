"""Convert one patient's DICOM folder (CT + RTSTRUCT + RTPLAN [+ RTDOSE]) into
a compact, model-ready ``.npz`` sample on a fixed, axis-aligned grid."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from . import dicom_io
from .geometry import Geometry, make_target_grid, resample
from .rtplan import (aperture_projection, control_point_tokens, load_rtplan,
                     plan_scalar_features, plan_to_text)
from .structures import DEFAULT_OARS, body_from_ct, classify

log = logging.getLogger(__name__)


def preprocess_patient(patient_dir: str, out_path: Optional[str] = None,
                       spacing=(3.0, 3.0, 3.0), shape=(128, 128, 128),
                       oars: Dict[str, List[str]] = None, site: Optional[str] = None,
                       max_cp_tokens: int = 128, require_dose: bool = True) -> dict:
    oars = oars or DEFAULT_OARS
    files = dicom_io.scan_patient_dir(patient_dir)
    if not files.rtstruct or not files.rtplan:
        raise FileNotFoundError(f"{patient_dir}: RTSTRUCT and RTPLAN are required")

    ct, ct_geom, ct_ds = dicom_io.load_ct(files.ct)
    plan = load_rtplan(files.rtplan[0])
    raw_masks = dicom_io.load_rtstruct(files.rtstruct[0], ct_geom)
    body, targets, oar_masks = classify(raw_masks, plan.prescription_gy, oars)
    if not targets:
        raise ValueError(f"{patient_dir}: no PTV structure found in RTSTRUCT "
                         f"(names: {sorted(raw_masks)})")

    rx = plan.prescription_gy or max(d for _, d, _ in targets)
    if not rx:
        raise ValueError(f"{patient_dir}: prescription dose unknown (no RTPLAN DoseReference and no "
                         f"dose in PTV names)")
    targets = [(n, d if d > 0 else rx, m) for n, d, m in targets]

    # ---- target grid: centred on the PTV union centroid ----------------------
    union = np.zeros(ct_geom.shape, dtype=bool)
    for _, _, m in targets:
        union |= m
    kji = np.argwhere(union).mean(axis=0)
    center = ct_geom.index_to_world(kji[::-1])
    grid = make_target_grid(center, spacing, shape)

    def rs_mask(m):
        return resample(m.astype(np.float32), ct_geom, grid, order=1) > 0.5

    ct_r = resample(ct, ct_geom, grid, order=1, cval=-1000.0)
    body_r = rs_mask(body) if body is not None else body_from_ct(ct_r)
    ptv_dose = np.zeros(grid.shape, dtype=np.float32)
    ptv_union = np.zeros(grid.shape, dtype=bool)
    for _, d, m in targets:
        mr = rs_mask(m)
        ptv_union |= mr
        ptv_dose = np.maximum(ptv_dose, mr * (d / rx))
    oar_names = list(oars.keys())
    oar_r = np.zeros((len(oar_names),) + grid.shape, dtype=bool)
    for i, name in enumerate(oar_names):
        if name in oar_masks:
            oar_r[i] = rs_mask(oar_masks[name])

    beam = aperture_projection(plan, grid, body_mask=body_r)

    dose_r = None
    if files.rtdose:
        dpaths = dicom_io.pick_plan_dose(files.rtdose, plan.sop_uid)
        if dpaths:
            dose, dose_geom, _ = dicom_io.load_rtdose_sum(dpaths)
            dose_r = resample(dose, dose_geom, grid, order=1, cval=0.0)
    if dose_r is None and require_dose:
        raise FileNotFoundError(f"{patient_dir}: no RTDOSE found for plan {plan.sop_uid}")

    present = [f"{n} ({d:.1f} Gy)" for n, d, _ in targets] + [n for n in oar_names if n in oar_masks]
    text = plan_to_text(plan, present, site=site)

    sample = dict(
        ct=ct_r.astype(np.float16), body=body_r, ptv=ptv_union, ptv_dose=ptv_dose.astype(np.float16),
        oars=oar_r, beam=beam.astype(np.float16), rx=np.float32(rx),
        cp_tokens=control_point_tokens(plan, max_cp_tokens), scalars=plan_scalar_features(plan),
        text=np.array(text), oar_names=np.array(oar_names),
        geometry=np.array(json.dumps(grid.to_dict())),
        meta=np.array(json.dumps(dict(
            patient_dir=os.path.abspath(patient_dir),
            patient_id=str(getattr(ct_ds, "PatientID", "")),
            patient_name=str(getattr(ct_ds, "PatientName", "")),
            study_uid=str(getattr(ct_ds, "StudyInstanceUID", "")),
            frame_of_reference_uid=str(getattr(ct_ds, "FrameOfReferenceUID", "")),
            plan_uid=plan.sop_uid, plan_label=plan.label,
            targets=[(n, d) for n, d, _ in targets],
        ))),
    )
    if dose_r is not None:
        sample["dose"] = dose_r.astype(np.float32)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(out_path, **sample)
    return sample


def load_sample(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    s["text"] = str(s["text"])
    s["oar_names"] = [str(n) for n in s["oar_names"]]
    s["geometry"] = Geometry.from_dict(json.loads(str(s["geometry"])))
    s["meta"] = json.loads(str(s["meta"]))
    return s
