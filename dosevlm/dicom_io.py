"""Readers for CT, RTSTRUCT, RTPLAN and RTDOSE DICOM objects.

The readers are deliberately dependency-light (pydicom + numpy + scipy +
scikit-image) and return plain numpy arrays together with a
:class:`~dosevlm.geometry.Geometry` describing where they live in patient
space.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pydicom
from skimage.draw import polygon as draw_polygon

from .geometry import Geometry, resample

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Directory scanning
# --------------------------------------------------------------------------- #
@dataclass
class PatientFiles:
    ct: List[str] = field(default_factory=list)
    rtstruct: List[str] = field(default_factory=list)
    rtplan: List[str] = field(default_factory=list)
    rtdose: List[str] = field(default_factory=list)


def scan_patient_dir(path: str) -> PatientFiles:
    """Walk ``path`` and sort DICOM files by modality.

    If several CT series are present, the one referenced by the RTSTRUCT is
    kept (falling back to the series with the most slices).
    """
    series: Dict[str, List[str]] = defaultdict(list)
    files = PatientFiles()
    for root, _, names in os.walk(path):
        for name in sorted(names):
            fp = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:  # not DICOM
                continue
            modality = str(getattr(ds, "Modality", "")).upper()
            if modality == "CT":
                series[str(getattr(ds, "SeriesInstanceUID", ""))].append(fp)
            elif modality == "RTSTRUCT":
                files.rtstruct.append(fp)
            elif modality == "RTPLAN":
                files.rtplan.append(fp)
            elif modality == "RTDOSE":
                files.rtdose.append(fp)
    if not series:
        raise FileNotFoundError(f"No CT images found under {path}")

    chosen = None
    if files.rtstruct:
        ref = _referenced_ct_series(pydicom.dcmread(files.rtstruct[0], stop_before_pixels=True))
        if ref in series:
            chosen = ref
    if chosen is None:
        chosen = max(series, key=lambda s: len(series[s]))
    files.ct = series[chosen]
    return files


def _referenced_ct_series(rs) -> Optional[str]:
    try:
        study = rs.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        return str(study.RTReferencedSeriesSequence[0].SeriesInstanceUID)
    except (AttributeError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# CT
# --------------------------------------------------------------------------- #
def load_ct(files: List[str]):
    """Load a CT series -> (hu volume [z,y,x] float32, Geometry, first dataset)."""
    slices = [pydicom.dcmread(f) for f in files]
    iop = np.array(slices[0].ImageOrientationPatient, dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda s: float(np.dot(normal, np.array(s.ImagePositionPatient, float))))

    positions = np.array([s.ImagePositionPatient for s in slices], dtype=np.float64)
    if len(slices) > 1:
        dz = np.diff(positions @ normal)
        if np.ptp(dz) > 1e-2:
            log.warning("Non-uniform CT slice spacing (%.3f-%.3f mm); using median.", dz.min(), dz.max())
        sz = float(np.median(dz))
    else:
        sz = float(getattr(slices[0], "SliceThickness", 1.0))
    sy, sx = (float(v) for v in slices[0].PixelSpacing)  # row spacing, column spacing

    vol = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
        + float(getattr(s, "RescaleIntercept", 0.0))
        for s in slices
    ])
    geom = Geometry(positions[0], (sx, sy, sz), vol.shape,
                    np.stack([row_dir, col_dir, normal], axis=1))
    return vol, geom, slices[0]


# --------------------------------------------------------------------------- #
# RTSTRUCT
# --------------------------------------------------------------------------- #
def load_rtstruct(path: str, geom: Geometry, names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Rasterise RTSTRUCT contours onto ``geom``.

    Contours on the same slice are combined with XOR, which correctly handles
    holes / islands as written by all major TPS vendors.
    """
    ds = pydicom.dcmread(path)
    roi_names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    masks: Dict[str, np.ndarray] = {}
    nz, ny, nx = geom.shape
    for rc in getattr(ds, "ROIContourSequence", []):
        name = roi_names.get(int(rc.ReferencedROINumber), f"ROI_{rc.ReferencedROINumber}")
        if names is not None and name not in names:
            continue
        mask = np.zeros(geom.shape, dtype=bool)
        for c in getattr(rc, "ContourSequence", []):
            pts = np.asarray(c.ContourData, dtype=np.float64).reshape(-1, 3)
            if len(pts) < 3:
                continue
            ijk = geom.world_to_index(pts)
            k = int(round(float(np.median(ijk[:, 2]))))
            if not 0 <= k < nz:
                continue
            rr, cc = draw_polygon(ijk[:, 1], ijk[:, 0], shape=(ny, nx))
            sl = np.zeros((ny, nx), dtype=bool)
            sl[rr, cc] = True
            mask[k] ^= sl
        if mask.any():
            masks[name] = masks[name] | mask if name in masks else mask
    return masks


# --------------------------------------------------------------------------- #
# RTDOSE
# --------------------------------------------------------------------------- #
def load_rtdose(path: str):
    """Return (dose [z,y,x] in Gy, Geometry, dataset)."""
    ds = pydicom.dcmread(path)
    dose = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    if dose.ndim == 2:
        dose = dose[None]
    iop = np.array(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
    offsets = np.array(getattr(ds, "GridFrameOffsetVector", [0.0]), dtype=np.float64)
    # GridFrameOffsetVector is relative when its first element is 0, otherwise
    # it holds absolute positions along the slice normal (PS3.3 C.8.8.3.2).
    if offsets[0] != 0:
        offsets = offsets - float(np.dot(ipp, normal))
    order = np.argsort(offsets)
    if not np.all(order == np.arange(len(order))):
        dose, offsets = dose[order], offsets[order]
    origin = ipp + offsets[0] * normal
    sz = float(np.median(np.diff(offsets))) if len(offsets) > 1 else 1.0
    sy, sx = (float(v) for v in ds.PixelSpacing)
    geom = Geometry(origin, (sx, sy, sz), dose.shape, np.stack([row_dir, col_dir, normal], axis=1))
    return dose, geom, ds


def load_rtdose_sum(paths: List[str]):
    """Sum several RTDOSE objects (e.g. per-beam doses) on the first one's grid."""
    dose, geom, ds = load_rtdose(paths[0])
    if str(getattr(ds, "DoseSummationType", "PLAN")).upper() == "PLAN" or len(paths) == 1:
        return dose, geom, ds
    for p in paths[1:]:
        d, g, _ = load_rtdose(p)
        dose = dose + resample(d, g, geom)
    return dose, geom, ds


def pick_plan_dose(dose_paths: List[str], plan_uid: Optional[str]) -> List[str]:
    """Choose the RTDOSE file(s) belonging to ``plan_uid`` (prefer PLAN sums)."""
    cands = []
    for p in dose_paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        ref = None
        try:
            ref = str(ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)
        except (AttributeError, IndexError):
            pass
        if plan_uid is None or ref is None or ref == plan_uid:
            cands.append((str(getattr(ds, "DoseSummationType", "")).upper(), p))
    plan_level = [p for t, p in cands if t == "PLAN"]
    return plan_level[:1] if plan_level else [p for _, p in cands]
