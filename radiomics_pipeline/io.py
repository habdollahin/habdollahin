"""Patient discovery, DICOM series reading and mask loading (NIfTI/NRRD, RTSTRUCT, DICOM SEG,
binary DICOM series). Every loader returns a SimpleITK image in its *native* geometry; masks are
brought onto the image grid later by :mod:`radiomics_pipeline.registration`.
"""
from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

log = logging.getLogger(__name__)

VOLUME_EXT = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".nhdr")
NON_IMAGE_MODALITIES = {"RTSTRUCT", "SEG", "RTDOSE", "RTPLAN", "RTRECORD", "SR", "PR", "KO", "REG"}


@dataclass
class DicomSeries:
    uid: str
    modality: str
    description: str
    files: list[Path] = field(default_factory=list)
    first: pydicom.Dataset | None = None


@dataclass
class PatientCase:
    patient_id: str
    folder: Path
    image_files: list[Path] | Path | None = None
    image_modality: str = ""
    mask_path: Path | list[Path] | None = None
    mask_format: str = "auto"
    mask_reference: Path | list[Path] | None = None
    header: pydicom.Dataset | None = None


# --------------------------------------------------------------------------- discovery
def is_volume_file(p: Path) -> bool:
    return p.name.lower().endswith(VOLUME_EXT)


def _read_header(p: Path) -> pydicom.Dataset | None:
    try:
        return pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
    except Exception:
        return None


def scan_dicom(folder: Path) -> tuple[dict[str, DicomSeries], list[tuple[Path, pydicom.Dataset]]]:
    """Group image DICOM files by SeriesInstanceUID; return non-image objects separately."""
    series: dict[str, DicomSeries] = {}
    others: list[tuple[Path, pydicom.Dataset]] = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or is_volume_file(p) or p.suffix.lower() in (".xlsx", ".csv", ".txt", ".json", ".png"):
            continue
        ds = _read_header(p)
        if ds is None or "SOPClassUID" not in ds and "Modality" not in ds:
            continue
        mod = str(getattr(ds, "Modality", "")).upper()
        if mod in NON_IMAGE_MODALITIES:
            others.append((p, ds))
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        s = series.get(uid)
        if s is None:
            s = series[uid] = DicomSeries(uid, mod, str(getattr(ds, "SeriesDescription", "")), [], ds)
        s.files.append(p)
    return series, others


def sort_slices(files: list[Path]) -> list[Path]:
    """Sort slice files along the slice normal (robust to missing InstanceNumber)."""
    keyed = []
    for f in files:
        ds = _read_header(f)
        try:
            iop = np.array(ds.ImageOrientationPatient, float)
            normal = np.cross(iop[:3], iop[3:])
            keyed.append((float(np.dot(normal, np.array(ds.ImagePositionPatient, float))), f))
        except Exception:
            keyed.append((float(getattr(ds, "InstanceNumber", 0) or 0), f))
    keyed.sort(key=lambda t: t[0])
    return [f for _, f in keyed]


def choose_series(series: dict[str, DicomSeries], modalities, desc_regex=None) -> DicomSeries | None:
    cands = [s for s in series.values() if not modalities or s.modality in modalities]
    if desc_regex:
        rx = re.compile(desc_regex, re.I)
        filt = [s for s in cands if rx.search(s.description)]
        if filt:
            cands = filt
        else:
            log.warning("no series matches description regex %r; falling back to largest", desc_regex)
    if not cands:
        return None
    return max(cands, key=lambda s: len(s.files))


def _match_any(name: str, patterns) -> bool:
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in patterns or [])


def discover_patients(cfg: dict) -> list[PatientCase]:
    d = cfg["data"]
    root = Path(d["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    folders = sorted(p for p in root.glob(d["patient_glob"]) if p.is_dir())
    rx = re.compile(d["patient_id_regex"]) if d.get("patient_id_regex") else None
    cases = []
    for folder in folders:
        pid = folder.name
        if rx:
            m = rx.search(pid)
            if not m:
                log.warning("skip %s: id regex did not match", folder)
                continue
            pid = m.group(1) if m.groups() else m.group(0)
        try:
            case = _discover_one(folder, pid, d)
        except Exception as e:  # keep going; report later
            log.error("discovery failed for %s: %s", folder, e)
            continue
        if case is not None:
            cases.append(case)
    log.info("discovered %d patients in %s", len(cases), root)
    return cases


def _discover_one(folder: Path, pid: str, d: dict) -> PatientCase | None:
    img_root = folder
    if d.get("image_subdir"):
        hits = [p for p in folder.glob(d["image_subdir"]) if p.is_dir()]
        if hits:
            img_root = hits[0]
    # mask candidates (by file/dir name)
    mask_pat = d["mask_pattern"]
    ref_pat = d.get("mask_reference_pattern")
    mask_files = [p for p in sorted(folder.rglob("*")) if _match_any(p.name, mask_pat)]
    ref_hits = [p for p in sorted(folder.rglob("*")) if ref_pat and _match_any(p.name, ref_pat)]

    series, others = scan_dicom(img_root)
    case = PatientCase(pid, folder, mask_format=d.get("mask_format", "auto"))

    # exclude series that live inside a mask/reference directory
    excl = {p.resolve() for p in mask_files + ref_hits if p.is_dir()}

    def in_excluded(f: Path) -> bool:
        return any(e in f.resolve().parents for e in excl)

    img_series = {u: s for u, s in series.items() if not all(in_excluded(f) for f in s.files)}
    chosen = choose_series(img_series, d.get("modalities"), d.get("series_description_regex"))
    if chosen is not None:
        case.image_files = sort_slices(chosen.files)
        case.image_modality = chosen.modality
        case.header = chosen.first
    else:
        vols = [p for p in sorted(img_root.rglob("*")) if is_volume_file(p) and p not in mask_files and p not in ref_hits]
        if not vols:
            log.warning("skip %s: no image series found", folder)
            return None
        case.image_files = vols[0]
        case.image_modality = "VOLUME"

    # mask: prefer explicit files matching the pattern, then RTSTRUCT/SEG found by header
    fmt = case.mask_format
    for p in mask_files:
        if p.is_dir():
            dcm = [f for f in p.rglob("*") if f.is_file() and _read_header(f) is not None]
            if dcm:
                case.mask_path, fmt = sort_slices(dcm), ("dicom_series" if fmt == "auto" else fmt)
                break
        elif is_volume_file(p):
            case.mask_path, fmt = p, ("nifti" if fmt == "auto" else fmt)
            break
        else:
            ds = _read_header(p)
            if ds is not None and str(getattr(ds, "Modality", "")).upper() in ("RTSTRUCT", "SEG"):
                case.mask_path = p
                fmt = str(ds.Modality).lower() if fmt == "auto" else fmt
                break
    if case.mask_path is None:
        for p, ds in others:
            mod = str(ds.Modality).upper()
            if mod in ("RTSTRUCT", "SEG"):
                case.mask_path, fmt = p, (mod.lower() if fmt == "auto" else fmt)
                break
    if case.mask_path is None:
        log.warning("skip %s: no mask found", folder)
        return None
    case.mask_format = fmt

    if ref_hits:
        r = ref_hits[0]
        case.mask_reference = (sort_slices([f for f in r.rglob("*") if f.is_file() and _read_header(f) is not None])
                               if r.is_dir() else r)
    return case


# --------------------------------------------------------------------------- reading images
def read_image(files: list[Path] | Path) -> sitk.Image:
    if isinstance(files, (str, Path)):
        img = sitk.ReadImage(str(files))
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in files])
        img = reader.Execute()
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    return sitk.Cast(img, sitk.sitkFloat32)


def _dicom_time(date: str, t: str) -> datetime:
    t = (t or "000000").split(".")[0].ljust(6, "0")
    return datetime.strptime(f"{date}{t}", "%Y%m%d%H%M%S")


def suv_factor(ds: pydicom.Dataset) -> float:
    """SUV body-weight factor (Bq/ml -> g/ml) with decay correction to scan start."""
    weight_g = float(ds.PatientWeight) * 1000.0
    rp = ds.RadiopharmaceuticalInformationSequence[0]
    dose = float(rp.RadionuclideTotalDose)
    half_life = float(rp.RadionuclideHalfLife)
    inj_date = getattr(rp, "RadiopharmaceuticalStartDateTime", None)
    if inj_date:
        inj = _dicom_time(str(inj_date)[:8], str(inj_date)[8:])
    else:
        inj = _dicom_time(ds.SeriesDate, rp.RadiopharmaceuticalStartTime)
    scan = _dicom_time(ds.SeriesDate, ds.SeriesTime)
    decay = np.exp(-np.log(2) * (scan - inj).total_seconds() / half_life)
    units = str(getattr(ds, "Units", "BQML")).upper()
    if units != "BQML":
        log.warning("PET units are %s, SUV conversion assumes BQML", units)
    return weight_g / (dose * decay)


# --------------------------------------------------------------------------- reading masks
def read_mask(case: PatientCase, reference: sitk.Image, roi_regex: str | None = None,
              label: int | None = None) -> sitk.Image:
    fmt = case.mask_format
    path = case.mask_path
    if fmt in ("nifti", "volume"):
        m = sitk.ReadImage(str(path))
    elif fmt == "dicom_series":
        m = read_image(path)
    elif fmt == "rtstruct":
        m = rtstruct_to_mask(Path(path), reference, roi_regex)
    elif fmt == "seg":
        m = dicom_seg_to_mask(Path(path), roi_regex)
    else:
        raise ValueError(f"unknown mask format {fmt!r}")
    if m.GetNumberOfComponentsPerPixel() > 1:
        m = sitk.VectorIndexSelectionCast(m, 0)
    arr = sitk.GetArrayFromImage(m)
    binm = (np.rint(arr) == label) if label is not None else (arr > 0.5 if arr.dtype.kind == "f" else arr > 0)
    out = sitk.GetImageFromArray(binm.astype(np.uint8))
    out.CopyInformation(m)
    return out


def _pick_roi(names: dict[int, str], roi_regex: str | None) -> int:
    if not names:
        raise ValueError("no ROIs/segments in mask object")
    if roi_regex:
        rx = re.compile(roi_regex, re.I)
        hits = [k for k, n in names.items() if rx.search(n)]
        if not hits:
            raise ValueError(f"no ROI matches {roi_regex!r}; available: {sorted(names.values())}")
        return hits[0]
    if len(names) > 1:
        log.warning("several ROIs %s and no roi_name_regex; using %r", list(names.values()), next(iter(names.values())))
    return next(iter(names))


def rtstruct_to_mask(path: Path, reference: sitk.Image, roi_regex: str | None) -> sitk.Image:
    """Rasterise closed planar RTSTRUCT contours onto the reference grid (even-odd rule for holes)."""
    from skimage.draw import polygon

    ds = pydicom.dcmread(str(path))
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    roi = _pick_roi(names, roi_regex)
    contours = None
    for rc in ds.ROIContourSequence:
        if int(rc.ReferencedROINumber) == roi:
            contours = getattr(rc, "ContourSequence", [])
            break
    if not contours:
        raise ValueError(f"ROI {names[roi]!r} has no contours")
    size = reference.GetSize()  # (x, y, z)
    mask = np.zeros(size[::-1], dtype=np.uint8)  # (z, y, x)
    for c in contours:
        if str(getattr(c, "ContourGeometricType", "CLOSED_PLANAR")).upper() not in ("CLOSED_PLANAR", "POINT", ""):
            continue
        pts = np.asarray(c.ContourData, float).reshape(-1, 3)
        idx = np.array([reference.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
        z = int(round(float(np.median(idx[:, 2]))))
        if not 0 <= z < size[2]:
            continue
        rr, cc = polygon(idx[:, 1], idx[:, 0], shape=(size[1], size[0]))
        sl = np.zeros((size[1], size[0]), dtype=np.uint8)
        sl[rr, cc] = 1
        mask[z] ^= sl  # XOR: inner contours carve holes
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(reference)
    log.debug("RTSTRUCT ROI %r rasterised: %d voxels", names[roi], int(mask.sum()))
    return out


def dicom_seg_to_mask(path: Path, roi_regex: str | None) -> sitk.Image:
    """Read one segment of a DICOM SEG object into a volume in the SEG's own frame geometry."""
    ds = pydicom.dcmread(str(path))
    names = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", s.SegmentNumber)) for s in ds.SegmentSequence}
    seg_no = _pick_roi(names, roi_regex)
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    shared = ds.SharedFunctionalGroupsSequence[0]
    pm = shared.PixelMeasuresSequence[0]
    iop = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    sel = []
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        if int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber) != seg_no:
            continue
        pos = np.array(fg.PlanePositionSequence[0].ImagePositionPatient, float)
        sel.append((float(np.dot(normal, pos)), pos, i))
    if not sel:
        raise ValueError("selected segment has no frames")
    sel.sort(key=lambda t: t[0])
    dists = np.array([s[0] for s in sel])
    dz = float(np.median(np.diff(dists))) if len(dists) > 1 else float(getattr(pm, "SliceThickness", 1.0))
    n = int(round((dists[-1] - dists[0]) / dz)) + 1 if len(dists) > 1 else 1
    vol = np.zeros((n, frames.shape[1], frames.shape[2]), dtype=np.uint8)
    for dist, _, i in sel:
        vol[int(round((dist - dists[0]) / dz))] |= (frames[i] > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(vol)
    ps = [float(v) for v in pm.PixelSpacing]
    img.SetSpacing((ps[1], ps[0], abs(dz)))
    img.SetOrigin(tuple(sel[0][1]))
    img.SetDirection(tuple(np.column_stack([iop[:3], iop[3:], normal]).ravel()))
    return img
