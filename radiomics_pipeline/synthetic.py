"""Synthetic DICOM cohort for smoke tests / demos: CT series per patient, a lesion whose
texture, size and intensity depend on the class, masks as NIfTI (on a different grid, to
exercise resampling) or as RTSTRUCT, plus an Excel label file."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid
from scipy import ndimage


def _base(meta_sop, sop_uid, pid):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = meta_sop
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta_sop
    ds.SOPInstanceUID = sop_uid
    ds.PatientID = pid
    ds.PatientName = pid
    ds.StudyDate = ds.SeriesDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = ds.SeriesTime = "120000"
    return ds


def make_volume(label: int, rng, shape=(40, 64, 64)):
    z, y, x = np.indices(shape)
    c = np.array(shape) / 2 + rng.uniform(-4, 4, 3)
    r = rng.uniform(6, 11) + 0.8 * label
    ell = ((z - c[0]) / (r * 0.6)) ** 2 + ((y - c[1]) / r) ** 2 + ((x - c[2]) / (r * rng.uniform(0.8, 1.2))) ** 2
    mask = ell <= 1
    body = ((y - 32) / 30) ** 2 + ((x - 32) / 28) ** 2 <= 1
    img = np.where(body, 40.0, -1000.0) + rng.normal(0, 12, shape)
    # lesion: class 1 is brighter and more heterogeneous (coarser texture)
    tex = ndimage.gaussian_filter(rng.normal(0, 1, shape), rng.uniform(0.8, 1.6) + 0.35 * label)
    tex /= tex.std() + 1e-8
    img[mask] = 60 + 4 * label + rng.normal(0, 6) + (rng.uniform(10, 22) + 4 * label) * tex[mask]
    return img.astype(np.float32), mask.astype(np.uint8)


def write_ct_series(img, folder: Path, pid, spacing=(0.9, 0.9, 2.5), origin=(-30.0, -30.0, -50.0)):
    folder.mkdir(parents=True, exist_ok=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    uids = []
    for k in range(img.shape[0]):
        uid = generate_uid()
        uids.append(uid)
        ds = _base(CTImageStorage, uid, pid)
        ds.Modality = "CT"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.SeriesDescription = "synthetic CT"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + k * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.Rows, ds.Columns = img.shape[1], img.shape[2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
        ds.PixelData = np.clip(np.rint(img[k] + 1024), -32768, 32767).astype(np.int16).tobytes()
        # shuffle file names to test slice sorting
        ds.save_as(str(folder / f"IM_{(k * 7919) % 1000:04d}_{k}.dcm"), enforce_file_format=True)
    return {"study": study, "series": series, "frame": frame, "uids": uids, "spacing": spacing, "origin": origin}


def write_rtstruct(mask, info, path: Path, pid, roi="GTV"):
    from skimage import measure

    ds = _base(RTStructureSetStorage, generate_uid(), pid)
    ds.Modality = "RTSTRUCT"
    ds.StudyInstanceUID, ds.SeriesInstanceUID = info["study"], generate_uid()
    ds.FrameOfReferenceUID = info["frame"]
    roi_item = pydicom.Dataset()
    roi_item.ROINumber, roi_item.ROIName, roi_item.ReferencedFrameOfReferenceUID = 1, roi, info["frame"]
    other = pydicom.Dataset()
    other.ROINumber, other.ROIName, other.ReferencedFrameOfReferenceUID = 2, "BODY", info["frame"]
    ds.StructureSetROISequence = [other, roi_item]
    sx, sy, sz = info["spacing"]
    ox, oy, oz = info["origin"]
    contours = []
    for k in range(mask.shape[0]):
        if not mask[k].any():
            continue
        for cnt in measure.find_contours(np.pad(mask[k], 1).astype(float), 0.5):
            cnt = cnt - 1
            pts = np.column_stack([ox + cnt[:, 1] * sx, oy + cnt[:, 0] * sy, np.full(len(cnt), oz + k * sz)])
            c = pydicom.Dataset()
            c.ContourGeometricType = "CLOSED_PLANAR"
            c.NumberOfContourPoints = len(pts)
            c.ContourData = [f"{v:.3f}" for v in pts.ravel()]
            contours.append(c)
    rc = pydicom.Dataset()
    rc.ReferencedROINumber = 1
    rc.ContourSequence = contours
    empty = pydicom.Dataset()
    empty.ReferencedROINumber = 2
    ds.ROIContourSequence = [empty, rc]
    ds.save_as(str(path), enforce_file_format=True)


def make_cohort(out: str | Path, n: int = 40, seed: int = 0, rtstruct_every: int = 3) -> Path:
    out = Path(out)
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2) + [0] * (n % 2))
    rng.shuffle(labels)
    rows = []
    for i, lab in enumerate(labels):
        pid = f"PAT{i + 1:03d}"
        # ~10% label noise keeps the task realistically imperfect
        img, mask = make_volume(int(lab) if rng.random() > 0.1 else 1 - int(lab), rng)
        info = write_ct_series(img, out / pid / "CT", pid)
        if rtstruct_every and i % rtstruct_every == 0:
            write_rtstruct(mask, info, out / pid / f"RS_{pid}.dcm", pid)
        else:
            m = sitk.GetImageFromArray(mask)
            m.SetSpacing(info["spacing"])
            m.SetOrigin(info["origin"])
            # store half of the NIfTI masks on a finer grid -> exercises mask resampling
            if i % 2:
                from .preprocess import resample

                m = sitk.Cast(resample(sitk.Cast(m, sitk.sitkFloat32), (0.45, 0.45, 1.25), "linear", 0.0) >= 0.5,
                              sitk.sitkUInt8)
            sitk.WriteImage(m, str(out / pid / f"{pid}_mask.nii.gz"))
        rows.append({"Patient ID": f"pat{i + 1:03d}" if i % 4 == 0 else pid,  # case mismatch on purpose
                     "Outcome": "positive" if lab else "negative",
                     "Age": int(rng.normal(62, 9) + 3 * lab), "Sex": rng.choice(["M", "F"])})
    pd.DataFrame(rows).to_excel(out / "labels.xlsx", index=False)
    return out
