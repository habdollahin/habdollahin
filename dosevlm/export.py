"""Write a predicted dose distribution as a DICOM RT Dose object."""
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .geometry import Geometry

RT_DOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
RT_PLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"


def write_rtdose(dose_gy: np.ndarray, geom: Geometry, path: str, meta: Optional[dict] = None,
                 reference_ds=None, description: str = "DoseVLM predicted dose") -> str:
    """Save ``dose_gy`` (z, y, x) on ``geom`` as RTDOSE.

    ``reference_ds`` (e.g. a CT slice) supplies patient / study identifiers; if
    absent, they are taken from ``meta`` (as written by the preprocessor).
    The output is flagged as a prediction in SeriesDescription and must not be
    used for treatment without review.
    """
    meta = meta or {}
    dose = np.clip(np.asarray(dose_gy, dtype=np.float64), 0, None)
    # DS values are limited to 16 characters; round the scaling up so the
    # maximum dose still fits in uint32 after the string round-trip.
    scaling_str = f"{max(float(dose.max()), 1e-6) / (2 ** 32 - 1) * 1.000001:.8e}"
    scaling = float(scaling_str)
    pixels = np.clip(np.round(dose / scaling), 0, 2 ** 32 - 1).astype(np.uint32)

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = RT_DOSE_STORAGE
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=fm, preamble=b"\0" * 128)

    now = _dt.datetime.now()
    ds.SOPClassUID = RT_DOSE_STORAGE
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "DoseVLM"
    ds.SeriesDescription = description
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 999

    if reference_ds is not None:
        for tag in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID",
                    "StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName",
                    "FrameOfReferenceUID"):
            if hasattr(reference_ds, tag):
                setattr(ds, tag, getattr(reference_ds, tag))
    else:
        ds.PatientName = meta.get("patient_name", "")
        ds.PatientID = meta.get("patient_id", "")
        ds.StudyInstanceUID = meta.get("study_uid") or generate_uid()
        ds.FrameOfReferenceUID = meta.get("frame_of_reference_uid") or generate_uid()
    for tag in ("PatientBirthDate", "PatientSex", "StudyDate", "StudyTime", "StudyID",
                "AccessionNumber", "ReferringPhysicianName"):
        if not hasattr(ds, tag):
            setattr(ds, tag, "")
    ds.PositionReferenceIndicator = ""
    ds.InstanceNumber = 1

    nz, ny, nx = dose.shape
    d = geom.direction
    ds.ImagePositionPatient = [float(v) for v in geom.origin]
    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([d[:, 0], d[:, 1]])]
    ds.PixelSpacing = [float(geom.spacing[1]), float(geom.spacing[0])]
    ds.SliceThickness = float(geom.spacing[2])
    ds.GridFrameOffsetVector = [float(k * geom.spacing[2]) for k in range(nz)]
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)
    ds.NumberOfFrames = nz
    ds.Rows, ds.Columns = ny, nx
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = scaling_str
    if meta.get("plan_uid"):
        ref = pydicom.Dataset()
        ref.ReferencedSOPClassUID = RT_PLAN_STORAGE
        ref.ReferencedSOPInstanceUID = meta["plan_uid"]
        ds.ReferencedRTPlanSequence = Sequence([ref])
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path
