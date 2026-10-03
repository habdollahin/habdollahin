"""Synthetic CT / RTSTRUCT / RTPLAN / RTDOSE generator.

Produces small, self-consistent DICOM-RT "patients" (an elliptical water
phantom with a bony spine, PTV(s) and OARs, a two-arc VMAT plan with conformal
MLC apertures, and a crude dose distribution) for smoke-testing the pipeline.
The dose model is *not* physically accurate - it only needs to be a learnable
function of the inputs.
"""
from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy import ndimage
from skimage import measure

from .export import write_rtdose
from .geometry import Geometry, make_target_grid, resample
from .rtplan import aperture_projection, load_rtplan

CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
RTPLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"


def _base(sop_class, modality, common):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = modality
    for k, v in common.items():
        setattr(ds, k, v)
    ds.SeriesInstanceUID = generate_uid()
    return ds


def make_patient(out_dir: str, seed: int = 0, n_xy: int = 96, n_z: int = 48, px: float = 4.0,
                 dz: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    today = _dt.date.today().strftime("%Y%m%d")
    common = dict(PatientName=f"Synthetic^{seed:03d}", PatientID=f"SYN{seed:03d}", PatientBirthDate="",
                  PatientSex="O", StudyInstanceUID=generate_uid(), StudyDate=today, StudyTime="120000",
                  StudyID="1", AccessionNumber="", ReferringPhysicianName="",
                  FrameOfReferenceUID=generate_uid())

    # ---------------- phantom ----------------------------------------------
    origin = np.array([-(n_xy - 1) * px / 2, -(n_xy - 1) * px / 2, -(n_z - 1) * dz / 2])
    geom = Geometry(origin, (px, px, dz), (n_z, n_xy, n_xy))
    w = geom.world_grid()
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    a, b = rng.uniform(140, 175), rng.uniform(95, 120)
    body = (x / a) ** 2 + (y / b) ** 2 <= 1.0
    hu = np.where(body, rng.normal(20, 15, body.shape), -1000.0)
    spine_c = np.array([0.0, b - 45.0])
    spine = ((x - spine_c[0]) ** 2 + (y - spine_c[1]) ** 2) <= 18.0 ** 2
    cord = ((x - spine_c[0]) ** 2 + (y - spine_c[1]) ** 2) <= 7.0 ** 2
    hu = np.where(spine & ~cord, 700.0, hu)

    # PTV: sphere/ellipsoid, offset from centre
    ptv_c = np.array([rng.uniform(-50, 50), rng.uniform(-40, 10), rng.uniform(-20, 20)])
    r = rng.uniform(18, 35)
    ptv = ((x - ptv_c[0]) ** 2 + (y - ptv_c[1]) ** 2 + ((z - ptv_c[2]) / 1.2) ** 2) <= r ** 2
    ptv &= body
    rx, fx = [(70.0, 35), (60.0, 30), (50.0, 25), (66.0, 33)][rng.integers(4)]
    two_levels = rng.random() < 0.5
    ptv_low = None
    if two_levels:
        r2 = r + rng.uniform(10, 18)
        ptv_low = (((x - ptv_c[0]) ** 2 + (y - ptv_c[1]) ** 2 + ((z - ptv_c[2]) / 1.2) ** 2) <= r2 ** 2) & body
        low_dose = round(rx * 0.8)
    par_l = ((x - 75) ** 2 + (y + 5) ** 2 + (z - 10) ** 2) <= 22 ** 2
    par_r = ((x + 75) ** 2 + (y + 5) ** 2 + (z - 10) ** 2) <= 22 ** 2

    # ---------------- CT ------------------------------------------------------
    ct_dir = os.path.join(out_dir, "CT")
    os.makedirs(ct_dir, exist_ok=True)
    series_uid = generate_uid()
    ct_first = None
    for k in range(n_z):
        ds = _base(CT_STORAGE, "CT", common)
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2] + k * dz)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [px, px]
        ds.SliceThickness = dz
        ds.Rows = ds.Columns = n_xy
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PositionReferenceIndicator = ""
        ds.PixelData = np.clip(hu[k] + 1024, 0, 4095).astype(np.uint16).tobytes()
        ds.save_as(os.path.join(ct_dir, f"CT_{k:03d}.dcm"), enforce_file_format=True)
        if ct_first is None:
            ct_first = ds

    # ---------------- RTSTRUCT -----------------------------------------------
    ptv_name = rng.choice([f"PTV_{rx:.0f}", f"PTV{rx * 100:.0f}", f"PTV_{rx:.0f}Gy"])
    structs = [("BODY", body, "EXTERNAL"), (ptv_name, ptv, "PTV"),
               (rng.choice(["SpinalCord", "Cord", "Spinal Cord"]), cord, "ORGAN"),
               (rng.choice(["Parotid_L", "L Parotid", "Parotid Lt"]), par_l & body, "ORGAN"),
               (rng.choice(["Parotid_R", "R Parotid", "Parotid Rt"]), par_r & body, "ORGAN")]
    if two_levels:
        structs.append((f"PTV_{low_dose:.0f}", ptv_low, "PTV"))
    rs = _base(RTSTRUCT_STORAGE, "RTSTRUCT", common)
    rs.StructureSetLabel = "SYN"
    rs.StructureSetDate = today
    rs.StructureSetTime = "120000"
    frs = Dataset()
    frs.FrameOfReferenceUID = common["FrameOfReferenceUID"]
    st = Dataset()
    st.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    st.ReferencedSOPInstanceUID = common["StudyInstanceUID"]
    se = Dataset()
    se.SeriesInstanceUID = series_uid
    st.RTReferencedSeriesSequence = Sequence([se])
    frs.RTReferencedStudySequence = Sequence([st])
    rs.ReferencedFrameOfReferenceSequence = Sequence([frs])
    ssroi, roic, obs = [], [], []
    for n, (name, m, typ) in enumerate(structs, start=1):
        d = Dataset()
        d.ROINumber = n
        d.ReferencedFrameOfReferenceUID = common["FrameOfReferenceUID"]
        d.ROIName = str(name)
        d.ROIGenerationAlgorithm = "MANUAL"
        ssroi.append(d)
        rc = Dataset()
        rc.ReferencedROINumber = n
        rc.ROIDisplayColor = [255, 0, 0]
        contours = []
        for k in range(n_z):
            if not m[k].any():
                continue
            for c in measure.find_contours(np.pad(m[k], 1).astype(float), 0.5):
                c = c - 1  # undo pad
                ijk = np.stack([c[:, 1], c[:, 0], np.full(len(c), k)], axis=1)
                pts = geom.index_to_world(ijk)
                cs = Dataset()
                cs.ContourGeometricType = "CLOSED_PLANAR"
                cs.NumberOfContourPoints = len(pts)
                cs.ContourData = [round(float(v), 3) for v in pts.reshape(-1)]
                contours.append(cs)
        rc.ContourSequence = Sequence(contours)
        roic.append(rc)
        o = Dataset()
        o.ObservationNumber = n
        o.ReferencedROINumber = n
        o.RTROIInterpretedType = typ
        o.ROIInterpreter = ""
        obs.append(o)
    rs.StructureSetROISequence = Sequence(ssroi)
    rs.ROIContourSequence = Sequence(roic)
    rs.RTROIObservationsSequence = Sequence(obs)
    rs.save_as(os.path.join(out_dir, "RS.dcm"), enforce_file_format=True)

    # ---------------- RTPLAN --------------------------------------------------
    plan = _base(RTPLAN_STORAGE, "RTPLAN", common)
    plan.RTPlanLabel = "VMAT_SYN"
    plan.RTPlanName = "VMAT"
    plan.RTPlanDate = today
    plan.RTPlanTime = "120000"
    plan.RTPlanGeometry = "PATIENT"
    dr = Dataset()
    dr.DoseReferenceNumber = 1
    dr.DoseReferenceStructureType = "SITE"
    dr.DoseReferenceType = "TARGET"
    dr.TargetPrescriptionDose = rx
    plan.DoseReferenceSequence = Sequence([dr])
    ps = Dataset()
    ps.PatientPosition = "HFS"
    ps.PatientSetupNumber = 1
    plan.PatientSetupSequence = Sequence([ps])
    iso = [round(float(v), 2) for v in ptv_c]
    leaf_bounds = np.linspace(-200, 200, 41)
    leaf_centres = 0.5 * (leaf_bounds[1:] + leaf_bounds[:-1])
    radius_iso = (r2 if two_levels else r) + 3.0
    n_arcs = int(rng.integers(1, 3)) + (1 if rng.random() < 0.3 else 0)
    beams, refs = [], []
    energy = float(rng.choice([6.0, 10.0]))
    for bi in range(n_arcs):
        cw = bi % 2 == 0
        start, stop = (181.0, 179.0) if cw else (179.0, 181.0)
        angles = (start + (np.arange(0, 359, 8) if cw else -np.arange(0, 359, 8))) % 360.0
        angles = np.append(angles, stop)
        mu = float(rng.uniform(1.8, 3.2) * rx / fx * 100 / n_arcs)
        weights = rng.uniform(0.5, 1.5, len(angles))
        weights[0] = 0
        cumw = np.cumsum(weights) / weights.sum()
        beam = Dataset()
        beam.BeamNumber = bi + 1
        beam.BeamName = f"Arc{bi + 1}"
        beam.BeamType = "DYNAMIC"
        beam.RadiationType = "PHOTON"
        beam.TreatmentDeliveryType = "TREATMENT"
        beam.TreatmentMachineName = "SynLinac"
        beam.SourceAxisDistance = 1000.0
        beam.NumberOfWedges = 0
        beam.NumberOfCompensators = 0
        beam.NumberOfBoli = 0
        beam.NumberOfBlocks = 0
        beam.FinalCumulativeMetersetWeight = 1.0
        bld = []
        for t, n in (("ASYMX", 2), ("ASYMY", 2)):
            d = Dataset()
            d.RTBeamLimitingDeviceType = t
            d.NumberOfLeafJawPairs = 1
            bld.append(d)
        mlc = Dataset()
        mlc.RTBeamLimitingDeviceType = "MLCX"
        mlc.NumberOfLeafJawPairs = 40
        mlc.LeafPositionBoundaries = [float(v) for v in leaf_bounds]
        bld.append(mlc)
        beam.BeamLimitingDeviceSequence = Sequence(bld)
        coll = 30.0 if cw else 330.0
        cps = []
        for ci, (g, cw_) in enumerate(zip(angles, cumw)):
            cp = Dataset()
            cp.ControlPointIndex = ci
            cp.CumulativeMetersetWeight = float(cw_)
            cp.GantryAngle = float(g)
            if ci == 0:
                cp.GantryRotationDirection = "CW" if cw else "CC"
                cp.BeamLimitingDeviceAngle = coll
                cp.PatientSupportAngle = 0.0
                cp.IsocenterPosition = iso
                cp.NominalBeamEnergy = energy
            half = np.sqrt(np.clip(radius_iso ** 2 - leaf_centres ** 2, 0, None))
            jitter = rng.normal(0, 2.0, (2, 40))
            open_ = half > 0
            a_pos = np.where(open_, -half + jitter[0], -1.0)
            b_pos = np.where(open_, half + jitter[1], 1.0)
            b_pos = np.maximum(b_pos, a_pos + 0.5)
            pos = []
            for t, v in (("ASYMX", [-radius_iso - 5, radius_iso + 5]), ("ASYMY", [-radius_iso - 5, radius_iso + 5]),
                         ("MLCX", list(a_pos) + list(b_pos))):
                d = Dataset()
                d.RTBeamLimitingDeviceType = t
                d.LeafJawPositions = [round(float(q), 2) for q in v]
                pos.append(d)
            cp.BeamLimitingDevicePositionSequence = Sequence(pos)
            cps.append(cp)
        beam.NumberOfControlPoints = len(cps)
        beam.ControlPointSequence = Sequence(cps)
        beams.append(beam)
        rb = Dataset()
        rb.ReferencedBeamNumber = bi + 1
        rb.BeamMeterset = round(mu, 2)
        refs.append(rb)
    plan.BeamSequence = Sequence(beams)
    fg = Dataset()
    fg.FractionGroupNumber = 1
    fg.NumberOfFractionsPlanned = fx
    fg.NumberOfBeams = n_arcs
    fg.ReferencedBeamSequence = Sequence(refs)
    plan.FractionGroupSequence = Sequence([fg])
    plan_path = os.path.join(out_dir, "RP.dcm")
    plan.save_as(plan_path, enforce_file_format=True)

    # ---------------- "ground-truth" dose -------------------------------------
    p = load_rtplan(plan_path)
    fluence = aperture_projection(p, geom, body_mask=body)
    # crude depth attenuation from the body surface + scatter blur
    depth = ndimage.distance_transform_edt(body, sampling=(dz, px, px))
    dose = fluence * np.exp(-0.004 * depth)
    dose = ndimage.gaussian_filter(dose, sigma=(6 / dz, 6 / px, 6 / px))
    dose *= body
    dose = dose / max(np.median(dose[ptv]), 1e-6) * rx
    if two_levels:
        # emulate optimiser pulling the low-dose PTV down to its prescription
        shell = ptv_low & ~ndimage.binary_dilation(ptv, iterations=1)
        dose = np.where(shell, np.minimum(dose, low_dose * rng.uniform(1.0, 1.05)), dose)
        dose = ndimage.gaussian_filter(dose, sigma=1.0) * body
    # spare the cord a little (as an optimiser would)
    dose = np.where(cord, dose * 0.7, dose)
    # write on a coarser, offset dose grid to exercise resampling
    dgrid = make_target_grid(np.array([3.0, -2.0, 1.0]), (5.0, 5.0, 5.0),
                             (int(n_z * dz / 5) - 2, int(n_xy * px / 5) - 2, int(n_xy * px / 5) - 2))
    dose_dg = resample(dose, geom, dgrid, order=1)
    write_rtdose(dose_dg, dgrid, os.path.join(out_dir, "RD.dcm"),
                 meta={"plan_uid": str(plan.SOPInstanceUID)}, reference_ds=ct_first,
                 description="Synthetic reference dose")
    return {"rx": rx, "fractions": fx, "ptv": ptv_name, "arcs": n_arcs}
