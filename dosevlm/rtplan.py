"""RTPLAN parsing and plan-derived model inputs.

Three representations of the VMAT plan are produced for the model:

1. ``plan_to_text``   - a natural-language description (the *language* input).
2. ``control_point_tokens`` - a numeric token per (sub-sampled) control point
   describing gantry/collimator/couch angle, MU weight and MLC aperture.
3. ``aperture_projection`` - a 3-D "beam's-eye-view fluence" image obtained by
   projecting every control point's MLC/jaw aperture through the patient from
   the source. This gives the vision branch explicit arc geometry.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pydicom

from .geometry import Geometry

log = logging.getLogger(__name__)

SAD_DEFAULT = 1000.0


@dataclass
class ControlPoint:
    gantry: float
    collimator: float
    couch: float
    cum_weight: float
    jaw_x: tuple = (-200.0, 200.0)
    jaw_y: tuple = (-200.0, 200.0)
    mlc_a: Optional[np.ndarray] = None  # bank A (negative X side) leaf positions
    mlc_b: Optional[np.ndarray] = None  # bank B (positive X side)
    mu: float = 0.0  # MU delivered in the segment ending at this CP


@dataclass
class Beam:
    number: int
    name: str
    beam_type: str
    radiation_type: str
    energy: float
    machine: str
    sad: float
    isocenter: np.ndarray
    meterset: float
    gantry_start: float
    gantry_stop: float
    rotation: str
    collimator: float
    couch: float
    leaf_boundaries: Optional[np.ndarray]
    control_points: List[ControlPoint] = field(default_factory=list)

    @property
    def is_arc(self) -> bool:
        return self.rotation in ("CW", "CC")

    @property
    def arc_length(self) -> float:
        if not self.is_arc:
            return 0.0
        d = (self.gantry_stop - self.gantry_start) % 360.0
        return d if self.rotation == "CW" else (360.0 - d) % 360.0


@dataclass
class Plan:
    sop_uid: str
    label: str
    name: str
    patient_position: str
    prescription_gy: Optional[float]
    fractions: Optional[int]
    target_doses: List[float]
    beams: List[Beam]

    @property
    def total_mu(self) -> float:
        return float(sum(b.meterset for b in self.beams))

    @property
    def dose_per_fraction(self) -> Optional[float]:
        if self.prescription_gy and self.fractions:
            return self.prescription_gy / self.fractions
        return None


# --------------------------------------------------------------------------- #
def _get(ds, name, default=None):
    return getattr(ds, name, default) if ds is not None and hasattr(ds, name) else default


def load_rtplan(path: str) -> Plan:
    ds = pydicom.dcmread(path)

    # --- prescription -------------------------------------------------------
    target_doses = []
    for dr in _get(ds, "DoseReferenceSequence", []) or []:
        for key in ("TargetPrescriptionDose", "DeliveryMaximumDose", "TargetMaximumDose"):
            v = _get(dr, key)
            if v is not None and str(_get(dr, "DoseReferenceType", "TARGET")).upper() == "TARGET":
                target_doses.append(float(v))
                break
    fractions, beam_mu = None, {}
    fgs = _get(ds, "FractionGroupSequence", []) or []
    if fgs:
        fg = fgs[0]
        fractions = int(_get(fg, "NumberOfFractionsPlanned", 0) or 0) or None
        for rb in _get(fg, "ReferencedBeamSequence", []) or []:
            mu = _get(rb, "BeamMeterset")
            if mu is not None:
                beam_mu[int(rb.ReferencedBeamNumber)] = float(mu)
    rx = max(target_doses) if target_doses else None

    pp = "HFS"
    pss = _get(ds, "PatientSetupSequence", []) or []
    if pss:
        pp = str(_get(pss[0], "PatientPosition", "HFS") or "HFS")

    # --- beams ----------------------------------------------------------------
    beams: List[Beam] = []
    for b in _get(ds, "BeamSequence", []) or []:
        if str(_get(b, "TreatmentDeliveryType", "TREATMENT")).upper() != "TREATMENT":
            continue
        cps = list(_get(b, "ControlPointSequence", []) or [])
        if not cps:
            continue
        cp0 = cps[0]
        leaf_bounds = None
        for bld in _get(b, "BeamLimitingDeviceSequence", []) or []:
            if "MLC" in str(bld.RTBeamLimitingDeviceType).upper() and hasattr(bld, "LeafPositionBoundaries"):
                leaf_bounds = np.asarray(bld.LeafPositionBoundaries, dtype=np.float64)
        number = int(b.BeamNumber)
        meterset = beam_mu.get(number, 0.0)
        final_weight = float(_get(b, "FinalCumulativeMetersetWeight", 1.0) or 1.0)

        # Carry forward values that are only specified when they change.
        state = dict(
            gantry=float(_get(cp0, "GantryAngle", 0.0)),
            collimator=float(_get(cp0, "BeamLimitingDeviceAngle", 0.0)),
            couch=float(_get(cp0, "PatientSupportAngle", 0.0)),
            jaw_x=(-200.0, 200.0), jaw_y=(-200.0, 200.0), mlc_a=None, mlc_b=None,
        )
        rotation = str(_get(cp0, "GantryRotationDirection", "NONE") or "NONE").upper()
        iso = np.asarray(_get(cp0, "IsocenterPosition", [0, 0, 0]), dtype=np.float64)
        energy = float(_get(cp0, "NominalBeamEnergy", 6.0) or 6.0)
        cplist: List[ControlPoint] = []
        prev_w = 0.0
        for cp in cps:
            for key, attr in (("gantry", "GantryAngle"), ("collimator", "BeamLimitingDeviceAngle"),
                              ("couch", "PatientSupportAngle")):
                if hasattr(cp, attr):
                    state[key] = float(getattr(cp, attr))
            for bldp in _get(cp, "BeamLimitingDevicePositionSequence", []) or []:
                t = str(bldp.RTBeamLimitingDeviceType).upper()
                pos = np.asarray(bldp.LeafJawPositions, dtype=np.float64)
                if t in ("X", "ASYMX"):
                    state["jaw_x"] = (pos[0], pos[1])
                elif t in ("Y", "ASYMY"):
                    state["jaw_y"] = (pos[0], pos[1])
                elif "MLC" in t:
                    n = len(pos) // 2
                    if t == "MLCY":  # rare; treat as X-moving for projection purposes
                        log.warning("MLCY leaves are approximated as MLCX.")
                    state["mlc_a"], state["mlc_b"] = pos[:n], pos[n:]
            w = float(_get(cp, "CumulativeMetersetWeight", prev_w) or 0.0)
            cplist.append(ControlPoint(
                gantry=state["gantry"], collimator=state["collimator"], couch=state["couch"],
                cum_weight=w, jaw_x=state["jaw_x"], jaw_y=state["jaw_y"],
                mlc_a=state["mlc_a"], mlc_b=state["mlc_b"],
                mu=meterset * max(w - prev_w, 0.0) / final_weight,
            ))
            prev_w = w

        beams.append(Beam(
            number=number, name=str(_get(b, "BeamName", f"Beam{number}")),
            beam_type=str(_get(b, "BeamType", "")), radiation_type=str(_get(b, "RadiationType", "PHOTON")),
            energy=energy, machine=str(_get(b, "TreatmentMachineName", "")),
            sad=float(_get(b, "SourceAxisDistance", SAD_DEFAULT) or SAD_DEFAULT), isocenter=iso,
            meterset=meterset, gantry_start=cplist[0].gantry, gantry_stop=cplist[-1].gantry,
            rotation=rotation, collimator=cplist[0].collimator, couch=cplist[0].couch,
            leaf_boundaries=leaf_bounds, control_points=cplist,
        ))

    return Plan(
        sop_uid=str(ds.SOPInstanceUID), label=str(_get(ds, "RTPlanLabel", "")),
        name=str(_get(ds, "RTPlanName", "")), patient_position=pp, prescription_gy=rx,
        fractions=fractions, target_doses=sorted(set(target_doses), reverse=True), beams=beams,
    )


# --------------------------------------------------------------------------- #
# Aperture helpers
# --------------------------------------------------------------------------- #
def aperture_area_cm2(cp: ControlPoint, leaf_bounds: Optional[np.ndarray]) -> float:
    x1, x2 = cp.jaw_x
    y1, y2 = cp.jaw_y
    if cp.mlc_a is None or leaf_bounds is None:
        return max(x2 - x1, 0) * max(y2 - y1, 0) / 100.0
    lo = np.clip(leaf_bounds[:-1], y1, y2)
    hi = np.clip(leaf_bounds[1:], y1, y2)
    width = np.clip(hi - lo, 0, None)
    gap = np.clip(np.minimum(cp.mlc_b, x2) - np.maximum(cp.mlc_a, x1), 0, None)
    return float(np.sum(width * gap)) / 100.0


# --------------------------------------------------------------------------- #
# Natural-language description (language branch)
# --------------------------------------------------------------------------- #
def plan_to_text(plan: Plan, structure_names: Optional[List[str]] = None, site: Optional[str] = None) -> str:
    arcs = [b for b in plan.beams if b.is_arc]
    statics = [b for b in plan.beams if not b.is_arc]
    parts = []
    tech = "VMAT" if arcs else "IMRT"
    parts.append(f"{tech} radiotherapy plan" + (f" for {site} cancer" if site else "") + ".")
    if plan.prescription_gy:
        s = f"Prescription {plan.prescription_gy:.1f} Gy"
        if plan.fractions:
            s += f" in {plan.fractions} fractions ({plan.dose_per_fraction:.2f} Gy per fraction)"
        parts.append(s + ".")
    if len(plan.target_doses) > 1:
        parts.append("Dose levels " + ", ".join(f"{d:.1f} Gy" for d in plan.target_doses) + ".")
    parts.append(f"Patient position {plan.patient_position}.")
    if arcs:
        parts.append(f"{len(arcs)} arcs.")
    for b in arcs:
        parts.append(
            f"Arc {b.name}: {b.radiation_type.lower()} {b.energy:g} MV, gantry {b.gantry_start:.0f} to "
            f"{b.gantry_stop:.0f} {'clockwise' if b.rotation == 'CW' else 'counter-clockwise'}, "
            f"{b.arc_length:.0f} degrees, collimator {b.collimator:.0f}, couch {b.couch:.0f}, "
            f"{b.meterset:.0f} MU, {len(b.control_points)} control points."
        )
    for b in statics:
        parts.append(f"Static field {b.name}: {b.energy:g} MV, gantry {b.gantry_start:.0f}, "
                     f"couch {b.couch:.0f}, {b.meterset:.0f} MU.")
    parts.append(f"Total {plan.total_mu:.0f} MU.")
    if structure_names:
        parts.append("Structures: " + ", ".join(structure_names) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Numeric control-point tokens (plan-sequence branch)
# --------------------------------------------------------------------------- #
CP_FEATURES = 12


def control_point_tokens(plan: Plan, max_tokens: int = 128) -> np.ndarray:
    """Return an array (max_tokens, CP_FEATURES), zero padded; last column is a validity flag."""
    rows = []
    total_mu = max(plan.total_mu, 1e-6)
    for bi, b in enumerate(plan.beams):
        for ci, cp in enumerate(b.control_points):
            g, c, t = (math.radians(a) for a in (cp.gantry, cp.collimator, cp.couch))
            rows.append([
                math.sin(g), math.cos(g), math.sin(c), math.cos(c), math.sin(t), math.cos(t),
                cp.mu / total_mu * len(b.control_points),  # ~1 for uniform weighting
                aperture_area_cm2(cp, b.leaf_boundaries) / 100.0,
                b.energy / 10.0, bi / max(len(plan.beams), 1), ci / max(len(b.control_points) - 1, 1),
                1.0,
            ])
    arr = np.zeros((max_tokens, CP_FEATURES), dtype=np.float32)
    if rows:
        rows = np.asarray(rows, dtype=np.float32)
        if len(rows) > max_tokens:
            # Merge consecutive control points: keep geometry of a representative
            # CP, but sum the MU of the merged group so no dose is lost.
            edges = np.linspace(0, len(rows), max_tokens + 1).astype(int)
            merged = []
            for a, e in zip(edges[:-1], edges[1:]):
                grp = rows[a:e]
                r = grp[len(grp) // 2].copy()
                r[6] = grp[:, 6].mean()
                r[7] = grp[:, 7].mean()
                merged.append(r)
            rows = np.stack(merged)
        arr[: len(rows)] = rows
    return arr


def plan_scalar_features(plan: Plan) -> np.ndarray:
    """Global numeric plan descriptors (normalised to ~unit range)."""
    arcs = [b for b in plan.beams if b.is_arc]
    energies = [b.energy for b in plan.beams] or [6.0]
    return np.array([
        (plan.prescription_gy or 0.0) / 70.0,
        (plan.fractions or 0) / 35.0,
        (plan.dose_per_fraction or 0.0) / 2.0,
        plan.total_mu / 1000.0,
        len(arcs) / 4.0,
        sum(b.arc_length for b in arcs) / 720.0,
        float(np.mean(energies)) / 10.0,
        len(plan.target_doses) / 3.0,
        (min(plan.target_doses) / max(plan.target_doses)) if plan.target_doses else 1.0,
        float(any(abs(b.couch) > 1 for b in plan.beams)),
    ], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 3-D aperture projection (vision branch)
# --------------------------------------------------------------------------- #
def _patient_to_iec_fixed(p: np.ndarray, position: str) -> np.ndarray:
    """DICOM patient (LPS) offsets -> IEC 61217 fixed-system offsets."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    position = position.upper()
    if position == "HFP":
        return np.stack([-x, z, y], axis=-1)
    if position == "FFS":
        return np.stack([-x, -z, -y], axis=-1)
    if position == "FFP":
        return np.stack([x, -z, y], axis=-1)
    if position != "HFS":
        log.warning("Patient position %s not supported; assuming HFS.", position)
    return np.stack([x, z, -y], axis=-1)


def aperture_projection(plan: Plan, geom: Geometry, max_cps_per_beam: int = 36,
                        body_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """MU-weighted, inverse-square-corrected sum of projected apertures.

    For every (sub-sampled) control point each voxel is tested for lying inside
    the jaw-and-MLC defined aperture as seen from the source. The result is
    normalised to a maximum of 1. It is a purely geometric fluence proxy: no
    attenuation or scatter is modelled (the network learns those).
    """
    world = geom.world_grid().reshape(-1, 3)
    out = np.zeros(world.shape[0], dtype=np.float32)
    if body_mask is not None:
        sel = body_mask.reshape(-1).astype(bool)
    else:
        sel = np.ones(world.shape[0], dtype=bool)
    if not sel.any():
        return out.reshape(geom.shape)
    pts_w = world[sel]

    for b in plan.beams:
        cps = b.control_points
        if not cps:
            continue
        rel = _patient_to_iec_fixed(pts_w - b.isocenter, plan.patient_position)
        # For arcs the MU of segment i is delivered between CP i-1 and CP i; use
        # the segment mid-angle and the aperture of CP i-1 (VMAT convention).
        segs = []
        if len(cps) == 1:
            segs = [(cps[0], cps[0].gantry, max(b.meterset, 1.0))]
        else:
            for i in range(1, len(cps)):
                g0, g1 = cps[i - 1].gantry, cps[i].gantry
                dg = ((g1 - g0 + 180.0) % 360.0) - 180.0
                if cps[i].mu > 0:
                    segs.append((cps[i - 1], g0 + dg / 2.0, cps[i].mu))
            if not segs:  # static IMRT step-and-shoot with zero weights?
                segs = [(cps[0], cps[0].gantry, max(b.meterset, 1.0))]
        if len(segs) > max_cps_per_beam:
            edges = np.linspace(0, len(segs), max_cps_per_beam + 1).astype(int)
            merged = []
            for a, e in zip(edges[:-1], edges[1:]):
                grp = segs[a:e]
                cp, ang, _ = grp[len(grp) // 2]
                merged.append((cp, ang, sum(s[2] for s in grp)))
            segs = merged

        for cp, gantry, mu in segs:
            t = math.radians(cp.couch)
            # Undo couch rotation (about IEC Z) to get gantry-frame-independent coords.
            ct_, st_ = math.cos(t), math.sin(t)
            px = ct_ * rel[:, 0] - st_ * rel[:, 1]
            py = st_ * rel[:, 0] + ct_ * rel[:, 1]
            pz = rel[:, 2]
            g = math.radians(gantry)
            cg, sg = math.cos(g), math.sin(g)
            xg = cg * px - sg * pz
            zg = sg * px + cg * pz  # towards source
            yg = py
            denom = np.maximum(b.sad - zg, 1.0)
            scale = b.sad / denom
            xi, yi = xg * scale, yg * scale
            c = math.radians(cp.collimator)
            cc, sc = math.cos(c), math.sin(c)
            xb = cc * xi + sc * yi
            yb = -sc * xi + cc * yi
            inside = (xb >= cp.jaw_x[0]) & (xb <= cp.jaw_x[1]) & (yb >= cp.jaw_y[0]) & (yb <= cp.jaw_y[1])
            if cp.mlc_a is not None and b.leaf_boundaries is not None:
                lb = b.leaf_boundaries
                idx = np.searchsorted(lb, yb) - 1
                valid = (idx >= 0) & (idx < len(cp.mlc_a))
                idx_c = np.clip(idx, 0, len(cp.mlc_a) - 1)
                inside &= valid & (xb >= cp.mlc_a[idx_c]) & (xb <= cp.mlc_b[idx_c])
            contrib = np.where(inside, mu * scale ** 2, 0.0).astype(np.float32)
            out[np.flatnonzero(sel)] += contrib
    m = out.max()
    if m > 0:
        out /= m
    return out.reshape(geom.shape)
