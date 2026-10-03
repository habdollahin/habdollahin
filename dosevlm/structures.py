"""Mapping of clinical (free-text) structure names onto canonical channels."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

# Canonical OAR channels -> regex aliases (case-insensitive, matched on a
# normalised name with spaces/hyphens replaced by underscores). Edit or extend
# via the ``oars`` section of the YAML config.
DEFAULT_OARS: Dict[str, List[str]] = {
    "SpinalCord": [r"^(spinal_?)?cord", r"^spinal_?canal", r"^sc$"],
    "Brainstem": [r"^brain_?stem"],
    "Parotid_L": [r"^parotid_?(l|lt|left)$", r"^(l|lt|left)_?parotid$"],
    "Parotid_R": [r"^parotid_?(r|rt|right)$", r"^(r|rt|right)_?parotid$"],
    "Mandible": [r"^mandible", r"^jaw$"],
    "Larynx": [r"^larynx", r"^glottis"],
    "Esophagus": [r"^(o)?esophagus"],
    "OralCavity": [r"^oral_?cavity"],
    "Lung_L": [r"^lung_?(l|lt|left)$", r"^(l|lt|left)_?lung$"],
    "Lung_R": [r"^lung_?(r|rt|right)$", r"^(r|rt|right)_?lung$"],
    "Heart": [r"^heart"],
    "Liver": [r"^liver"],
    "Kidneys": [r"^kidney", r"^(l|r|lt|rt)_?kidney"],
    "Bowel": [r"^(small_?)?bowel", r"^bowel_?bag"],
    "Rectum": [r"^rectum"],
    "Bladder": [r"^bladder"],
    "FemoralHeads": [r"^femoral_?head", r"^femur", r"^fem_?head"],
}
BODY_ALIASES = [r"^body", r"^external", r"^outer_?contour", r"^skin$", r"^patient$"]
TARGET_ALIASES = [r"^ptv"]


def normalise(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def match(name: str, patterns: List[str]) -> bool:
    n = normalise(name)
    return any(re.search(p, n) for p in patterns)


def target_dose_from_name(name: str, rx: Optional[float]) -> Optional[float]:
    """Parse ``PTV_70``, ``PTV7000``, ``PTV_54.0Gy``, ``PTV_High`` -> Gy."""
    nums = re.findall(r"\d+(?:[.,]\d+)?", name)
    for s in nums:
        v = float(s.replace(",", "."))
        if v > 200:  # cGy
            v /= 100.0
        if 10.0 <= v <= 120.0:
            return v
    if rx is not None and re.search(r"high|boost", name, re.I):
        return rx
    return None


def body_from_ct(hu: np.ndarray, threshold: float = -400.0) -> np.ndarray:
    m = hu > threshold
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    for k in range(m.shape[0]):
        m[k] = ndimage.binary_fill_holes(m[k])
    return m


def classify(masks: Dict[str, np.ndarray], rx: Optional[float],
             oars: Dict[str, List[str]] = None):
    """Split raw masks into body / targets(with dose) / canonical OARs."""
    oars = oars or DEFAULT_OARS
    body = None
    targets: List[Tuple[str, float, np.ndarray]] = []
    oar_masks: Dict[str, np.ndarray] = {}
    for name, m in masks.items():
        if body is None and match(name, BODY_ALIASES):
            body = m
            continue
        if match(name, TARGET_ALIASES):
            if re.search(r"(eval|opt|ring|minus|[-_]oar|_crop)", name, re.I):
                continue  # derived optimisation structures
            d = target_dose_from_name(name, rx)
            targets.append((name, d if d is not None else (rx or 0.0), m))
            continue
        for canon, pats in oars.items():
            if match(name, pats):
                oar_masks[canon] = oar_masks[canon] | m if canon in oar_masks else m
                break
    # If no dose could be determined and no prescription is known, use relative levels.
    return body, targets, oar_masks
