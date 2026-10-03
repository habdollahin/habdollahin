"""Image preprocessing: clipping, bias-field correction, denoising, resampling, intensity
normalisation and mask clean-up. Order follows IBSI recommendations
(clip -> N4 -> denoise -> resample -> normalise; discretisation happens in feature extraction).
"""
from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "lanczos": sitk.sitkLanczosWindowedSinc,
    "gaussian": sitk.sitkGaussian,
}


# --------------------------------------------------------------------------- intensity
def clip(img: sitk.Image, lo: float, hi: float) -> sitk.Image:
    return sitk.Clamp(img, sitk.sitkFloat32, float(lo), float(hi))


def n4_bias_correction(img: sitk.Image, shrink: int = 4, iterations=(50, 50, 30)) -> sitk.Image:
    """N4ITK bias-field correction (MR). Fitted on a shrunk image, applied at full resolution."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    head = sitk.OtsuThreshold(img, 0, 1, 200)
    small = sitk.Shrink(img, [shrink] * img.GetDimension())
    small_mask = sitk.Shrink(head, [shrink] * img.GetDimension())
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations(list(iterations))
    corr.Execute(small, small_mask)
    log_bias = corr.GetLogBiasFieldAsImage(img)
    return sitk.Cast(img / sitk.Exp(log_bias), sitk.sitkFloat32)


def denoise(img: sitk.Image, cfg: dict) -> sitk.Image:
    method = (cfg or {}).get("method", "none").lower()
    img = sitk.Cast(img, sitk.sitkFloat32)
    if method == "none":
        return img
    if method == "gaussian":
        return sitk.SmoothingRecursiveGaussian(img, float(cfg.get("sigma", 0.5)))
    if method == "median":
        return sitk.Median(img, [int(cfg.get("radius", 1))] * img.GetDimension())
    if method == "bilateral":
        return sitk.Bilateral(img, float(cfg.get("domain_sigma", 1.0)), float(cfg.get("range_sigma", 25.0)))
    if method == "curvature_flow":
        return sitk.CurvatureFlow(img, timeStep=float(cfg.get("time_step", 0.0625)),
                                  numberOfIterations=int(cfg.get("iterations", 5)))
    if method == "anisotropic":
        return sitk.CurvatureAnisotropicDiffusion(img, timeStep=float(cfg.get("time_step", 0.0625)),
                                                  conductanceParameter=float(cfg.get("conductance", 3.0)),
                                                  numberOfIterations=int(cfg.get("iterations", 5)))
    if method == "patch_based":  # non-local-means style; slow
        f = sitk.PatchBasedDenoisingImageFilter()
        f.SetKernelBandwidthSigma(float(cfg.get("sigma", 400.0)))
        f.SetPatchRadius(int(cfg.get("patch_radius", 2)))
        f.SetNumberOfIterations(int(cfg.get("iterations", 1)))
        return f.Execute(img)
    raise ValueError(f"unknown denoise method {method!r}")


def normalize(img: sitk.Image, mask: sitk.Image | None, cfg: dict) -> sitk.Image:
    method = (cfg or {}).get("method", "none").lower()
    if method == "none":
        return img
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if method == "zscore_roi" and mask is not None:
        ref = arr[sitk.GetArrayFromImage(mask) > 0]
    elif cfg.get("foreground_only", False):
        ref = arr[arr > np.percentile(arr, 5)]
    else:
        ref = arr.ravel()
    if ref.size == 0:
        ref = arr.ravel()
    if method in ("zscore", "zscore_roi"):
        out = (arr - ref.mean()) / (ref.std() + 1e-8) * float(cfg.get("scale", 1.0))
        if cfg.get("shift") is not None:
            out = out + float(cfg["shift"])
    elif method == "minmax":
        out = (arr - ref.min()) / (ref.max() - ref.min() + 1e-8) * float(cfg.get("scale", 1.0))
    elif method == "percentile":
        lo, hi = np.percentile(ref, cfg.get("percentiles", [1, 99]))
        out = (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-8) * float(cfg.get("scale", 1.0))
    elif method == "window":
        lo, hi = cfg["window"]
        out = (np.clip(arr, lo, hi) - lo) / (hi - lo) * float(cfg.get("scale", 1.0))
    else:
        raise ValueError(f"unknown normalisation {method!r}")
    res = sitk.GetImageFromArray(out.astype(np.float32))
    res.CopyInformation(img)
    return res


# --------------------------------------------------------------------------- geometry
def resample(img: sitk.Image, spacing, interpolator="bspline", default=None, reference: sitk.Image | None = None) -> sitk.Image:
    """Resample to new spacing (keeping physical extent) or onto a reference grid."""
    interp = INTERPOLATORS[interpolator] if isinstance(interpolator, str) else interpolator
    if default is None:
        default = float(sitk.GetArrayViewFromImage(img).min())
    if reference is not None:
        return sitk.Resample(img, reference, sitk.Transform(), interp, default, img.GetPixelID())
    spacing = [float(s) if s else o for s, o in zip(spacing, img.GetSpacing())]
    osp = np.array(img.GetSpacing())
    size = [max(1, int(round(sz * o / n))) for sz, o, n in zip(img.GetSize(), osp, spacing)]
    # IBSI grid alignment: keep the centre of the image extent fixed
    shift = 0.5 * ((np.array(img.GetSize()) - 1) * osp - (np.array(size) - 1) * np.array(spacing))
    D = np.array(img.GetDirection()).reshape(3, 3)
    origin = tuple(np.array(img.GetOrigin()) + D @ shift)
    return sitk.Resample(img, size, sitk.Transform(), interp, origin, spacing, img.GetDirection(),
                         default, img.GetPixelID())


def resample_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary mask onto a grid: linear interpolation of the label + 0.5 threshold
    (smoother and less biased than nearest neighbour when the spacing changes)."""
    m = sitk.Resample(sitk.Cast(mask, sitk.sitkFloat32), reference, sitk.Transform(), sitk.sitkLinear, 0.0)
    return sitk.Cast(m >= 0.5, sitk.sitkUInt8)


def clean_mask(mask: sitk.Image, cfg: dict) -> sitk.Image:
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    if cfg.get("fill_holes", True):
        mask = sitk.BinaryFillhole(mask)
    if cfg.get("keep_largest_component", False):
        cc = sitk.ConnectedComponent(mask)
        cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
        mask = sitk.Cast(cc == 1, sitk.sitkUInt8)
    return mask


def perturb_mask(mask: sitk.Image, radius: int) -> sitk.Image:
    """Erode (radius<0) or dilate (radius>0) by |radius| voxels - used for robustness analysis."""
    if radius == 0:
        return mask
    f = sitk.BinaryDilate if radius > 0 else sitk.BinaryErode
    out = f(mask, [abs(int(radius))] * mask.GetDimension())
    if sitk.GetArrayViewFromImage(out).sum() == 0:  # never erode the ROI away
        return mask
    return sitk.Cast(out, sitk.sitkUInt8)


# --------------------------------------------------------------------------- full chain
def preprocess_image(img: sitk.Image, cfg: dict, modality: str = "") -> sitk.Image:
    """Steps that do not need the mask (performed before mask alignment)."""
    p = cfg["preprocess"]
    if p.get("clip_range"):
        img = clip(img, *p["clip_range"])
    if p.get("n4_bias_correction"):
        if modality not in ("MR", "VOLUME", ""):
            log.warning("N4 bias correction requested for modality %s", modality)
        img = n4_bias_correction(img)
    img = denoise(img, p.get("denoise"))
    r = p.get("resample", {})
    if r.get("enabled", True) and r.get("spacing"):
        img = resample(img, r["spacing"], r.get("interpolator", "bspline"))
    return img
