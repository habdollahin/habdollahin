"""Bring a mask onto the (preprocessed) image grid.

* Same geometry -> used as is.
* Same voxel grid but slightly different header (rounding / missing origin) -> header copied.
* Different grid, same frame of reference -> physical-space resampling (linear + 0.5 threshold).
* ``mask_reference`` given (the image the mask was delineated on, e.g. T2 when features come from
  ADC, or a planning CT vs a follow-up CT) -> intensity-based registration of that reference
  image to the target image (rigid / affine / B-spline, multi-resolution, Mattes MI), and the
  resulting transform is applied to the mask.
"""
from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

from .preprocess import resample_mask

log = logging.getLogger(__name__)


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol * 100)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def _metric(reg: sitk.ImageRegistrationMethod, name: str):
    if name == "mattes":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif name == "correlation":
        reg.SetMetricAsCorrelation()
    elif name == "meansquares":
        reg.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"unknown metric {name!r}")


def register(fixed: sitk.Image, moving: sitk.Image, method: str = "rigid", metric: str = "mattes",
             iterations: int = 200, seed: int = 42) -> sitk.Transform:
    """Multi-resolution registration of ``moving`` onto ``fixed``. Returns moving->fixed transform
    (usable directly with sitk.Resample(moving, fixed, tx))."""
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    if method == "rigid":
        base = sitk.Euler3DTransform()
    elif method in ("affine", "bspline"):
        base = sitk.AffineTransform(3)
    else:
        raise ValueError(f"unknown registration method {method!r}")
    init = sitk.CenteredTransformInitializer(fixed, moving, base, sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    _metric(reg, metric)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=iterations,
                                                 relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(fixed, moving)
    log.info("%s registration: metric=%.4f, stop=%s", method, reg.GetMetricValue(),
             reg.GetOptimizerStopConditionDescription())
    if method != "bspline":
        return tx

    # deformable refinement on top of the affine
    mesh = [max(1, int(s * sp / 50.0)) for s, sp in zip(fixed.GetSize(), fixed.GetSpacing())]  # ~5 cm grid
    bs = sitk.BSplineTransformInitializer(fixed, mesh)
    reg2 = sitk.ImageRegistrationMethod()
    _metric(reg2, metric)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.2, seed)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iterations)
    reg2.SetShrinkFactorsPerLevel([2, 1])
    reg2.SetSmoothingSigmasPerLevel([1, 0])
    reg2.SetMovingInitialTransform(tx)
    reg2.SetInitialTransform(bs, inPlace=True)
    reg2.Execute(fixed, moving)
    comp = sitk.CompositeTransform(tx)
    comp.AddTransform(bs)
    return comp


def align_mask(image: sitk.Image, mask: sitk.Image, cfg: dict, mask_reference: sitk.Image | None = None) -> sitk.Image:
    r = cfg["registration"]
    method = r.get("method", "auto")
    tol = r.get("geometry_tolerance", 1e-3)
    if method == "none":
        if mask.GetSize() != image.GetSize():
            raise ValueError("registration.method=none but mask and image grids differ")
        m = sitk.Image(mask)
        m.CopyInformation(image)
        return m
    # a reference image always triggers registration, even when the voxel grids coincide
    if mask_reference is not None and method in ("auto", "rigid", "affine", "bspline"):
        how = "rigid" if method == "auto" else method
        tx = register(image, mask_reference, how, r.get("metric", "mattes"), int(r.get("iterations", 200)), cfg["seed"])
        warped = sitk.Resample(sitk.Cast(mask, sitk.sitkFloat32), image, tx, sitk.sitkLinear, 0.0)
        return sitk.Cast(warped >= 0.5, sitk.sitkUInt8)

    if same_geometry(image, mask, tol):
        m = sitk.Image(mask)
        m.CopyInformation(image)
        return m

    if method in ("rigid", "affine", "bspline"):
        log.warning("registration.method=%s needs data.mask_reference_pattern; using physical resampling", method)

    out = resample_mask(mask, image)
    if sitk.GetArrayViewFromImage(out).sum() == 0 and mask.GetSize() == image.GetSize():
        # header mismatch (e.g. mask exported without origin) but identical voxel grid
        log.warning("mask does not overlap image in physical space; assuming identical voxel grid")
        out = sitk.Image(mask)
        out.CopyInformation(image)
    return out


def dice(a: sitk.Image, b: sitk.Image) -> float:
    x = sitk.GetArrayViewFromImage(a) > 0
    y = sitk.GetArrayViewFromImage(b) > 0
    s = x.sum() + y.sum()
    return float(2 * (x & y).sum() / s) if s else 1.0
