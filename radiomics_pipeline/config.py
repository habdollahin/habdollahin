"""Configuration loading: YAML file deep-merged over the defaults below."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "n_jobs": 1,
    "output_dir": "runs/radiomics",
    "data": {
        "root": "data/radiomics",          # one sub-folder per patient
        "patient_glob": "*",               # which sub-folders are patients
        "patient_id_regex": None,          # regex with one group applied to the folder name
        "image_subdir": None,              # restrict the image search to this sub-folder (glob)
        "modalities": ["CT", "MR", "PT"],  # allowed image modalities
        "series_description_regex": None,  # pick a series by description (e.g. "T2|t2_tse")
        "mask_pattern": ["*mask*", "*seg*", "*label*", "RS*", "*RTSTRUCT*"],
        "mask_format": "auto",             # auto | nifti | rtstruct | seg | dicom_series
        "roi_name_regex": None,            # RTSTRUCT ROI / SEG segment name, e.g. "GTV|tumou?r"
        "mask_label": None,                # integer label in a multi-label mask (None = any > 0)
        "mask_reference_pattern": None,    # image the mask was drawn on (enables registration)
        "suv": False,                      # convert PET to SUVbw
    },
    "preprocess": {
        "clip_range": None,                # e.g. [-1000, 3000] (HU) before anything else
        "n4_bias_correction": False,       # MR bias-field correction
        "denoise": {"method": "none"},     # none|gaussian|median|bilateral|curvature_flow|anisotropic|patch_based
        "resample": {"enabled": True, "spacing": [1.0, 1.0, 1.0], "interpolator": "bspline"},
        "normalize": {"method": "none"},   # none|zscore|zscore_roi|minmax|percentile|window
        "save_preprocessed": False,
        "qc_png": True,
    },
    "mask": {
        "keep_largest_component": False,
        "fill_holes": True,
        "min_voxels": 10,
    },
    "registration": {
        "method": "auto",                  # auto|resample|rigid|affine|bspline|none
        "metric": "mattes",                # mattes|correlation|meansquares
        "iterations": 200,
        "geometry_tolerance": 1e-3,
    },
    "features": {
        "bin_width": None,                 # fixed bin width (IBSI recommended for CT HU)
        "bin_count": 32,                   # used when bin_width is null
        "resegment_range": None,           # e.g. [-150, 250]
        "image_types": {"Original": {}, "LoG": {"sigma": [1.0, 2.0, 3.0]}, "Wavelet": {}},
        "feature_classes": ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        "force_2d": False,
        "extra_settings": {},
        "robustness": {"enabled": False, "perturbations": [-1, 1], "icc_threshold": 0.75},
    },
    "labels": {
        "file": "data/radiomics/labels.xlsx",
        "sheet": 0,
        "id_column": "PatientID",
        "label_column": "Label",
        "positive_class": None,            # binary: value treated as 1 (None = auto)
        "id_normalization": ["strip", "lower"],
        "clinical_columns": [],            # optional extra predictors from the label file
        "split_column": None,              # optional column with train/test values
        "group_column": None,              # optional grouping (e.g. several lesions / patient)
    },
    "modeling": {
        "test_size": 0.25,                 # ignored when labels.split_column is set; 0 = CV only
        "outer_folds": 5,
        "outer_repeats": 1,
        "inner_folds": 3,
        "search": "random",                # grid|random|optuna|halving
        "n_iter": 20,                      # random / optuna budget per fit
        "scoring": "roc_auc",
        "scaler": "standard",              # standard|robust|minmax|quantile
        "correlation_threshold": 0.9,
        "imbalance": "class_weight",       # none|class_weight|smote|adasyn|undersample
        "calibration": None,               # null|sigmoid|isotonic
        "selectors": ["anova", "mrmr", "lasso", "rf_importance"],
        "selector_k": [5, 10, 20],
        "models": ["logistic", "svm", "random_forest", "xgboost"],
        "ensemble_top_n": 3,
        "threshold": "youden",             # youden|0.5|f1|sens90|spec90
        "n_bootstrap": 1000,
        "shap": True,
    },
}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("image_types", "extra_settings"):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        deep_update(cfg, overrides)
    return cfg
