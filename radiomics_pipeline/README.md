# Radiomics pipeline

An end-to-end, configurable radiomics workflow. It goes from a folder of patient DICOM
data to a validated classifier with a full statistical report:

```
patient folders (DICOM CT/MR/PET + mask)
  │
  ├─ io.py            discover patients · pick the image series · sort slices · read masks
  │                   (NIfTI/NRRD/MHA, DICOM RTSTRUCT, DICOM SEG, binary DICOM series) · PET → SUVbw
  ├─ preprocess.py    HU/intensity clipping → N4 bias correction (MR) → denoising
  │                   (gaussian | median | bilateral | curvature flow | anisotropic diffusion | patch-based NLM)
  │                   → isotropic resampling (IBSI grid alignment; B-spline image, linear+0.5 mask)
  │                   → intensity normalisation (z-score | ROI z-score | min-max | percentile | window)
  ├─ registration.py  mask → image: same grid / physical-space resampling / header repair, or
  │                   rigid | affine | B-spline registration (Mattes MI, multi-resolution) when the
  │                   mask was drawn on another image (e.g. T2 mask → ADC features)
  ├─ features.py      PyRadiomics: Original, LoG, Wavelet, Square, LBP3D …; fixed bin width or bin count
  │                   (discretisation), re-segmentation range · parallel, resumable · QC overlays
  │                   · optional robustness: erode/dilate the mask, keep features with ICC(3,1) ≥ 0.75
  ├─ labels.py        read classes (+ optional clinical covariates) from Excel/CSV; ID normalisation
  │                   (case, zero padding, punctuation, regex); unmatched-ID report
  ├─ selection.py     feature selectors fitted inside each CV fold:
  │                   ANOVA · Mann-Whitney/Kruskal · mutual information · BH-FDR · mRMR · LASSO ·
  │                   elastic net · RFE (SVM / LR) · random forest / extra trees importance · Boruta · ReliefF
  ├─ models.py        logistic (L1/L2/EN) · SVM · random forest · extra trees · gradient boosting ·
  │                   AdaBoost · XGBoost · LightGBM · (CatBoost) · KNN · naive Bayes · LDA · MLP
  │                   tuning: grid · random · successive halving · Optuna (Bayesian TPE)
  ├─ training.py      hold-out test split + repeated nested CV for every selector × model pair,
  │                   class imbalance (class weights / SMOTE / ADASYN / under-sampling),
  │                   optional probability calibration, soft-voting ensemble of the top-N
  ├─ metrics.py       AUC (bootstrap + DeLong CI), PR-AUC, accuracy, balanced accuracy, sensitivity,
  │                   specificity, PPV, NPV, F1, MCC, Cohen's κ, LR+/LR-, Youden J, Brier, log loss,
  │                   ECE, calibration intercept and slope, Hosmer-Lemeshow, DeLong tests, decision curves
  └─ plots.py/report.py
                      ROC (all models; best model with a 95% CI band and its operating point), PR curves,
                      calibration curves, decision-curve analysis, confusion matrices, AUC heat-map,
                      metric bars with CIs, feature stability, permutation importance, SHAP,
                      feature correlation, per-class probability plots → report.html + results.xlsx
```

## Avoiding data leakage

* Imputation, scaling, correlation filtering, feature selection, oversampling and
  hyper-parameter tuning all live inside one scikit-learn / imbalanced-learn `Pipeline`.
  Each one is refitted on the training part of every fold.
* **Nested CV**: the outer folds (optionally repeated) give out-of-fold (OOF) predictions. The
  inner folds tune the hyper-parameters, including the number of selected features `k`.
* Models are **ranked by nested-CV AUC**. The held-out test set is scored only once.
* The decision threshold (Youden, F1, sens ≥ 90 %, spec ≥ 90 % or a fixed value) is chosen
  on the OOF training predictions. It is then applied unchanged to the test set.
* Steps that don't use labels run once over the whole cohort, before modelling:
  the robustness (ICC) filter and the per-image preprocessing.
* `labels.split_column` sets a fixed external or temporal test set. `labels.group_column`
  keeps all lesions of one patient in the same fold (`StratifiedGroupKFold`).

## Installation

```bash
pip install numpy versioneer
pip install --no-build-isolation pyradiomics   # pyradiomics' sdist needs numpy at build time
pip install -r requirements-radiomics.txt
```

## Data layout

```
data/radiomics/
  PAT001/
    CT/ *.dcm              ← image series (any sub-folder / file names; chosen from DICOM headers)
    PAT001_mask.nii.gz     ← mask: NIfTI/NRRD …, or RS*.dcm (RTSTRUCT), or a SEG, or a folder "mask/" of DICOM
  PAT002/
    ...
  labels.xlsx              ← columns e.g. PatientID | Label | Age | Sex | Split
```

* **Image series.** If a folder holds several series, the one with the most slices is
  used. Set `data.series_description_regex` to pick a specific one (e.g. `"t2_tse"`).
  `data.modalities` limits the choice to CT / MR / PT.
* **Masks.** Files are found by `data.mask_pattern`. RTSTRUCT and SEG files are also
  found by their DICOM modality. Use `data.roi_name_regex` to choose the ROI or segment
  (e.g. `"^GTV"`), and `data.mask_label` to choose one label in a multi-label NIfTI.
* **Registration.** Set `data.mask_reference_pattern` to point at the image the mask was
  drawn on. That image is then registered to the analysed image and the mask is warped.

## Usage

```bash
# synthetic demo (80 CT patients; RTSTRUCT and NIfTI masks, some on a finer grid)
python -m radiomics_pipeline synth --out data/radiomics_synth --n 80
python -m radiomics_pipeline run --config configs/radiomics_synthetic.yaml

# your data
python -m radiomics_pipeline extract --config configs/radiomics.yaml   # features only
python -m radiomics_pipeline model   --config configs/radiomics.yaml   # selection + ML + report
python -m radiomics_pipeline run     --config configs/radiomics.yaml   # both
#   overrides: --root DIR  --labels FILE  --out DIR  --n-jobs N

# apply the best model to new patients (same preprocessing / extraction settings)
python -m radiomics_pipeline predict --model runs/radiomics/models/best_model.joblib --root new_patients/
```

All options are documented in [`configs/radiomics.yaml`](../configs/radiomics.yaml), with
defaults in `config.py`. Modality tips:

| Modality | Suggested settings |
|---|---|
| CT | `normalize: none`, `bin_width: 25` (HU), optional `clip_range` / `resegment_range` |
| MR | `n4_bias_correction: true`, `normalize: {method: zscore}`, `bin_width: null`, `bin_count: 32` |
| PET | `data.suv: true`, `normalize: none`, `bin_width: 0.25` (SUV) |

## Outputs (`output_dir`)

| Path | Content |
|---|---|
| `features/features.csv` | patients × radiomic features |
| `features/qc.csv`, `features/qc/*.png` | spacing, size, ROI volume; tri-planar mask overlay per patient |
| `features/robustness_icc.csv` | ICC(3,1) of every feature under mask erosion/dilation |
| `features/failures.csv` | patients that failed, with the reason |
| `id_matching_report.csv` | IDs present only in the features or only in the label file |
| `results.xlsx` | nested-CV and test metrics with 95 % CIs, hyper-parameters, selected features, feature stability, DeLong p-value matrices |
| `metrics_nested_cv.csv`, `metrics_test.csv` | the same tables as CSV |
| `predictions/*.csv` | OOF and test probabilities per combination |
| `figures/*.png` | all plots |
| `report.html` | self-contained report |
| `models/best_model.joblib` | final model + feature list + threshold + config, used by `predict` |

## Notes

* Run time grows as patients × selectors × models × (outer folds × repeats + 1) × the
  search budget. Start with a few selectors and models, then widen.
* Multiclass labels are supported: one-vs-rest macro AUC, macro sensitivity, specificity
  and F1, and multiclass Brier score. No threshold is used for multiclass.
* DICOM SEG reading uses a lightweight pydicom parser. For unusual SEG files, convert them
  to NIfTI first (e.g. with `dcmqi`).
* Research use only.
