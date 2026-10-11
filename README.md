# DoseVLM: vision-language 3-D dose prediction for clinical VMAT

DoseVLM reads a patient's **CT**, **RTSTRUCT** and **RTPLAN** DICOM. It learns from the
clinical **RTDOSE** and predicts the full **3-D dose distribution** of a VMAT plan. The
prediction is written back out as a DICOM RT Dose file that a TPS or viewer can import.

The model is a *vision-language* model. The vision branch is a 3-D U-Net over the CT, the
structures and the projected arc apertures. The language branch reads the plan as text,
with the prescription, fractionation, arcs, energy, MU and structures, plus a token for
each control point. Cross-attention between the two branches is the fusion step.

> ⚠️ **Research use only.** Predicted doses are not a substitute for a commissioned dose
> calculation. Do not use them for treatment without independent verification. The
> exported RTDOSE is labelled `DoseVLM predicted dose`.

---

## Pipeline

```
DICOM folder per patient
  CT series ─┐
  RTSTRUCT ──┼─► preprocess ─► fixed grid (default 128³ @ 3 mm, centred on the PTV)
  RTPLAN ────┤                   ├─ image channels: CT (HU), PTV dose-level map (Rx-relative),
  RTDOSE ────┘                   │   PTV union, body, arc-aperture projection, N OAR masks
                                 ├─ plan text   : "VMAT ... Prescription 70.0 Gy in 35 fractions ... Arc 1: 6 MV,
                                 │                gantry 181 to 179 clockwise, collimator 30, 312 MU ..."
                                 ├─ control-point tokens: gantry/collimator/couch (sin, cos), MU, MLC area, energy
                                 ├─ scalar plan features: Rx, fractions, total MU, #arcs, arc length, ...
                                 └─ target: dose / Rx   (relative dose, so different sites share one model)
```

### Model (`dosevlm/model.py`)

```
image (5+N channels) ─► 3-D residual U-Net encoder ─────────── skips ───────────┐
                              │ level L-2: cross-attention voxels ← plan tokens  │
                              │ bottleneck: N × [self-attn, cross-attn, MLP]     │
                              ▼                                                  ▼
                         U-Net decoder + FiLM (pooled plan embedding) ─► softplus ─► dose / Rx

plan tokens = [scalar token] + [control-point tokens] + [text tokens] ─► plan transformer
text encoder = built-in word/digit tokenizer + transformer (default)
             | any pretrained HuggingFace encoder, e.g. BiomedBERT (`model.hf_text_model`)
```

* **Arc-aperture projection channel.** Each control point's jaws and MLCX leaves are
  projected from the source through every voxel, using IEC 61217 gantry, collimator and
  couch geometry and a per-position (HFS/HFP/FFS/FFP) patient transform. Each projection
  is weighted by the segment MU and an inverse-square factor. This gives the CNN the
  explicit beam geometry.
* **Losses** (`dosevlm/losses.py`):
  * MAE inside the body, plus extra MAE weight in the PTV.
  * A differentiable **DVH loss**: per structure, the L1 between sorted voxel doses, which
    equals the Wasserstein distance between the two DVHs.
  * A dose-gradient loss.
* **Metrics** (`dosevlm/metrics.py`):
  * OpenKBP *dose score*: MAE in the body, in Gy.
  * OpenKBP *DVH score*: D99, D95, D1 and Dmean for the PTV; Dmean and D0.1cc for OARs.
  * Global 3-D **gamma** pass rate at 3%/3 mm with a 10% threshold.

## Installation

```bash
pip install -r requirements.txt          # torch, pydicom, numpy, scipy, scikit-image, pyyaml
pip install transformers                 # optional: pretrained clinical text encoder
```

## Data layout

Use one folder per patient. File names and sub-folders don't matter; modality is read from
the DICOM headers:

```
data/dicom/
  PAT001/  CT*.dcm  RS*.dcm  RP*.dcm  RD*.dcm
  PAT002/  ...
```

* **CT series:** when there are several, the one referenced by the RTSTRUCT is used.
* **RTDOSE:** a `PLAN`-level dose that references the RTPLAN is preferred. Per-beam doses
  are summed instead.
* **Targets:** `PTV*` structures. The dose level is parsed from the name (`PTV_70`,
  `PTV7000`, `PTV_54Gy`, `PTV_High`) and falls back to the RTPLAN prescription.
  Optimisation helpers (`PTV_eval`, `*_opt`, rings) are skipped.
* **OARs:** free-text names are mapped to canonical channels by regex aliases
  (`dosevlm/structures.py`, `DEFAULT_OARS`). The mapping can be overridden with
  `preprocess.oars` in the config. Missing OARs are just empty channels, so one model
  covers head and neck, thorax and pelvis.

## Usage

```bash
# 0) (optional) synthetic smoke test: 24 fake DICOM-RT patients, trains in ~2 min on CPU
python -m dosevlm.cli synth      --out data/synthetic --n 24
python -m dosevlm.cli preprocess --input data/synthetic --output data/synthetic_npz --config configs/synthetic.yaml
python -m dosevlm.cli train      --config configs/synthetic.yaml

# 1) preprocess clinical DICOM (do this per split)
python -m dosevlm.cli preprocess --input data/dicom/train --output data/npz/train --config configs/default.yaml
python -m dosevlm.cli preprocess --input data/dicom/val   --output data/npz/val   --config configs/default.yaml

# 2) train (GPU recommended; 96³ random crops, AMP, cosine LR)
python -m dosevlm.cli train --config configs/default.yaml

# 3) evaluate on held-out cases -> per-case CSV of dose score / DVH score / gamma
python -m dosevlm.cli evaluate --checkpoint runs/dosevlm/best.pt --data data/npz/test --out test_metrics.csv

# 4) predict a new patient (CT + RTSTRUCT + RTPLAN) -> RD_predicted.dcm
python -m dosevlm.cli predict --checkpoint runs/dosevlm/best.pt --patient data/dicom/PAT123 --out pred/PAT123
```

`predict` writes:
* `RD_predicted.dcm`: an RTDOSE in the patient's frame of reference, referencing the RTPLAN.
* `predicted_dose_gy.npy`
* `metrics.json`, only when the folder also contains a clinical RTDOSE to compare against.

Python API:

```python
from dosevlm.engine import load_checkpoint, predict_sample
from dosevlm.preprocess import preprocess_patient
model, tok, ck = load_checkpoint("runs/dosevlm/best.pt", "cuda")
sample = preprocess_patient("data/dicom/PAT123", require_dose=False)
dose_gy = predict_sample(model, tok, sample, "cuda")    # (z, y, x) on sample["geometry"]
```

## Tests

```bash
python -m pytest -q tests     # covers DICOM I/O, geometry, export round-trip, model, losses, gamma, and train/predict
```

## Practical notes and limitations

* **Grid.** The default grid is 128³ at 3 mm, a 38.4 cm cube around the PTV, which fits
  head-and-neck and most thorax and pelvis cases. For very large fields, raise the
  `preprocess.shape`. Dose outside the cube is not predicted.
* **Memory.** Full-volume inference at 128³ needs several GB of GPU memory (not benchmarked here) with the
  default channels. Training uses 96³ crops. Reduce `model.channels` or `patch_size` if needed.
* **Aperture projection.** The aperture channel is a geometric fluence proxy. It does not
  model attenuation, heterogeneity, scatter or MLC transmission; the network learns those
  from the data. MLCY devices are approximated as MLCX.
* **Data and plan scope.** Training data should come from one institution's planning
  style, or be clearly separated by site in the text (`preprocess.site`). Predictions
  reflect the planning practice in the training cohort. Proton and electron plans are
  out of scope.
* **Prediction targets.** Two prediction modes are supported:
  * *Knowledge-based planning*: predict achievable dose from anatomy, Rx and arc set-up.
  * *Plan-specific dose*: control-point MU and MLC apertures are also in the input. For
    the plan-specific mode to be meaningful, the RTPLAN must be the delivered and
    optimised plan.

## Further reading

* [Predictive virtual trials using medical imaging: a short review](docs/predictive_virtual_trials_review.md).
  It covers in silico imaging and radiotherapy trials, how credible they are, and how
  DoseVLM can act as the dose engine for virtual trial arms.
