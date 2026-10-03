"""Command-line interface.

    python -m dosevlm.cli synth      --out data/synthetic --n 40
    python -m dosevlm.cli preprocess --input data/dicom --output data/npz/train --config configs/default.yaml
    python -m dosevlm.cli train      --config configs/default.yaml
    python -m dosevlm.cli evaluate   --checkpoint runs/dosevlm/best.pt --data data/npz/test --out results.csv
    python -m dosevlm.cli predict    --checkpoint runs/dosevlm/best.pt --patient data/dicom/PAT01 --out pred/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import yaml

log = logging.getLogger("dosevlm")


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _pp_kwargs(cfg):
    p = cfg.get("preprocess", {})
    kw = dict(spacing=tuple(p.get("spacing", (3.0, 3.0, 3.0))), shape=tuple(p.get("shape", (128, 128, 128))),
              site=p.get("site"), max_cp_tokens=p.get("max_cp_tokens", 128))
    if p.get("oars"):
        kw["oars"] = p["oars"]
    return kw


def _patient_dirs(root):
    """Each immediate sub-directory is a patient; a root with DICOM files itself is one patient."""
    subs = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    has_files = any(os.path.isfile(os.path.join(root, f)) for f in os.listdir(root))
    return [root] if has_files or not subs else subs


def cmd_synth(a):
    from .synthetic import make_patient

    for i in range(a.n):
        info = make_patient(os.path.join(a.out, f"SYN{i:03d}"), seed=a.seed + i)
        log.info("SYN%03d %s", i, info)


def cmd_preprocess(a):
    from .preprocess import preprocess_patient

    cfg = _load_cfg(a.config) if a.config else {}
    kw = _pp_kwargs(cfg)
    os.makedirs(a.output, exist_ok=True)
    ok, failed = 0, []
    for d in _patient_dirs(a.input):
        name = os.path.basename(os.path.normpath(d))
        try:
            preprocess_patient(d, os.path.join(a.output, f"{name}.npz"), require_dose=not a.no_dose, **kw)
            ok += 1
            log.info("preprocessed %s", name)
        except Exception as e:  # keep going over a cohort
            failed.append((name, str(e)))
            log.error("FAILED %s: %s", name, e)
    log.info("%d preprocessed, %d failed", ok, len(failed))
    if failed:
        with open(os.path.join(a.output, "failed.json"), "w") as f:
            json.dump(failed, f, indent=2)


def cmd_train(a):
    from .engine import train

    cfg = _load_cfg(a.config)
    if a.output_dir:
        cfg.setdefault("train", {})["output_dir"] = a.output_dir
    if a.epochs:
        cfg.setdefault("train", {})["epochs"] = a.epochs
    print(train(cfg))


def cmd_evaluate(a):
    import torch

    from .dataset import list_samples
    from .engine import load_checkpoint, predict_sample
    from .metrics import evaluate_case
    from .preprocess import load_sample

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, _ = load_checkpoint(a.checkpoint, device)
    rows = []
    for f in list_samples(a.data):
        s = load_sample(f)
        if "dose" not in s:
            continue
        pred = predict_sample(model, tok, s, device)
        m = evaluate_case(pred, s["dose"], s)
        rows.append({"case": os.path.basename(f), "dose_score_gy": m["dose_score"],
                     "dvh_score_gy": m["dvh_score"], "gamma_3%3mm": m["gamma_3%3mm"]})
        log.info("%s  dose %.3f Gy  DVH %.3f Gy  gamma %.1f%%", rows[-1]["case"], m["dose_score"],
                 m["dvh_score"], 100 * m["gamma_3%3mm"])
    if not rows:
        log.error("no samples with reference dose found")
        return
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "case"}
    log.info("MEAN  %s", json.dumps(mean))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            w.writerow({"case": "MEAN", **mean})


def cmd_predict(a):
    import pydicom
    import torch

    from . import dicom_io
    from .engine import load_checkpoint, predict_sample
    from .export import write_rtdose
    from .geometry import Geometry
    from .metrics import evaluate_case
    from .preprocess import preprocess_patient

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, ck = load_checkpoint(a.checkpoint, device)
    kw = _pp_kwargs(ck["cfg"])
    sample = preprocess_patient(a.patient, None, require_dose=False, **kw)
    if [str(n) for n in sample["oar_names"]] != list(ck["oar_names"]):
        raise ValueError("OAR channel list differs from the one the model was trained with")
    sample["text"] = str(sample["text"])
    geom = Geometry.from_dict(json.loads(str(sample["geometry"])))
    meta = json.loads(str(sample["meta"]))
    log.info("plan description: %s", sample["text"])
    pred = predict_sample(model, tok, sample, device)

    os.makedirs(a.out, exist_ok=True)
    ref_ct = pydicom.dcmread(dicom_io.scan_patient_dir(a.patient).ct[0], stop_before_pixels=True)
    out = write_rtdose(pred, geom, os.path.join(a.out, "RD_predicted.dcm"), meta=meta, reference_ds=ref_ct)
    np.save(os.path.join(a.out, "predicted_dose_gy.npy"), pred)
    log.info("wrote %s (max %.2f Gy, Rx %.2f Gy)", out, pred.max(), float(sample["rx"]))

    if "dose" in sample:
        eval_s = dict(sample, geometry=geom, oar_names=[str(n) for n in sample["oar_names"]])
        m = evaluate_case(pred, sample["dose"], eval_s)
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        log.info("vs clinical dose: dose score %.3f Gy, DVH score %.3f Gy, gamma(3%%/3mm) %.1f%%",
                 m["dose_score"], m["dvh_score"], 100 * m["gamma_3%3mm"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dosevlm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="generate synthetic DICOM-RT patients")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("preprocess", help="DICOM folders -> .npz samples")
    p.add_argument("--input", required=True, help="folder of patient folders (or one patient folder)")
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--no-dose", action="store_true", help="do not require RTDOSE (inference-only data)")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--epochs", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out")
    p.add_argument("--device")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict dose for one DICOM patient folder and write RTDOSE")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device")
    p.set_defaults(func=cmd_predict)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    a.func(a)


if __name__ == "__main__":
    main()
