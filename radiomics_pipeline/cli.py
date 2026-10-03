"""Command line interface.

  python -m radiomics_pipeline.cli synth    --out data/radiomics_synth --n 40
  python -m radiomics_pipeline.cli extract  --config configs/radiomics.yaml
  python -m radiomics_pipeline.cli model    --config configs/radiomics.yaml
  python -m radiomics_pipeline.cli run      --config configs/radiomics.yaml      # extract + model + report
  python -m radiomics_pipeline.cli predict  --model runs/radiomics/models/best_model.joblib --root new_patients/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config


def _setup_logging(out_dir: Path | None, verbose: bool):
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "pipeline.log"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("radiomics", "matplotlib", "PIL", "shap", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _overrides(args) -> dict:
    o: dict = {}
    if getattr(args, "root", None):
        o.setdefault("data", {})["root"] = args.root
    if getattr(args, "labels", None):
        o.setdefault("labels", {})["file"] = args.labels
    if getattr(args, "out", None):
        o["output_dir"] = args.out
    if getattr(args, "n_jobs", None):
        o["n_jobs"] = args.n_jobs
    return o


def cmd_extract(cfg) -> pd.DataFrame:
    from .features import extract_all

    return extract_all(cfg)


def cmd_model(cfg, features: pd.DataFrame | None = None):
    from .labels import merge_features_labels
    from .report import make_figures, write_report
    from .training import run_experiment

    out = Path(cfg["output_dir"])
    if features is None:
        features = pd.read_csv(out / "features" / "features.csv", index_col=0)
    features.index = features.index.astype(str)
    data = merge_features_labels(features, cfg, out)
    exp = run_experiment(data, cfg)
    figs = make_figures(exp, cfg)
    return write_report(exp, cfg, figs)


def cmd_predict(args):
    import joblib

    from .features import process_case
    from .io import discover_patients
    from .training import predict_proba_saved

    bundle = joblib.load(args.model)
    cfg = bundle["config"]
    cfg["data"]["root"] = args.root
    cfg["output_dir"] = args.out or str(Path(args.root) / "predictions")
    cfg["features"]["robustness"]["enabled"] = False
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    rows = []
    feats = {}
    for case in discover_patients(cfg):
        r = process_case(case, cfg, cfg["output_dir"])
        if r["error"]:
            logging.error("%s: %s", case.patient_id, r["error"])
            continue
        feats[case.patient_id] = r["features"]
    X = pd.DataFrame(feats).T
    clin = [f for f in bundle["features"] if f.startswith("clinical_")]
    if clin:
        logging.warning("model uses clinical columns %s; supply them with --clinical (missing -> imputed)", clin)
        if args.clinical:
            from .labels import normalize_id, read_table

            lc = cfg["labels"]
            tab = read_table(args.clinical, lc.get("sheet", 0))
            tab.index = tab[lc["id_column"]].map(lambda v: normalize_id(v, lc["id_normalization"]))
            C = pd.get_dummies(tab[lc["clinical_columns"]], drop_first=True, dtype=float)
            C.columns = [f"clinical_{c}" for c in C.columns]
            keys = [normalize_id(i, lc["id_normalization"]) for i in X.index]
            X = pd.concat([X, C.reindex(keys).set_axis(X.index)], axis=1)
    P = predict_proba_saved(bundle, X)
    classes = bundle["classes"]
    for pid, p in zip(X.index, P):
        if P.ndim == 1:
            rows.append({"PatientID": pid, f"P({classes[1]})": p,
                         "prediction": classes[int(p >= bundle["threshold"])], "threshold": bundle["threshold"]})
        else:
            rows.append({"PatientID": pid, **{f"P({c})": v for c, v in zip(classes, p)},
                         "prediction": classes[int(p.argmax())]})
    out = Path(cfg["output_dir"]) / "predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logging.info("predictions for %d patients -> %s (model %s)", len(rows), out, bundle["name"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="radiomics_pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "model", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None)
        p.add_argument("--root", help="override data.root")
        p.add_argument("--labels", help="override labels.file")
        p.add_argument("--out", help="override output_dir")
        p.add_argument("--n-jobs", type=int, dest="n_jobs")
        p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("synth")
    p.add_argument("--out", default="data/radiomics_synth")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--clinical", default=None, help="label/clinical table for models with clinical features")
    p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "synth":
        _setup_logging(None, False)
        from .synthetic import make_cohort

        out = make_cohort(args.out, args.n, args.seed)
        logging.info("synthetic cohort written to %s (labels: %s)", out, out / "labels.xlsx")
        return
    if args.cmd == "predict":
        _setup_logging(None, args.verbose)
        cmd_predict(args)
        return

    cfg = load_config(args.config, _overrides(args))
    out = Path(cfg["output_dir"])
    _setup_logging(out, args.verbose)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), f, sort_keys=False)
    feats = None
    if args.cmd in ("extract", "run"):
        feats = cmd_extract(cfg)
    if args.cmd in ("model", "run"):
        cmd_model(cfg, feats)


if __name__ == "__main__":
    main()
