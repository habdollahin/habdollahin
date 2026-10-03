"""Read class labels (and optional clinical covariates) from Excel/CSV and match them to
patient features using normalised IDs."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def normalize_id(x, steps) -> str:
    s = str(x)
    if isinstance(x, float) and x.is_integer():  # Excel turns 0012 into 12.0
        s = str(int(x))
    for st in steps or []:
        if st == "strip":
            s = s.strip()
        elif st == "lower":
            s = s.lower()
        elif st == "alnum":
            s = re.sub(r"[^0-9a-zA-Z]", "", s)
        elif st == "nozeros":  # PAT0012 == pat12
            s = re.sub(r"(?<![0-9])0+(?=[0-9])", "", s)
        elif st.startswith("regex:"):
            m = re.search(st[6:], s)
            s = (m.group(1) if m and m.groups() else m.group(0)) if m else s
        else:
            raise ValueError(f"unknown id normalisation step {st!r}")
    return s


def read_table(path: str | Path, sheet=0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def encode_labels(y: pd.Series, positive=None) -> tuple[np.ndarray, list]:
    """Binary -> {0,1} (positive class = 1); multiclass -> 0..K-1 in sorted order."""
    classes = sorted(pd.unique(y.dropna()), key=lambda v: str(v))
    if len(classes) < 2:
        raise ValueError(f"need >= 2 classes, got {classes}")
    if len(classes) == 2:
        if positive is None:
            # numeric 0/1, yes/no, true/false ... -> the "larger" one is positive
            pos = classes[1]
            for c in classes:
                if str(c).strip().lower() in ("1", "yes", "true", "positive", "pos", "malignant", "pcr", "event"):
                    pos = c
        else:
            matches = [c for c in classes if str(c) == str(positive)]
            if not matches:
                raise ValueError(f"positive_class {positive!r} not in {classes}")
            pos = matches[0]
        neg = [c for c in classes if c != pos][0]
        return (y == pos).astype(int).to_numpy(), [neg, pos]
    mapping = {c: i for i, c in enumerate(classes)}
    return y.map(mapping).astype(int).to_numpy(), classes


def merge_features_labels(features: pd.DataFrame, cfg: dict, out_dir: Path | None = None) -> dict:
    """Return dict(X, y, classes, ids, split, groups, clinical_columns)."""
    lc = cfg["labels"]
    tab = read_table(lc["file"], lc.get("sheet", 0))
    for col in [lc["id_column"], lc["label_column"]]:
        if col not in tab.columns:
            raise KeyError(f"column {col!r} not in label file; columns: {list(tab.columns)}")
    steps = lc.get("id_normalization", ["strip", "lower"])
    tab = tab.dropna(subset=[lc["label_column"]]).copy()
    tab["_key"] = tab[lc["id_column"]].map(lambda v: normalize_id(v, steps))
    dup = tab["_key"].duplicated(keep=False)
    if dup.any():
        log.warning("duplicate IDs in label file (first kept): %s", sorted(tab.loc[dup, "_key"].unique())[:10])
        tab = tab.drop_duplicates("_key")
    feats = features.copy()
    feats["_key"] = [normalize_id(i, steps) for i in feats.index]

    merged = feats.reset_index().merge(tab, on="_key", how="inner", suffixes=("", "_label"))
    only_f = sorted(set(feats["_key"]) - set(tab["_key"]))
    only_l = sorted(set(tab["_key"]) - set(feats["_key"]))
    log.info("label matching: %d matched, %d features-only, %d labels-only", len(merged), len(only_f), len(only_l))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unmatched_feature_ids": pd.Series(only_f), "unmatched_label_ids": pd.Series(only_l)}) \
            .to_csv(out_dir / "id_matching_report.csv", index=False)
    if len(merged) < 10:
        raise ValueError(f"only {len(merged)} patients matched between features and labels - check id_normalization")

    idx_col = features.index.name or "index"
    ids = merged[idx_col].astype(str).to_numpy()
    y, classes = encode_labels(merged[lc["label_column"]], lc.get("positive_class"))
    X = merged[[c for c in features.columns]].astype(float)

    clin = list(lc.get("clinical_columns") or [])
    if clin:
        C = merged[clin].copy()
        cat = [c for c in clin if C[c].dtype == object or str(C[c].dtype) == "category"]
        C = pd.get_dummies(C, columns=cat, drop_first=True, dtype=float)
        C.columns = [f"clinical_{c}" for c in C.columns]
        X = pd.concat([X, C.astype(float)], axis=1)
    X.index = ids

    split = None
    if lc.get("split_column"):
        s = merged[lc["split_column"]].astype(str).str.lower().str.strip()
        split = np.where(s.isin(["test", "validation", "external", "val", "1", "true"]), "test", "train")
    groups = merged[lc["group_column"]].astype(str).to_numpy() if lc.get("group_column") else None
    return {"X": X, "y": y, "classes": classes, "ids": ids, "split": split, "groups": groups}
