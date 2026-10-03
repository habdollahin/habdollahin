"""PyTorch dataset over preprocessed ``.npz`` samples."""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import load_sample


def list_samples(paths: Sequence[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.npz")))
        elif any(c in p for c in "*?["):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    return out


def build_image(s: dict) -> np.ndarray:
    """Stack model input channels -> (C, Z, Y, X) float32."""
    ct = np.clip(s["ct"].astype(np.float32), -1000.0, 2000.0) / 1000.0
    chans = [ct, s["ptv_dose"].astype(np.float32), s["ptv"].astype(np.float32),
             s["body"].astype(np.float32), s["beam"].astype(np.float32)]
    chans += [o.astype(np.float32) for o in s["oars"]]
    return np.stack(chans)


def num_input_channels(n_oars: int) -> int:
    return 5 + n_oars


class DoseDataset(Dataset):
    def __init__(self, files: List[str], patch_size: Optional[Sequence[int]] = None,
                 train: bool = False, ptv_focus: float = 0.5, cache: bool = False):
        self.files = files
        self.patch = tuple(patch_size) if patch_size else None
        self.train = train
        self.ptv_focus = ptv_focus
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.files)

    def _load(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        s = load_sample(self.files[i])
        if self._cache is not None:
            self._cache[i] = s
        return s

    def _crop_start(self, s, shape):
        starts = []
        if self.ptv_focus > 0 and np.random.rand() < self.ptv_focus and s["ptv"].any():
            pts = np.argwhere(s["ptv"])
            c = pts[np.random.randint(len(pts))]
            for ax in range(3):
                lo = c[ax] - self.patch[ax] // 2
                starts.append(int(np.clip(lo, 0, shape[ax] - self.patch[ax])))
        else:
            for ax in range(3):
                starts.append(np.random.randint(0, shape[ax] - self.patch[ax] + 1))
        return starts

    def __getitem__(self, i):
        s = self._load(i)
        img = build_image(s)
        dose = s.get("dose")
        rx = float(s["rx"])
        target = (dose / rx).astype(np.float32)[None] if dose is not None else np.zeros_like(img[:1])
        # structure masks used by the DVH loss / metrics: [PTV, OAR...]
        struct = np.concatenate([s["ptv"][None], s["oars"]], axis=0)

        if self.patch is not None and self.train:
            shape = img.shape[1:]
            if any(p > d for p, d in zip(self.patch, shape)):
                raise ValueError(f"patch {self.patch} larger than volume {shape}")
            z, y, x = self._crop_start(s, shape)
            pz, py, px = self.patch
            sl = (slice(None), slice(z, z + pz), slice(y, y + py), slice(x, x + px))
            img, target, struct = img[sl], target[sl], struct[sl]

        if self.train:
            # mild CT intensity augmentation (HU calibration differences between scanners)
            img[0] = img[0] * np.random.uniform(0.97, 1.03) + np.random.uniform(-0.02, 0.02)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "struct": torch.from_numpy(np.ascontiguousarray(struct)),
            "body": torch.from_numpy(np.ascontiguousarray(img[3:4] > 0.5)),
            "scalars": torch.from_numpy(s["scalars"]),
            "cp_tokens": torch.from_numpy(s["cp_tokens"]),
            "text": s["text"],
            "rx": torch.tensor(rx, dtype=torch.float32),
            "index": i,
        }


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ("image", "target", "struct", "body", "scalars", "cp_tokens", "rx")}
    out["text"] = [b["text"] for b in batch]
    out["index"] = [b["index"] for b in batch]
    return out
