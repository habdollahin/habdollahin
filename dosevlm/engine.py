"""Training / inference loops."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DoseDataset, collate, list_samples, num_input_channels
from .losses import DoseLoss
from .metrics import evaluate_case
from .model import DoseVLM, build_model
from .preprocess import load_sample
from .text import SimpleTokenizer

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(model: DoseVLM, tokenizer: Optional[SimpleTokenizer], texts: List[str], device):
    enc = model.plan_encoder.text_encoder
    if hasattr(enc, "tokenize"):  # HuggingFace encoder
        return enc.tokenize(texts, device)
    ids, mask = tokenizer.batch(texts)
    return ids.to(device), mask.to(device)


def forward_batch(model, tokenizer, batch, device):
    ids, mask = tokenize(model, tokenizer, batch["text"], device)
    return model(batch["image"].to(device, non_blocking=True), batch["scalars"].to(device),
                 batch["cp_tokens"].to(device), ids, mask)


# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, tokenizer, cfg, in_channels, oar_names, epoch, best, optimizer=None):
    state = {
        "model": model.state_dict(), "cfg": cfg, "in_channels": in_channels, "oar_names": oar_names,
        "tokenizer": tokenizer.state_dict() if tokenizer else None, "epoch": epoch, "best": best,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = SimpleTokenizer.from_state_dict(ck["tokenizer"]) if ck.get("tokenizer") else None
    model = build_model(ck["cfg"], ck["in_channels"], tok.vocab_size if tok else 0)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, tok, ck


# --------------------------------------------------------------------------- #
def train(cfg: Dict):
    tcfg, dcfg = cfg.get("train", {}), cfg.get("data", {})
    seed_everything(tcfg.get("seed", 42))
    out_dir = tcfg.get("output_dir", "runs/dosevlm")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_files = list_samples(dcfg["train"])
    val_files = list_samples(dcfg.get("val", [])) if dcfg.get("val") else []
    if not val_files and len(train_files) > 1:
        rng = random.Random(tcfg.get("seed", 42))
        files = train_files[:]
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * dcfg.get("val_fraction", 0.15))))
        val_files, train_files = files[:n_val], files[n_val:]
    if not train_files:
        raise ValueError("no training samples found")
    log.info("train %d / val %d samples", len(train_files), len(val_files))

    probe = load_sample(train_files[0])
    oar_names = probe["oar_names"]
    in_ch = num_input_channels(len(oar_names))

    tokenizer = None
    if not cfg.get("model", {}).get("hf_text_model"):
        texts = [load_sample(f)["text"] for f in train_files]
        tokenizer = SimpleTokenizer.build(texts, max_len=cfg.get("model", {}).get("text_max_len", 256))

    device = torch.device(tcfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, in_ch, tokenizer.vocab_size if tokenizer else 0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: %.2fM trainable parameters on %s", n_params / 1e6, device)

    train_ds = DoseDataset(train_files, dcfg.get("patch_size"), train=True,
                           ptv_focus=dcfg.get("ptv_focus", 0.5), cache=dcfg.get("cache", False))
    val_ds = DoseDataset(val_files, None, train=False, cache=dcfg.get("cache", False))
    nw = tcfg.get("num_workers", 2)
    train_dl = DataLoader(train_ds, batch_size=tcfg.get("batch_size", 2), shuffle=True, num_workers=nw,
                          collate_fn=collate, drop_last=len(train_ds) > tcfg.get("batch_size", 2),
                          pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw, collate_fn=collate)

    lcfg = cfg.get("loss", {})
    criterion = DoseLoss(lcfg.get("l1", 1.0), lcfg.get("ptv", 1.0), lcfg.get("dvh", 0.5), lcfg.get("grad", 0.1))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.get("lr", 3e-4), weight_decay=tcfg.get("weight_decay", 1e-4))
    epochs = tcfg.get("epochs", 200)
    steps = epochs * max(1, len(train_dl))
    warmup = min(tcfg.get("warmup_steps", 500), steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    accum = tcfg.get("grad_accum", 1)

    best = float("inf")
    start_epoch = 0
    if tcfg.get("resume"):
        ck = torch.load(tcfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        for _ in range(start_epoch * len(train_dl)):
            sched.step()

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path) or start_epoch == 0
    logf = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["epoch", "train_loss", "val_dose_score_gy", "val_dvh_score_gy", "lr", "seconds"])

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_dl):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                pred = forward_batch(model, tokenizer, batch, device)
            loss, _ = criterion(pred.float(), batch["target"].to(device), batch["body"].to(device),
                                batch["struct"].to(device))
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or it + 1 == len(train_dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += float(loss.detach())
            nb += 1

        val_ds_score, val_dvh = float("nan"), float("nan")
        if val_files and ((epoch + 1) % tcfg.get("val_every", 1) == 0 or epoch + 1 == epochs):
            res = validate(model, tokenizer, val_dl, device, use_amp, amp_dtype)
            val_ds_score, val_dvh = res["dose_score"], res["dvh_score"]
            if val_ds_score < best:
                best = val_ds_score
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, tokenizer, cfg, in_ch, oar_names,
                                epoch, best)
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, tokenizer, cfg, in_ch, oar_names, epoch,
                        best, opt)
        if not val_files:
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, tokenizer, cfg, in_ch, oar_names,
                            epoch, best)
        dt = time.time() - t0
        writer.writerow([epoch, running / max(nb, 1), val_ds_score, val_dvh, sched.get_last_lr()[0], dt])
        logf.flush()
        msg = f"epoch {epoch}  loss {running / max(nb, 1):.4f}"
        if not math.isnan(val_ds_score):
            msg += f"  val dose-score {val_ds_score:.3f} Gy  val DVH-score {val_dvh:.3f} Gy"
        log.info("%s  (%.1fs)", msg, dt)
    logf.close()
    return os.path.join(out_dir, "best.pt")


@torch.no_grad()
def validate(model, tokenizer, loader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    ds_scores, dvh_scores = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            pred = forward_batch(model, tokenizer, batch, device).float()
        for b in range(pred.shape[0]):
            rx = float(batch["rx"][b])
            s = load_sample(loader.dataset.files[batch["index"][b]])
            p = pred[b, 0].cpu().numpy() * rx * s["body"]
            m = evaluate_case(p, s["dose"], s) if "dose" in s else None
            if m:
                ds_scores.append(m["dose_score"])
                dvh_scores.append(m["dvh_score"])
    return {"dose_score": float(np.mean(ds_scores)) if ds_scores else float("nan"),
            "dvh_score": float(np.nanmean(dvh_scores)) if dvh_scores else float("nan")}


@torch.no_grad()
def predict_sample(model, tokenizer, sample: dict, device) -> np.ndarray:
    """Predict absolute dose (Gy) for a sample dict from ``load_sample``/``preprocess_patient``."""
    from .dataset import build_image

    model.eval()
    img = torch.from_numpy(build_image(sample))[None].to(device)
    ids, mask = tokenize(model, tokenizer, [str(sample["text"])], device)
    pred = model(img, torch.from_numpy(sample["scalars"])[None].to(device),
                 torch.from_numpy(sample["cp_tokens"])[None].to(device), ids, mask)
    rel = pred[0, 0].float().cpu().numpy()
    return rel * float(sample["rx"]) * sample["body"].astype(np.float32)
