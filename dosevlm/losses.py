"""Training losses for relative-dose prediction."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred, target, mask, weight=None):
    diff = (pred - target).abs()
    if weight is not None:
        diff = diff * weight
    m = mask.float()
    return (diff * m).sum() / m.sum().clamp(min=1.0)


def dvh_loss(pred, target, struct, n_samples: int = 2048):
    """Differentiable DVH loss.

    For each structure the voxel doses of prediction and ground truth are
    sorted and compared (L1 between sorted values == 1-D Wasserstein distance
    between the two dose distributions, i.e. between the two DVH curves).
    Large structures are randomly sub-sampled to ``n_samples`` voxels.
    """
    losses = []
    b, c = struct.shape[:2]
    p = pred.flatten(2)  # (B, 1, N)
    t = target.flatten(2)
    s = struct.flatten(2)
    for bi in range(b):
        for ci in range(c):
            idx = torch.nonzero(s[bi, ci], as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue
            if idx.numel() > n_samples:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:n_samples]]
            # same voxel subset for both -> sorting gives matched quantiles
            pv, _ = torch.sort(p[bi, 0, idx])
            tv, _ = torch.sort(t[bi, 0, idx])
            losses.append((pv - tv).abs().mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def gradient_loss(pred, target, mask):
    loss = 0.0
    m = mask.float()
    for dim in (2, 3, 4):
        gp = torch.diff(pred, dim=dim)
        gt = torch.diff(target, dim=dim)
        mm = torch.narrow(m, dim, 0, m.shape[dim] - 1)
        loss = loss + ((gp - gt).abs() * mm).sum() / mm.sum().clamp(min=1.0)
    return loss / 3.0


class DoseLoss(torch.nn.Module):
    def __init__(self, w_l1=1.0, w_ptv=1.0, w_dvh=0.5, w_grad=0.1):
        super().__init__()
        self.w_l1, self.w_ptv, self.w_dvh, self.w_grad = w_l1, w_ptv, w_dvh, w_grad

    def forward(self, pred, target, body, struct):
        terms = {"l1": masked_l1(pred, target, body)}
        if self.w_ptv:
            terms["ptv"] = masked_l1(pred, target, struct[:, :1])
        if self.w_dvh:
            terms["dvh"] = dvh_loss(pred, target, struct)
        if self.w_grad:
            terms["grad"] = gradient_loss(pred, target, body)
        total = (self.w_l1 * terms["l1"] + self.w_ptv * terms.get("ptv", 0.0)
                 + self.w_dvh * terms.get("dvh", 0.0) + self.w_grad * terms.get("grad", 0.0))
        return total, {k: float(v.detach()) for k, v in terms.items()}
