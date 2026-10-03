"""DoseVLM: a 3-D vision-language model for VMAT dose prediction.

Architecture
------------
::

    CT + PTV dose-level map + OAR masks + body + arc-aperture projection
        │
        ▼
    3-D residual U-Net encoder  ──────────────── skips ──────────────┐
        │                                                            │
        ▼  (level L-2)  cross-attention: voxels ← plan tokens        │
        ▼  (bottleneck) N × [self-attn, cross-attn, MLP] transformer │
        │                                                            │
        ▼                                                            ▼
    3-D U-Net decoder with FiLM conditioning on the pooled plan embedding
        │
        ▼
    relative dose (dose / prescription), softplus ≥ 0

    Plan tokens = [scalar-plan token] + [control-point tokens] + [text tokens],
    jointly encoded by a small transformer ("plan encoder").
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import HFTextEncoder, SimpleTextEncoder


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ResBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h = self.act(self.n1(self.conv1(x)))
        h = self.drop(h)
        h = self.n2(self.conv2(h))
        return self.act(h + self.skip(x))


def sincos_3d(d: int, h: int, w: int, dim: int, device) -> torch.Tensor:
    """Fixed 3-D sinusoidal position encoding, shape (d*h*w, dim)."""
    per = (dim // 6) * 2  # even number of channels per axis
    freqs = torch.exp(torch.arange(0, per, 2, device=device).float() * (-math.log(10000.0) / per))
    out = []
    for n, axis in ((d, 0), (h, 1), (w, 2)):
        pos = torch.arange(n, device=device).float()
        ang = pos[:, None] * freqs[None]
        enc = torch.cat([ang.sin(), ang.cos()], dim=1)  # (n, per)
        shape = [1, 1, 1, per]
        shape[axis] = n
        out.append(enc.view(*shape).expand(d, h, w, per))
    pe = torch.cat(out, dim=-1).reshape(d * h * w, -1)
    if pe.shape[1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[1]))
    return pe


class FusionBlock(nn.Module):
    """Pre-norm transformer block: self-attention over image tokens (optional),
    cross-attention from image tokens to plan/text tokens, then an MLP."""

    def __init__(self, dim: int, heads: int, self_attn: bool = True, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.self_attn = self_attn
        if self_attn:
            self.n1 = nn.LayerNorm(dim)
            self.sa = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.ca = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim))
        # zero-init gates so the block starts as identity (stable fine-tuning)
        self.g_ca = nn.Parameter(torch.zeros(1))

    def forward(self, x, ctx, ctx_mask):
        if self.self_attn:
            h = self.n1(x)
            x = x + self.sa(h, h, h, need_weights=False)[0]
        k = self.nk(ctx)
        x = x + self.g_ca.tanh() * self.ca(self.n2(x), k, k, key_padding_mask=~ctx_mask,
                                           need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class VoxelCrossAttention(nn.Module):
    """Cross-attention of a feature map to plan tokens (no voxel self-attention)."""

    def __init__(self, channels: int, dim: int, heads: int, depth: int = 1):
        super().__init__()
        self.inp = nn.Linear(channels, dim)
        self.blocks = nn.ModuleList(FusionBlock(dim, heads, self_attn=False) for _ in range(depth))
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f, ctx, ctx_mask):
        b, c, d, h, w = f.shape
        x = f.flatten(2).transpose(1, 2)
        t = self.inp(x) + sincos_3d(d, h, w, self.inp.out_features, f.device)[None].to(x.dtype)
        for blk in self.blocks:
            t = blk(t, ctx, ctx_mask)
        return f + self.out(t).transpose(1, 2).reshape(b, c, d, h, w)


class FiLM(nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__()
        self.lin = nn.Linear(dim, channels * 2)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, f, z):
        g, b = self.lin(z).chunk(2, dim=-1)
        return f * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None]


class PlanEncoder(nn.Module):
    """Encodes the scalar plan token, control-point tokens and text tokens jointly."""

    def __init__(self, dim: int, cp_features: int, scalar_features: int, text_encoder: nn.Module,
                 depth: int = 2, heads: int = 8, max_cp: int = 512):
        super().__init__()
        self.text_encoder = text_encoder
        self.scalar = nn.Sequential(nn.Linear(scalar_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cp = nn.Sequential(nn.Linear(cp_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cp_pos = nn.Parameter(torch.zeros(1, max_cp, dim))
        self.type_emb = nn.Parameter(torch.zeros(3, dim))
        nn.init.trunc_normal_(self.cp_pos, std=0.02)
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, scalars, cp_tokens, text_ids, text_mask):
        b = scalars.shape[0]
        s = self.scalar(scalars)[:, None] + self.type_emb[0]
        cp_valid = cp_tokens[..., -1] > 0.5
        c = self.cp(cp_tokens) + self.cp_pos[:, : cp_tokens.shape[1]] + self.type_emb[1]
        t = self.text_encoder(text_ids, text_mask) + self.type_emb[2]
        tokens = torch.cat([s, c, t], dim=1)
        mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=s.device), cp_valid, text_mask], dim=1)
        tokens = self.norm(self.encoder(tokens, src_key_padding_mask=~mask))
        pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return tokens, mask, pooled


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class DoseVLM(nn.Module):
    def __init__(self, in_channels: int, vocab_size: int = 0, dim: int = 256,
                 channels: Sequence[int] = (32, 64, 128, 256, 320), fusion_depth: int = 4,
                 heads: int = 8, plan_depth: int = 2, text_depth: int = 2, cp_features: int = 12,
                 scalar_features: int = 10, max_cp: int = 512, text_max_len: int = 256,
                 hf_text_model: Optional[str] = None, freeze_text: bool = True, dropout: float = 0.0):
        super().__init__()
        self.channels = list(channels)
        self.levels = len(channels)
        if hf_text_model:
            text_enc = HFTextEncoder(hf_text_model, dim, freeze=freeze_text, max_len=text_max_len)
        else:
            text_enc = SimpleTextEncoder(vocab_size, dim, text_depth, heads, text_max_len)
        self.plan_encoder = PlanEncoder(dim, cp_features, scalar_features, text_enc, plan_depth, heads, max_cp)

        # encoder
        self.stem = ResBlock3d(in_channels, channels[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(1, self.levels):
            self.down.append(nn.Conv3d(channels[i - 1], channels[i], 3, stride=2, padding=1))
            self.enc.append(ResBlock3d(channels[i], channels[i], dropout))

        # vision-language fusion
        self.mid_xattn = VoxelCrossAttention(channels[-2], dim, heads, depth=1) if self.levels > 2 else None
        self.bott_in = nn.Linear(channels[-1], dim)
        self.bott_blocks = nn.ModuleList(FusionBlock(dim, heads, True, dropout=dropout)
                                         for _ in range(fusion_depth))
        self.bott_out = nn.Linear(dim, channels[-1])

        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.film = nn.ModuleList()
        for i in range(self.levels - 2, -1, -1):
            self.up.append(nn.ConvTranspose3d(channels[i + 1], channels[i], 2, stride=2))
            self.dec.append(ResBlock3d(channels[i] * 2, channels[i], dropout))
            self.film.append(FiLM(dim, channels[i]))
        self.head = nn.Conv3d(channels[0], 1, 1)

    # ------------------------------------------------------------------ #
    def forward(self, image, scalars, cp_tokens, text_ids, text_mask) -> torch.Tensor:
        ctx, ctx_mask, pooled = self.plan_encoder(scalars, cp_tokens, text_ids, text_mask)

        # pad to a multiple of 2**(levels-1)
        m = 2 ** (self.levels - 1)
        d, h, w = image.shape[-3:]
        pad = [(-s) % m for s in (w, h, d)]
        if any(pad):
            image = F.pad(image, (0, pad[0], 0, pad[1], 0, pad[2]))

        skips = [self.stem(image)]
        x = skips[0]
        for i, (dn, enc) in enumerate(zip(self.down, self.enc)):
            x = enc(dn(x))
            if i == self.levels - 3 and self.mid_xattn is not None:
                x = self.mid_xattn(x, ctx, ctx_mask)
            skips.append(x)

        b, c, bd, bh, bw = x.shape
        t = self.bott_in(x.flatten(2).transpose(1, 2))
        t = t + sincos_3d(bd, bh, bw, t.shape[-1], x.device)[None].to(t.dtype)
        for blk in self.bott_blocks:
            t = blk(t, ctx, ctx_mask)
        x = x + self.bott_out(t).transpose(1, 2).reshape(b, c, bd, bh, bw)

        for up, dec, film, skip in zip(self.up, self.dec, self.film, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], dim=1))
            x = film(x, pooled)
        out = F.softplus(self.head(x))
        return out[..., :d, :h, :w]


def build_model(cfg: Dict, in_channels: int, vocab_size: int) -> DoseVLM:
    m = cfg.get("model", {})
    return DoseVLM(
        in_channels=in_channels, vocab_size=vocab_size, dim=m.get("dim", 256),
        channels=tuple(m.get("channels", (32, 64, 128, 256, 320))), fusion_depth=m.get("fusion_depth", 4),
        heads=m.get("heads", 8), plan_depth=m.get("plan_depth", 2), text_depth=m.get("text_depth", 2),
        max_cp=max(512, cfg.get("preprocess", {}).get("max_cp_tokens", 128)), text_max_len=m.get("text_max_len", 256),
        hf_text_model=m.get("hf_text_model"), freeze_text=m.get("freeze_text", True),
        dropout=m.get("dropout", 0.0),
    )
