"""Language branch: tokeniser + text encoders.

Two encoders are provided:

* ``SimpleTextEncoder`` (default): word-level vocabulary built from the
  training set's plan descriptions; numbers are split into digits so the
  vocabulary stays small. Trained from scratch, no extra dependencies.
* ``HFTextEncoder``: any HuggingFace encoder (e.g. a clinical / biomedical
  BERT) loaded with ``transformers``. Weights are frozen by default and a
  trainable projection maps them to the model width.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

import torch
import torch.nn as nn

_TOKEN_RE = re.compile(r"[a-z]+|\d|[^\sa-z\d]")
SPECIALS = ["<pad>", "<unk>", "<cls>"]


class SimpleTokenizer:
    def __init__(self, vocab: List[str] = None, max_len: int = 256):
        self.itos = list(vocab) if vocab else list(SPECIALS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.max_len = max_len

    @staticmethod
    def split(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_len: int = 256) -> "SimpleTokenizer":
        cnt = Counter(t for s in texts for t in cls.split(s))
        # digits are always present so unseen numbers can be represented
        vocab = SPECIALS + [str(d) for d in range(10)]
        vocab += sorted(t for t, c in cnt.items() if c >= min_freq and t not in vocab)
        return cls(vocab, max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids = [self.stoi["<cls>"]] + [self.stoi.get(t, 1) for t in self.split(text)]
        return ids[: self.max_len]

    def batch(self, texts: List[str]):
        enc = [self.encode(t) for t in texts]
        n = max(len(e) for e in enc)
        ids = torch.zeros(len(enc), n, dtype=torch.long)
        mask = torch.zeros(len(enc), n, dtype=torch.bool)
        for i, e in enumerate(enc):
            ids[i, : len(e)] = torch.tensor(e)
            mask[i, : len(e)] = True
        return ids, mask

    def state_dict(self):
        return {"itos": self.itos, "max_len": self.max_len}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["itos"], d["max_len"])


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 2, heads: int = 8,
                 max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids) + self.pos[:, : ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(x)


class HFTextEncoder(nn.Module):
    """Wraps a pretrained HuggingFace text encoder (optional dependency)."""

    def __init__(self, name: str, dim: int, freeze: bool = True, max_len: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

        self.hf_tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_len = max_len
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(self.model.config.hidden_size, dim)

    def tokenize(self, texts: List[str], device):
        t = self.hf_tokenizer(texts, padding=True, truncation=True, max_length=self.max_len,
                              return_tensors="pt")
        return t["input_ids"].to(device), t["attention_mask"].to(device).bool()

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return self.proj(h)
