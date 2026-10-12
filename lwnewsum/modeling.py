"""LW-NEWSUM v2 and the encoder-side adapter baselines.

Architecture (matches Figure 1 and Section 4 of the manuscript):

    H  = Enc_T5(X)                                   (frozen, d = 512 for T5-Small)
    Z  = LN(W_d H + b_d) + PE                        (bottleneck, d_b = 256)
    L  = T_local(Z)      banded self-attention, |i - j| <= w/2
    G  = T_global(L)     full self-attention         (sequential topology)
         T_global(Z)                                  (parallel topology, ablation)
    F  = GELU(W_f [L; G] + b_f)                      (semantic fusion)
    U  = (W_u F + b_u) * m                           (m = padding mask)
    H~ = H + alpha U                                 (learnable residual scale)
    Y^ = Dec_T5(H~)                                  (frozen native decoder)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


def build_attention_mask(pad_mask: torch.Tensor, n_heads: int, window: int) -> torch.Tensor:
    """Boolean mask for nn.TransformerEncoder, shape (B * n_heads, T, T); True = blocked.

    ``window > 0`` gives banded (local) attention |i - j| <= window // 2;
    ``window <= 0`` gives full (global) attention. Padded keys are always blocked.
    The diagonal is always allowed so no query row is fully masked (avoids NaNs
    for padded queries, whose outputs are zeroed later anyway).
    """
    b, t = pad_mask.shape
    device = pad_mask.device
    key_ok = pad_mask.bool().unsqueeze(1).expand(b, t, t)  # (B, T, T) over keys
    allowed = key_ok.clone()
    if window and window > 0:
        idx = torch.arange(t, device=device)
        band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window // 2
        allowed &= band.unsqueeze(0)
    allowed |= torch.eye(t, dtype=torch.bool, device=device).unsqueeze(0)
    blocked = ~allowed
    return blocked.repeat_interleave(n_heads, dim=0)


def _encoder_stack(dim, n_heads, ffn, dropout, layers) -> Optional[nn.TransformerEncoder]:
    if layers <= 0:
        return None
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
        batch_first=True, norm_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


# --------------------------------------------------------------------------- #
# LW-NEWSUM contextual module
# --------------------------------------------------------------------------- #
class LocalGlobalModule(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim=256, n_heads=4, ffn_dim=512, local_layers=2,
                 global_layers=1, dropout=0.1, local_window=32, topology="sequential",
                 positional_encoding=True, fusion="concat", residual="learned", alpha_init=0.1, **_):
        super().__init__()
        assert topology in {"sequential", "parallel"}
        assert fusion in {"concat", "mean", "local_only", "global_only"}
        assert residual in {"learned", "fixed"}
        self.n_heads, self.window, self.topology, self.fusion_mode = n_heads, local_window, topology, fusion
        self.down = nn.Sequential(nn.Linear(d_model, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.Dropout(dropout))
        self.pe = SinusoidalPE(bottleneck_dim) if positional_encoding else None
        self.local = _encoder_stack(bottleneck_dim, n_heads, ffn_dim, dropout, local_layers)
        self.glob = _encoder_stack(bottleneck_dim, n_heads, ffn_dim, dropout, global_layers)
        self.fuse = (nn.Sequential(nn.Linear(2 * bottleneck_dim, bottleneck_dim), nn.GELU(), nn.Dropout(dropout))
                     if fusion == "concat" else None)
        self.up = nn.Linear(bottleneck_dim, d_model)
        if residual == "learned":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(1.0), persistent=False)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        z = self.down(h)
        if self.pe is not None:
            z = self.pe(z)
        m_local = build_attention_mask(pad_mask, self.n_heads, self.window)
        m_global = build_attention_mask(pad_mask, self.n_heads, 0)
        l = self.local(z, mask=m_local) if self.local is not None else z
        g_in = l if self.topology == "sequential" else z
        g = self.glob(g_in, mask=m_global) if self.glob is not None else g_in
        if self.fusion_mode == "concat":
            f = self.fuse(torch.cat([l, g], dim=-1))
        elif self.fusion_mode == "mean":
            f = 0.5 * (l + g)
        elif self.fusion_mode == "local_only":
            f = l
        else:
            f = g
        u = self.up(f) * pad_mask.unsqueeze(-1).to(f.dtype)
        return h + self.alpha.to(h.dtype) * u


class MLPAdapter(nn.Module):
    """Controlled baseline: same budget, no attention (token-wise residual bottleneck MLP)."""

    def __init__(self, d_model: int, bottleneck_dim: int = 256, dropout: float = 0.1, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, pad_mask):
        return h + self.net(h) * pad_mask.unsqueeze(-1).to(h.dtype)


# --------------------------------------------------------------------------- #
# Wrapper: frozen (or trainable) seq2seq backbone + encoder-side module
# --------------------------------------------------------------------------- #
class IdentityModule(nn.Module):
    """No-op module: lets a plain fine-tuned model generate through EncoderAdaptedSeq2Seq (timing control)."""

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return h


class EncoderAdaptedSeq2Seq(nn.Module):
    def __init__(self, backbone, module: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.module = module
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def config(self):
        return self.backbone.config

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep backbone dropout off when it is frozen
        return self

    def encode(self, input_ids, attention_mask) -> BaseModelOutput:
        enc = self.backbone.get_encoder()
        if self.freeze_backbone:
            with torch.no_grad():
                h = enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        else:
            h = enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        return BaseModelOutput(last_hidden_state=self.module(h, attention_mask))

    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.encode(input_ids, attention_mask)
        return self.backbone(attention_mask=attention_mask, encoder_outputs=enc, labels=labels, return_dict=True)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, **gen_kwargs):
        enc = self.encode(input_ids, attention_mask)
        return self.backbone.generate(encoder_outputs=enc, attention_mask=attention_mask, **gen_kwargs)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / max(total, 1)}
