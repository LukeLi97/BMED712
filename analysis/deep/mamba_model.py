import math
from typing import Optional

import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    """A very small Mamba-style block surrogate: depthwise 1D conv + gated mixing.

    This is NOT the official mamba-ssm; it is a lightweight surrogate that runs on CPU/MPS.
    """

    def __init__(self, d_model: int, d_state: int = 16, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.proj_in = nn.Linear(d_model, 2 * d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.norm(x)
        # depthwise conv along time
        h = h.transpose(1, 2)  # [B, C, T]
        h = self.dwconv(h)
        h = h.transpose(1, 2)  # [B, T, C]
        # gated mixing
        u, v = self.proj_in(h).chunk(2, dim=-1)
        v = self.act(v)
        h = u * v
        h = self.proj_out(h)
        h = self.dropout(h)
        return x + h


class MiniMamba(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 64, n_layers: int = 3, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        # temporal pooling (mean)
        h = h.mean(dim=1)
        return self.head(h)


__all__ = ["MiniMamba"]

