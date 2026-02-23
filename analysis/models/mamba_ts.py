from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _try_import_mamba():
    try:
        # mamba-ssm >=2.1 provides Mamba2; fall back to Mamba
        from mamba_ssm import Mamba
        return Mamba
    except Exception:
        return None


class ConvStem(nn.Module):
    def __init__(self, in_ch: int, d_model: int, k: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=k, padding=k // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, D, T]
        return self.net(x)


class TinyMambaTS(nn.Module):
    """A tiny time-series classifier with optional Mamba blocks.

    If mamba-ssm is unavailable, falls back to a Conv + GRU backbone.
    Inputs: x [B, C, T]; Outputs: logits [B, n_classes].
    """

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.stem = ConvStem(in_chans, d_model)
        self.use_mamba = use_mamba and (_try_import_mamba() is not None)
        if self.use_mamba:
            Mamba = _try_import_mamba()
            layers = []
            for _ in range(n_layers):
                layers.append(Mamba(d_model=d_model))
                layers.append(nn.Dropout(dropout))
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # [B, D, 1]
                nn.Flatten(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_classes),
            )
        else:
            # Fallback: BiGRU
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(2 * d_model, d_model)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.stem(x)  # [B, D, T]
        if self.use_mamba:
            # Mamba blocks expect [B, T, D]; transpose accordingly
            h2 = h.transpose(1, 2).contiguous()  # [B, T, D]
            # Run sequentially
            for layer in self.backbone:
                if hasattr(layer, "__call__") and not isinstance(layer, nn.Module):
                    h2 = layer(h2)
                else:
                    h2 = layer(h2)
            # Pool over time using avg pool on [B, D, T]
            h3 = h2.transpose(1, 2)
            out = self.head(h3)
            return out
        else:
            # GRU expects [B, T, D]
            h2 = h.transpose(1, 2).contiguous()
            y, _ = self.gru(h2)
            # Take last timestep + projection
            z = self.proj(y[:, -1, :])
            return self.head(z)

