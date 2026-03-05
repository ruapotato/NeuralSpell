"""SwiGLU Feed-Forward Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU activation.

    SwiGLU: gate(x) * up(x) where gate uses SiLU activation.
    Uses 2/3 of the intermediate size for each of gate and up projections
    to keep parameter count equivalent to a standard FFN.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU uses two parallel projections, so we scale to keep param count
        # roughly equivalent. With standard FFN: 2 * h * i params.
        # With SwiGLU: 3 * h * (2i/3) = 2 * h * i params.
        swiglu_intermediate = int(2 * intermediate_size / 3)
        self.gate_proj = nn.Linear(hidden_size, swiglu_intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, swiglu_intermediate, bias=False)
        self.down_proj = nn.Linear(swiglu_intermediate, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
