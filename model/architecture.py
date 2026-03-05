"""NeuralSpell 30M parameter encoder-only transformer.

Architecture:
  - 6 layers, 512 hidden, 8 heads, 2048 FFN intermediate
  - RMSNorm (not LayerNorm)
  - RoPE positional embeddings
  - SwiGLU activation in FFN
  - Tied input/output embeddings
  - Bidirectional attention (no causal mask)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.ffn import SwiGLUFFN

# Architecture hyperparameters
HIDDEN_SIZE = 512
NUM_LAYERS = 6
NUM_HEADS = 8
INTERMEDIATE_SIZE = 1536
MAX_SEQ_LENGTH = 256
VOCAB_SIZE = 32000
DROPOUT = 0.1


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_heads: int = NUM_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size, num_heads, max_seq_length, dropout
        )
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Pre-norm architecture
        h = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h


class NeuralSpellModel(nn.Module):
    """30M parameter encoder-only spell correction model."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # No learned position embeddings — RoPE is applied in attention

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, num_heads, intermediate_size, max_seq_length, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output projection tied with input embeddings
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=attend, 0=pad

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
