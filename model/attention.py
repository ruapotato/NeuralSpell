"""Multi-head attention with Rotary Position Embeddings (RoPE).

Three attention patterns for the encoder-decoder architecture:
  - MultiHeadAttention: bidirectional self-attention (encoder)
  - CausalMultiHeadAttention: masked self-attention (decoder)
  - CrossAttention: decoder queries attend to encoder keys/values
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(dim: int, max_seq_length: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor.

    Returns complex-valued tensor of shape (max_seq_length, dim // 2).
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_length, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex-valued

    Returns:
        Tensor with same shape as x, with RoPE applied.
    """
    # Reshape x to complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Apply rotation
    freqs = freqs[:x.shape[1], :]  # trim to actual seq length
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


class MultiHeadAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE (encoder)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_length),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (bidirectional — no causal mask)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask[:, None, None, :].float()
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(attn_output)


class CausalMultiHeadAttention(nn.Module):
    """Causal (masked) multi-head self-attention with RoPE (decoder)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_length),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # Padding mask
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].float()
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(attn_output)


class CrossAttention(nn.Module):
    """Cross-attention: decoder queries attend to encoder output.

    RoPE applied to both Q (decoder positions) and K (encoder positions)
    so the model can learn source-target positional relationships.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_length),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, dec_seq_len, _ = x.shape
        enc_seq_len = encoder_output.shape[1]

        q = self.q_proj(x).view(batch, dec_seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(encoder_output).view(batch, enc_seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(encoder_output).view(batch, enc_seq_len, self.num_heads, self.head_dim)

        # RoPE: Q uses decoder positions, K uses encoder positions
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        q = q.transpose(1, 2)  # (batch, heads, dec_seq, head_dim)
        k = k.transpose(1, 2)  # (batch, heads, enc_seq, head_dim)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        # (batch, heads, dec_seq, enc_seq)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Encoder padding mask only (no causal mask for cross-attention)
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask[:, None, None, :].float()
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, dec_seq_len, -1)

        return self.o_proj(attn_output)
