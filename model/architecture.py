"""NeuralSpell ~385M parameter encoder-decoder transformer.

Architecture:
  - Encoder: 12 layers, bidirectional self-attention + SwiGLU FFN
  - Decoder: 12 layers, causal self-attention + cross-attention + SwiGLU FFN
  - RMSNorm, RoPE positional embeddings, SwiGLU activation
  - Three-way tied embeddings (encoder input, decoder input, output projection)
  - Gradient checkpointing support for 24GB VRAM training
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from model.attention import MultiHeadAttention, CausalMultiHeadAttention, CrossAttention
from model.ffn import SwiGLUFFN

# Architecture hyperparameters
HIDDEN_SIZE = 1024
ENCODER_LAYERS = 12
DECODER_LAYERS = 12
NUM_HEADS = 16
INTERMEDIATE_SIZE = 4096
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


class EncoderBlock(nn.Module):
    """Encoder block: bidirectional self-attention + SwiGLU FFN."""

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
        self.use_checkpoint = False

    def _forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return gradient_checkpoint(self._forward, x, attention_mask, use_reentrant=False)
        return self._forward(x, attention_mask)


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention + SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_heads: int = NUM_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.self_attention = CausalMultiHeadAttention(
            hidden_size, num_heads, max_seq_length, dropout
        )
        self.cross_attention = CrossAttention(
            hidden_size, num_heads, max_seq_length, dropout
        )
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.norm3 = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = False

    def _forward(
        self,
        x: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Causal self-attention
        h = x + self.dropout(self.self_attention(self.norm1(x), decoder_attention_mask))
        # Cross-attention to encoder
        h = h + self.dropout(self.cross_attention(self.norm2(h), encoder_output, encoder_attention_mask))
        # FFN
        h = h + self.dropout(self.ffn(self.norm3(h)))
        return h

    def forward(
        self,
        x: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return gradient_checkpoint(
                self._forward, x, decoder_attention_mask,
                encoder_output, encoder_attention_mask,
                use_reentrant=False,
            )
        return self._forward(x, decoder_attention_mask, encoder_output, encoder_attention_mask)


class NeuralSpellModel(nn.Module):
    """~385M parameter encoder-decoder spell correction model."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        encoder_layers: int = ENCODER_LAYERS,
        decoder_layers: int = DECODER_LAYERS,
        num_heads: int = NUM_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Shared embedding (3-way tied: encoder input, decoder input, output)
        self.shared_embedding = nn.Embedding(vocab_size, hidden_size)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads, intermediate_size, max_seq_length, dropout)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = RMSNorm(hidden_size)
        self.encoder_dropout = nn.Dropout(dropout)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, intermediate_size, max_seq_length, dropout)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = RMSNorm(hidden_size)
        self.decoder_dropout = nn.Dropout(dropout)

        # Output projection (tied with shared embedding)
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_proj.weight = self.shared_embedding.weight

        self._init_weights(encoder_layers + decoder_layers)

    def _init_weights(self, total_layers: int):
        """Initialize weights with depth-scaled residual projections."""
        residual_scale = 1.0 / math.sqrt(2.0 * total_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale down residual-path output projections for training stability
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            # Attention output projections
            if hasattr(layer, 'attention'):
                torch.nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=0.02 * residual_scale)
            if hasattr(layer, 'self_attention'):
                torch.nn.init.normal_(layer.self_attention.o_proj.weight, mean=0.0, std=0.02 * residual_scale)
            if hasattr(layer, 'cross_attention'):
                torch.nn.init.normal_(layer.cross_attention.o_proj.weight, mean=0.0, std=0.02 * residual_scale)
            # FFN output projections
            torch.nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=0.02 * residual_scale)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all encoder and decoder blocks."""
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            layer.use_checkpoint = True

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode corrupted input text.

        Args:
            input_ids: (batch, enc_seq_len) token IDs
            attention_mask: (batch, enc_seq_len) 1=attend, 0=pad

        Returns:
            encoder_output: (batch, enc_seq_len, hidden_size)
        """
        x = self.shared_embedding(input_ids)
        x = self.encoder_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode corrected output from encoder representations.

        Args:
            decoder_input_ids: (batch, dec_seq_len) token IDs
            decoder_attention_mask: (batch, dec_seq_len) 1=attend, 0=pad
            encoder_output: (batch, enc_seq_len, hidden_size) from encode()
            encoder_attention_mask: (batch, enc_seq_len) 1=attend, 0=pad

        Returns:
            logits: (batch, dec_seq_len, vocab_size)
        """
        x = self.shared_embedding(decoder_input_ids)
        x = self.decoder_dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, decoder_attention_mask, encoder_output, encoder_attention_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Full forward pass (training convenience).

        Returns:
            logits: (batch, dec_seq_len, vocab_size)
        """
        encoder_output = self.encode(input_ids, attention_mask)
        return self.decode(decoder_input_ids, decoder_attention_mask, encoder_output, attention_mask)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
