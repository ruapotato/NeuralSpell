"""GECToR-style sequence tagger for spell/grammar correction.

Instead of generating the corrected sentence token-by-token (seq2seq),
this model predicts one edit operation per input token:
  $KEEP, $DELETE, $REPLACE_x, $APPEND_x, $CASE_x

Key advantages over seq2seq:
  - Non-autoregressive: all tokens classified in parallel (10x faster)
  - KEEP is the default: naturally conservative, few false positives
  - KEEP bias at inference: tunable precision/recall tradeoff
  - Iterative: apply edits, re-tag for cascading corrections (2-3 passes)
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from model.attention import MultiHeadAttention
from model.ffn import SwiGLUFFN


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, max_seq_length, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, max_seq_length, dropout)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = False

    def _forward(self, x, attention_mask=None):
        h = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h

    def forward(self, x, attention_mask=None):
        if self.use_checkpoint and self.training:
            return gradient_checkpoint(self._forward, x, attention_mask, use_reentrant=False)
        return self._forward(x, attention_mask)


class SpellTagger(nn.Module):
    """GECToR-style sequence tagger for correction.

    Architecture: bidirectional encoder + per-token classification head.
    Predicts edit operations (KEEP, DELETE, REPLACE_x, APPEND_x, etc).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_seq_length: int = 256,
        num_tags: int = 2000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tags = num_tags

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads, intermediate_size, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Tagging head
        self.tag_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.use_checkpoint = True

    def forward(self, input_ids, attention_mask=None):
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=attend, 0=pad

        Returns:
            logits: (batch, seq_len, num_tags) — edit tag logits per token
        """
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        return self.tag_head(x)

    def predict(self, input_ids, attention_mask=None, keep_bias: float = 0.0,
                min_error_prob: float = 0.0, keep_id: int = 0):
        """Predict edit tags with KEEP bias for inference.

        Args:
            keep_bias: additive bias on KEEP logit (higher = more conservative)
            min_error_prob: minimum probability for non-KEEP prediction
            keep_id: index of $KEEP in the tag vocabulary
        """
        logits = self.forward(input_ids, attention_mask)

        # Apply KEEP bias
        if keep_bias > 0:
            logits[:, :, keep_id] += keep_bias

        probs = torch.softmax(logits, dim=-1)
        pred_tags = logits.argmax(dim=-1)

        # Override to KEEP if not confident enough
        if min_error_prob > 0:
            max_non_keep = probs.clone()
            max_non_keep[:, :, keep_id] = 0
            max_error_prob = max_non_keep.max(dim=-1).values
            pred_tags[max_error_prob < min_error_prob] = keep_id

        return pred_tags

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_edit_vocab(path: Path) -> dict:
    """Load edit vocabulary from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data


def apply_edits(words: list[str], tags: list[str]) -> list[str]:
    """Apply edit tags to a word list, producing corrected text."""
    result = []
    for word, tag in zip(words, tags):
        if tag == "$KEEP":
            result.append(word)
        elif tag == "$DELETE":
            continue  # skip this word
        elif tag == "$CASE_LOWER":
            result.append(word.lower())
        elif tag == "$CASE_UPPER":
            result.append(word.upper())
        elif tag == "$CASE_TITLE":
            result.append(word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper())
        elif tag == "$MERGE" and result:
            result[-1] = result[-1] + word
        elif tag.startswith("$REPLACE_"):
            result.append(tag[9:])  # the replacement word
        elif tag.startswith("$APPEND_"):
            result.append(word)
            result.append(tag[8:])  # the appended word
        else:
            result.append(word)  # unknown tag, keep original

    return result
