# NeuralSpell Training Pipeline

## Status: Active — Encoder-Decoder Rewrite

~385M parameter encoder-decoder spell corrector, trained from scratch on
DFSG-compliant data only. Replaces the previous 30M encoder-only model
which was limited to same-token-count corrections.

The encoder-decoder architecture removes the token-length constraint,
enabling the model to learn all 11 corruption types including homophones,
phonetic misspellings, and missing-space corrections that change tokenization.

## Model Architecture

~385M parameter encoder-decoder transformer, implemented from scratch in PyTorch.

| Parameter        | Value |
|------------------|-------|
| Hidden size      | 1024  |
| Encoder layers   | 12    |
| Decoder layers   | 12    |
| Attention heads  | 16    |
| FFN intermediate | 4096  |
| Max sequence len | 256   |
| Vocab size       | 32000 |

Components: RMSNorm, RoPE positional embeddings, SwiGLU FFN,
three-way tied embeddings, gradient checkpointing, torch.compile.

**Encoder**: bidirectional self-attention (reads corrupted input)
**Decoder**: causal self-attention + cross-attention (generates corrected output)

## Corruption Engine

11 corruption types — reused from the original pipeline:

1. Keyboard adjacency (fat finger)
2. Character transposition
3. Character insertion
4. Character deletion
5. Phonetic substitution (via espeak-ng IPA)
6. Homophone/near-homophone swaps
7. Capitalization errors
8. Doubled letter errors
9. Common English suffix confusions (-ible/-able, -ie-/-ei-)
10. Real-word grammar errors (tense, agreement, articles)
11. Missing spaces (joined words)

## Data Sources (DFSG-Compliant Only)

- Wikipedia English (CC-BY-SA) — 2B tokens
- Project Gutenberg (Public Domain) — 1B tokens
- Stack Exchange score>=3 (CC-BY-SA) — 500M tokens

No Common Crawl derivatives. No AI-generated content.

## Quick Start

```bash
# System dependency
sudo apt install espeak-ng

# Python dependencies
pip install -r requirements.txt

# Download and prepare data
make data

# Build phonetic confusion database
make phonetics

# Train tokenizer
make tokenizer

# Verify architecture (~385M params)
make verify

# Phase 1: Denoising pretraining (~5-7 days, RTX 3090)
make pretrain

# Phase 2: Correction fine-tuning (~2-3 days)
make finetune

# Evaluate
make eval
```

## Resuming Training

Checkpoints save full state (model, optimizer, scaler, scheduler, RNG).
To resume after a crash:

```bash
PYTHONPATH=. python training/pretrain.py \
    --data-dir data/processed \
    --tokenizer tokenizer/tokenizer.model \
    --checkpoint-dir checkpoints/pretrain \
    --resume checkpoints/pretrain/step_50000.pt
```

## Monitoring Training

```bash
# Live dashboard (auto-refreshing matplotlib)
PYTHONPATH=. python tools/dashboard.py --watch

# Export snapshot to PNG
PYTHONPATH=. python tools/dashboard.py --export training.png

# TensorBoard
tensorboard --logdir checkpoints/pretrain/logs/tensorboard
```

Logged metrics per step: loss, token accuracy, learning rate,
tokens/sec, GPU memory, gradient norm, ETA.

## Training Strategy

**Phase 1: Denoising Pretraining** (300K steps)
- Corruption engine at 10-20% rate generates (corrupted, clean) pairs
- Encoder reads corrupted text, decoder generates clean text
- No token-length filtering — all corruption types included
- Gradient accumulation: batch 32 x 4 = effective 128

**Phase 2: Correction Fine-tuning** (100K steps)
- Higher corruption rates (15-40%), harder examples
- All corruption types at full weight
- Gradient accumulation: batch 32 x 2 = effective 64

## Hardware

RTX 3090 (24GB VRAM). Full pipeline: ~10 days.
FP16 mixed precision + gradient checkpointing.

## Previous Model (Mothballed)

The original 30M encoder-only model is preserved in git history.
It achieved ~90-95% on simple typos but couldn't learn homophones
or phonetic corrections due to the same-token-count constraint.
See commit `0fb865a` for the mothball state.

## License

GPL-3.0-or-later. All training data sources verified DFSG-compliant.
