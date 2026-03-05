# NeuralSpell Training Pipeline

## Status: Mothballed

32M parameter encoder-only spell corrector experiment. Trained from scratch
on DFSG-compliant data only. **Architecture limitations prevent this from
reaching production quality** — see "Limitations" below.

This repository contains the training pipeline, corruption engine, and
evaluation framework. The model and training code work, but the approach
needs a fundamental architecture change to handle harder corrections.

## What It Can Actually Do

**Works well:**
- Missing spaces: "Canthis" -> "Can this", "notto" -> "not to" (~95%)
- Simple typos: "hir" -> "her", "hed" -> "head", "bow" -> "now" (~90%)
- Case normalization: "THAT" -> "that", "WAS" -> "was" (~100%)
- Missing/extra letters: "ne'fr" -> "ne'er", "uer" -> "user" (~85%)
- Keyboard adjacency: "abd" -> "and", "das" -> "was" (~90%)

**Works poorly:**
- Homophones: "their" vs "there", "cake" vs "sake" — often fails (~30%)
- Phonetic misspellings: "Pyton" stays "Pyton", "nessesary" — often fails
- Rare/proper nouns: "Mrbeau" stays wrong
- Multi-token corruptions that change tokenization — cannot fix by design

## Why It's Mothballed

The encoder-only architecture requires input and output to have the **same
number of tokens**. This means:

1. Any corruption that changes tokenization (most homophones, phonetic
   misspellings, missing spaces with complex words) gets filtered out of
   training data entirely
2. The model literally cannot learn to fix the hardest, most important errors
3. 32M params is too small for the vocabulary + context understanding needed

**To reach production quality, this needs:**
- Encoder-decoder architecture (~400M params) for variable-length correction
- Trained from scratch on the same DFSG data (no pretrained model shortcuts)
- This is a significant rewrite — the corruption engine, data pipeline, eval
  sets, and tokenizer all carry over, but model + training loops need replacing

## Training Results

**Phase 1: MLM Pretraining** — 270K steps, ~14 hours on RTX 3090
- Loss: 10.2 -> 2.4 (plateaued around step 100K)
- Throughput: ~17-23K tokens/sec with torch.compile

**Phase 2: Corruption Fine-tuning** — 135K/200K steps, ~4 hours
- Loss: 0.24 -> 0.05
- Token accuracy: 97.2% -> 99.2%
- Stopped early due to architecture limitations making further training futile

## Model Architecture

~32M parameter BERT-style encoder, implemented from scratch in PyTorch.

| Parameter        | Value |
|------------------|-------|
| Hidden size      | 512   |
| Layers           | 6     |
| Attention heads  | 8     |
| FFN intermediate | 1536  |
| Max sequence len | 256   |
| Vocab size       | 32000 |

Components: RMSNorm, RoPE positional embeddings, SwiGLU FFN,
tied input/output embeddings, bidirectional attention, torch.compile.

## Corruption Engine (Reusable)

11 corruption types — this is the most valuable part of the project:

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

# Phase 1: MLM pretraining (~14 hours, RTX 3090)
make pretrain

# Phase 2: Corruption fine-tuning (~6 hours)
make finetune

# Evaluate
make eval
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

## Hardware

RTX 3090 (24GB VRAM). Full pipeline: ~1 day.
~17K tokens/sec with torch.compile, batch_size=128.

## License

GPL-3.0-or-later. All training data sources verified DFSG-compliant.
