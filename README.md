# NeuralSpell Training Pipeline

30M parameter encoder-only spell corrector. Better than Grammarly at its
core task, fully free software, trainable on a single RTX 3090.

This repository contains the **training pipeline only**. Deployment
(daemon, C API) comes later.

## What It Fixes

- Homophone errors (their/there/they're, to/too/two, your/you're)
- Fat-finger keyboard adjacency errors
- Phonetically plausible misspellings ("nife" -> "knife")
- Missing/doubled letters ("recieve", "occured")
- Contextual grammar errors ("I seen him" -> "I saw him")
- Capitalization errors in context
- Missing spaces ("missingspaces" -> "missing spaces")
- Does NOT overcorrect valid informal usage or names

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

## Training — Two Phases

**Phase 1: MLM Pretraining** on ~3.5B tokens of clean DFSG text
(Wikipedia, Project Gutenberg, Stack Exchange). Standard masked language
modeling teaches the model what valid English looks like.

**Phase 2: Corruption Fine-tuning** on ~50M synthetic corruption pairs.
A corruption engine generates realistic human errors. The model learns
token-level correction: for each input position, predict the correct
output token.

## Corruption Types

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

# Phase 1: MLM pretraining
make pretrain

# Phase 2: Corruption fine-tuning
make finetune

# Evaluate
make eval
```

## Monitoring Training

Training logs metrics to CSV and TensorBoard:

```bash
# Live dashboard (auto-refreshing matplotlib)
PYTHONPATH=. python tools/dashboard.py --watch

# Export snapshot to PNG
PYTHONPATH=. python tools/dashboard.py --export training.png

# TensorBoard (includes sample predictions)
tensorboard --logdir checkpoints/pretrain/logs/tensorboard
```

Sample predictions are logged every 5000 steps to
`checkpoints/pretrain/logs/samples.log`, showing masked tokens
vs model predictions to track learning progress.

## Target Metrics

| Metric              | Target |
|----------------------|--------|
| Homophone accuracy   | > 95%  |
| False positive rate   | < 2%   |
| Birkbeck accuracy    | > 90%  |

## Hardware

RTX 3090 (24GB VRAM). Phase 1: ~3 days. Phase 2: ~1-2 days.
~17K tokens/sec with torch.compile, batch_size=128.

## License

GPL-3.0-or-later. All training data sources verified DFSG-compliant.
