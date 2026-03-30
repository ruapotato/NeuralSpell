# NeuralSpell Training Pipeline

## Status: Training — Step 96K / 300K (32%)

**[Training Progress Dashboard](https://ruapotato.github.io/NeuralSpell/dashboard.html)**

~385M parameter encoder-decoder spell corrector, trained from scratch on
DFSG-compliant data only. Corruption engine upgraded to 22 error types
at step 80K, calibrated against real human error research (BEA-2019,
NUCLE, Birkbeck).

**JFLEG benchmark (real human errors, 100 sentences):**

| System | Params | Word Accuracy | Perfect Sentences |
|--------|--------|---------------|-------------------|
| aspell | dict | 88.0% | 23% |
| **NeuralSpell (step 90K)** | 385M | 86.9% | **26%** |
| BART-base (oliverguhr) | 139M | 64.8% | 4% |

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

Research-calibrated error generation based on BEA-2019, NUCLE, FCE,
Birkbeck, and GCSE studies. Weights match real human error frequencies.

**Word-level corruptions (~60%):**
1. Keyboard adjacency / fat finger (Damerau 1964)
2. Character deletion (most common typo at ~40% of single-edits)
3. Character insertion
4. Character transposition
5. Doubled letter errors
6. Phonetic substitution (via espeak-ng IPA)
7. Phonetic character rewrite (ph→f, ck→k, etc.)
8. Homophone/near-homophone swaps
9. Suffix confusions (-ible/-able, -ie-/-ei-)
10. Capitalization errors

**Grammar corruptions (~25%, dispatched to grammar module):**
11. Determiner/article errors — drop, insert, swap the/a/an (10-16% of human errors)
12. Preposition confusion — in/on/at/to/for/of swaps (7-11%)
13. Noun number — plural/singular, irregular plurals (3-8%)
14. Verb tense — past/present, "should of" for "should have" (6-7%)
15. Subject-verb agreement — expanded patterns (2-3%)
16. Word form — adjective/adverb, noun/verb confusion (3-5%)
17. Contraction errors — missing apostrophes (1-2%)

**Sentence-level corruptions (~15%):**
18. Missing words — dropped articles, prepositions, auxiliaries
19. Extra/repeated words — "I went went to"
20. Punctuation errors — comma, period, apostrophe
21. Missing spaces (joined words)
22. Split compound words — "cannot" → "can not", "inside" → "in side"

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
# Web dashboard (auto-refreshing, http://localhost:5000)
make dashboard

# Or with custom host/port
PYTHONPATH=. python tools/web_dashboard.py --host 0.0.0.0 --port 8080

# Matplotlib dashboard (local display)
PYTHONPATH=. python tools/dashboard.py --watch

# Export snapshot to PNG
PYTHONPATH=. python tools/dashboard.py --export training.png

# TensorBoard
tensorboard --logdir checkpoints/pretrain/logs/tensorboard

# Update the static GitHub Pages dashboard
make export-dashboard
git add docs/ && git commit -m "Update dashboard snapshot" && git push
```

The static dashboard is published at
https://ruapotato.github.io/NeuralSpell/dashboard.html
and can be regenerated anytime with `make export-dashboard`.

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
