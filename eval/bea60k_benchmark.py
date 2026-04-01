#!/usr/bin/env python3
"""BEA-60K benchmark — the standard spell correction evaluation.

63,044 sentences with real spelling errors from ESL learner corpora
(W&I, LOCNESS, FCE, Lang-8). This is the benchmark NeuSpell and others
report on, making results directly comparable to published numbers.

Metric: Word correction rate — percentage of misspelled words that
are corrected to the right word. Same metric as NeuSpell's paper.

Published baselines (from NeuSpell, EMNLP 2020):
  aspell:           48.7%
  JamSpell:         68.9%
  NeuSpell SC-LSTM: 76.7%
  NeuSpell BERT:    79.1%

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python eval/bea60k_benchmark.py \
        --model checkpoints/pretrain/step_140000.pt \
        --tokenizer tokenizer/tokenizer.model \
        --max-sentences 2000
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import torch
import sentencepiece as spm

from model.architecture import NeuralSpellModel

BOS_ID = 2
EOS_ID = 3
BENCHMARK_DIR = Path("eval/benchmarks")


def load_bea60k(max_sentences: int = 0) -> list[tuple[str, str]]:
    """Load BEA-60K as (noisy, clean) pairs."""
    clean_path = BENCHMARK_DIR / "test.bea60k"
    noisy_path = BENCHMARK_DIR / "test.bea60k.noise"
    pairs = []
    with open(clean_path) as cf, open(noisy_path) as nf:
        for clean_line, noisy_line in zip(cf, nf):
            clean = clean_line.strip()
            noisy = noisy_line.strip()
            if clean and noisy and clean != noisy:
                pairs.append((noisy, clean))
                if max_sentences and len(pairs) >= max_sentences:
                    break
    return pairs


def word_correction_rate(pairs_results: list[tuple[str, str, str]]) -> dict:
    """Compute NeuSpell-style word correction rate.

    For each (noisy, clean, predicted) triple, find words that differ
    between noisy and clean (the errors), then check if predicted
    matches clean at those positions.

    Returns: {total_errors, corrected, wrong_corrections, correction_rate}
    """
    total_errors = 0
    corrected = 0
    wrong_corrections = 0  # changed a correct word

    for noisy, clean, predicted in pairs_results:
        nw = noisy.split()
        cw = clean.split()
        pw = predicted.split()

        # BEA-60K guarantees same token count between noisy and clean
        for i in range(min(len(nw), len(cw))):
            if nw[i] != cw[i]:  # this position has an error
                total_errors += 1
                if i < len(pw) and pw[i] == cw[i]:
                    corrected += 1
            else:
                # Correct word — did we break it?
                if i < len(pw) and pw[i] != cw[i]:
                    wrong_corrections += 1

    correction_rate = corrected / total_errors * 100 if total_errors > 0 else 0
    return {
        "total_errors": total_errors,
        "corrected": corrected,
        "wrong_corrections": wrong_corrections,
        "correction_rate": correction_rate,
    }


# ─── Systems ─────────────────────────────────────────────────────

class NeuralSpellSystem:
    def __init__(self, model_path, tokenizer_path, device):
        self.device = device
        self.model = NeuralSpellModel()
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))
        step = ckpt.get("step", "?")
        self.name = f"NeuralSpell 385M (step {step})"

    def correct(self, sentence):
        ids = self.sp.Encode(sentence)[:256]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(inp)
        with torch.no_grad():
            enc = self.model.encode(inp, mask)
        gen = [BOS_ID]
        for _ in range(256):
            dec = torch.tensor([gen], dtype=torch.long, device=self.device)
            dm = torch.ones_like(dec)
            with torch.no_grad():
                logits = self.model.decode(dec, dm, enc, mask)
            tok = logits[0, -1, :].argmax().item()
            if tok == EOS_ID:
                break
            gen.append(tok)
        return self.sp.Decode(gen[1:]).strip()


class BARTSystem:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model="oliverguhr/spelling-correction-english-base",
            device=-1,
        )
        self.name = "BART-base (139M)"

    def correct(self, sentence):
        return self.pipe(sentence, max_length=512)[0]["generated_text"].strip()


class AspellSystem:
    def __init__(self):
        self.name = "aspell"

    def correct(self, sentence):
        tokens = re.findall(r"[\w']+|[^\w\s]+|\s+", sentence)
        return "".join(
            self._correct_word(t) if re.match(r"[\w']+", t) and len(t) > 1 else t
            for t in tokens
        )

    def _correct_word(self, word):
        try:
            result = subprocess.run(
                ["aspell", "-a", "--lang=en"],
                input=word, capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("&"):
                    parts = line.split(": ", 1)
                    if len(parts) > 1:
                        return parts[1].split(",")[0].strip()
                elif line.startswith("*"):
                    return word
            return word
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return word


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BEA-60K standard benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=2000,
                        help="Max sentences to test (0=all 63K, default 2000 for speed)")
    parser.add_argument("--skip-bart", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BEA-60K...")
    pairs = load_bea60k(max_sentences=args.max_sentences)
    print(f"  {len(pairs)} sentence pairs with errors")

    print("Loading systems...")
    systems = []
    print("  NeuralSpell...")
    systems.append(NeuralSpellSystem(args.model, args.tokenizer, device))
    if not args.skip_bart:
        print("  BART-base...")
        systems.append(BARTSystem())
    print("  aspell...")
    systems.append(AspellSystem())

    for sys in systems:
        print(f"\nRunning {sys.name}...")
        results = []
        t0 = time.time()
        for i, (noisy, clean) in enumerate(pairs):
            predicted = sys.correct(noisy)
            results.append((noisy, clean, predicted))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(pairs)}...", end="\r")
        elapsed = time.time() - t0

        metrics = word_correction_rate(results)
        print(f"\n  {sys.name}:")
        print(f"    Word correction rate: {metrics['correction_rate']:.1f}%")
        print(f"    Errors found: {metrics['total_errors']}")
        print(f"    Corrected: {metrics['corrected']}")
        print(f"    Wrong corrections: {metrics['wrong_corrections']}")
        print(f"    Time: {elapsed:.1f}s ({elapsed/len(pairs)*1000:.1f}ms/sentence)")

    # Published baselines for reference
    print(f"\n{'=' * 60}")
    print("Published baselines (NeuSpell, EMNLP 2020, full 63K):")
    print("  aspell:           48.7%")
    print("  JamSpell:         68.9%")
    print("  NeuSpell SC-LSTM: 76.7%")
    print("  NeuSpell BERT:    79.1%")


if __name__ == "__main__":
    main()
