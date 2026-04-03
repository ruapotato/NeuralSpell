"""Benchmark the hybrid aspell+NN model on BEA-60K and JFLEG.

Same interface as other benchmarks but feeds aspell suggestions
into the encoder alongside the corrupted text.

Usage:
    PYTHONPATH=. python eval/hybrid_benchmark.py \
        --model checkpoints/hybrid/best.pt \
        --tokenizer tokenizer/tokenizer.model
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import torch
import sentencepiece as spm

from model.architecture import NeuralSpellModel
from eval.bea60k_benchmark import load_bea60k, word_correction_rate, AspellSystem

BOS_ID = 2
EOS_ID = 3


def aspell_correct_sentence(sentence):
    try:
        result = subprocess.run(
            ["aspell", "-a", "--lang=en"],
            input=sentence, capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return sentence
    words = sentence.split()
    corrected = list(words)
    suggestions = {}
    for line in result.stdout.split("\n"):
        if line.startswith("&"):
            parts = line.split()
            original = parts[1]
            sugg = line.split(": ", 1)
            if len(sugg) > 1:
                suggestions[original] = sugg[1].split(",")[0].strip()
    for i, w in enumerate(corrected):
        clean = re.sub(r"[^\w'-]", "", w)
        if clean in suggestions:
            corrected[i] = w.replace(clean, suggestions[clean])
    return " ".join(corrected)


class HybridSystem:
    def __init__(self, model_path, tokenizer_path, device):
        self.device = device
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        mcfg = ckpt.get("model_config", {})
        self.model = NeuralSpellModel(**mcfg) if mcfg else NeuralSpellModel()
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(tokenizer_path))
        self.sep_id = self.sp.PieceToId("[SEP]")
        step = ckpt.get("step", "?")
        self.name = f"Hybrid 60M (step {step})"

    def correct(self, sentence):
        # Get aspell suggestion
        aspell_fixed = aspell_correct_sentence(sentence)

        # Encode: "corrupted [SEP] aspell_fixed"
        corrupt_ids = self.sp.Encode(sentence)[:255]
        aspell_ids = self.sp.Encode(aspell_fixed)[:255]
        encoder_ids = corrupt_ids + [self.sep_id] + aspell_ids

        inp = torch.tensor([encoder_ids], dtype=torch.long, device=self.device)
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


def main():
    parser = argparse.ArgumentParser(description="Hybrid model benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading systems...")
    hybrid = HybridSystem(args.model, args.tokenizer, device)
    aspell = AspellSystem()

    print("Loading BEA-60K...")
    pairs = load_bea60k(max_sentences=args.max_sentences)
    print(f"  {len(pairs)} sentence pairs")

    for sys in [hybrid, aspell]:
        print(f"\nRunning {sys.name}...")
        results = []
        t0 = time.time()
        for i, (noisy, clean) in enumerate(pairs):
            results.append((noisy, clean, sys.correct(noisy)))
            if (i + 1) % 50 == 0:
                m = word_correction_rate(results)
                print(f"  {i+1}/{len(pairs)} corr:{m['correction_rate']:.1f}% FP:{m['wrong_corrections']}", flush=True)
        elapsed = time.time() - t0
        m = word_correction_rate(results)
        print(f"\n  {sys.name}:")
        print(f"    Correction rate: {m['correction_rate']:.1f}% ({m['corrected']}/{m['total_errors']})")
        print(f"    Wrong corrections: {m['wrong_corrections']}")
        print(f"    Time: {elapsed:.1f}s")

    print(f"\nPublished baselines (NeuSpell, full 63K):")
    print(f"  aspell: 48.7% | NeuSpell BERT: 79.1%")


if __name__ == "__main__":
    main()
