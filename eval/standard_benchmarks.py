#!/usr/bin/env python3
"""Run standard spell-correction benchmarks against NeuralSpell and aspell.

Benchmarks:
  - Birkbeck misspelling corpus (36K+ misspellings)
  - Norvig spell-testset1 and spell-testset2

Each benchmark tests isolated word correction: given a misspelling,
produce the correct word. Reports accuracy for both systems.

Usage:
    PYTHONPATH=. python eval/standard_benchmarks.py \
        --model checkpoints/pretrain/step_70000.pt \
        --tokenizer tokenizer/tokenizer.model

    # CPU only (while GPU trains):
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python eval/standard_benchmarks.py \
        --model checkpoints/pretrain/step_70000.pt \
        --tokenizer tokenizer/tokenizer.model
"""

import argparse
import subprocess
import time
from pathlib import Path

import torch
import sentencepiece as spm

from model.architecture import NeuralSpellModel

BOS_ID = 2
EOS_ID = 3

BENCHMARK_DIR = Path("eval/benchmarks")


def load_model(model_path: Path, device: torch.device) -> NeuralSpellModel:
    model = NeuralSpellModel()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def neuralspell_correct(model, tokenizer, word: str, device: torch.device, max_length: int = 32) -> str:
    """Correct a single word using NeuralSpell autoregressive decoding."""
    input_ids = tokenizer.Encode(word)[:max_length]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    input_mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        encoder_output = model.encode(input_tensor, input_mask)

    generated = [BOS_ID]
    for _ in range(max_length):
        dec_input = torch.tensor([generated], dtype=torch.long, device=device)
        dec_mask = torch.ones_like(dec_input)
        with torch.no_grad():
            logits = model.decode(dec_input, dec_mask, encoder_output, input_mask)
        next_token = logits[0, -1, :].argmax().item()
        if next_token == EOS_ID:
            break
        generated.append(next_token)

    return tokenizer.Decode(generated[1:]).strip()


def aspell_correct(word: str) -> str:
    """Correct a single word using aspell. Returns top suggestion or original."""
    try:
        result = subprocess.run(
            ["aspell", "-a", "--lang=en"],
            input=word,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("&"):
                # & word offset count: suggestion1, suggestion2, ...
                suggestions = line.split(": ", 1)
                if len(suggestions) > 1:
                    return suggestions[1].split(",")[0].strip()
            elif line.startswith("*"):
                return word  # already correct
        return word
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return word


# ─── Benchmark parsers ───────────────────────────────────────────

def parse_birkbeck(path: Path, max_pairs: int = 0) -> list[tuple[str, str]]:
    """Parse Birkbeck format: $correct followed by misspellings."""
    pairs = []
    correct = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("$"):
            correct = line[1:].replace("_", " ")
        elif correct:
            misspelling = line.replace("_", " ")
            pairs.append((misspelling, correct))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def parse_norvig(path: Path) -> list[tuple[str, str]]:
    """Parse Norvig format: correct: wrong1 wrong2 ..."""
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        correct, wrongs = line.split(":", 1)
        correct = correct.strip()
        for wrong in wrongs.strip().split():
            pairs.append((wrong.strip(), correct))
    return pairs


# ─── Runner ──────────────────────────────────────────────────────

def run_benchmark(
    name: str,
    pairs: list[tuple[str, str]],
    model,
    tokenizer,
    device,
):
    """Run a benchmark and report results for both systems."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name} ({len(pairs)} pairs)")
    print(f"{'=' * 60}")

    ns_correct = 0
    as_correct = 0
    ns_times = []
    as_times = []
    errors_ns = []
    errors_as = []

    for i, (wrong, right) in enumerate(pairs):
        # NeuralSpell
        t0 = time.time()
        ns_pred = neuralspell_correct(model, tokenizer, wrong, device)
        ns_times.append(time.time() - t0)
        ns_match = ns_pred.lower() == right.lower()
        if ns_match:
            ns_correct += 1

        # Aspell
        # Only correct single words with aspell (skip multi-word)
        if " " not in wrong:
            t0 = time.time()
            as_pred = aspell_correct(wrong)
            as_times.append(time.time() - t0)
            as_match = as_pred.lower() == right.lower()
        else:
            as_pred = wrong
            as_match = False

        if as_match:
            as_correct += 1

        # Track interesting disagreements
        if ns_match and not as_match:
            errors_as.append((wrong, right, as_pred))
        elif as_match and not ns_match:
            errors_ns.append((wrong, right, ns_pred))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}...", end="\r")

    total = len(pairs)
    print(f"\n  {'':40s} NeuralSpell    aspell")
    print(f"  {'Correct:':40s} {ns_correct:>5d}/{total:<5d}   {as_correct:>5d}/{total:<5d}")
    print(f"  {'Accuracy:':40s} {ns_correct/total*100:>9.1f}%   {as_correct/total*100:>9.1f}%")
    if ns_times:
        print(f"  {'Avg time/word:':40s} {sum(ns_times)/len(ns_times)*1000:>7.1f}ms   {sum(as_times)/len(as_times)*1000:>7.1f}ms")

    # Show examples where one beats the other
    if errors_as[:5]:
        print(f"\n  NeuralSpell got right, aspell got wrong:")
        for wrong, right, as_pred in errors_as[:5]:
            print(f"    {wrong:20s} -> NS: {right:20s}  aspell: {as_pred}")

    if errors_ns[:5]:
        print(f"\n  Aspell got right, NeuralSpell got wrong:")
        for wrong, right, ns_pred in errors_ns[:5]:
            print(f"    {wrong:20s} -> aspell: {right:20s}  NS: {ns_pred}")

    return {
        "name": name,
        "total": total,
        "neuralspell_correct": ns_correct,
        "aspell_correct": as_correct,
        "neuralspell_accuracy": ns_correct / total,
        "aspell_accuracy": as_correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Standard spell-correction benchmarks")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--birkbeck-limit", type=int, default=1000,
                        help="Max Birkbeck pairs to test (0=all, default 1000 for speed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.model, device)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(args.tokenizer))

    print("Checking aspell...")
    test = aspell_correct("teh")
    print(f"  aspell('teh') = '{test}'")

    results = []

    # Norvig test sets
    for name, filename in [("Norvig Test 1", "norvig_test1.txt"), ("Norvig Test 2", "norvig_test2.txt")]:
        path = BENCHMARK_DIR / filename
        if path.exists():
            pairs = parse_norvig(path)
            if pairs:
                results.append(run_benchmark(name, pairs, model, tokenizer, device))

    # Birkbeck
    birkbeck_path = BENCHMARK_DIR / "birkbeck_missp.dat"
    if birkbeck_path.exists():
        pairs = parse_birkbeck(birkbeck_path, max_pairs=args.birkbeck_limit)
        if pairs:
            results.append(run_benchmark(
                f"Birkbeck (n={len(pairs)})", pairs, model, tokenizer, device
            ))

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Benchmark':30s} {'NeuralSpell':>12s} {'aspell':>12s}")
        print(f"  {'-'*30} {'-'*12} {'-'*12}")
        for r in results:
            print(f"  {r['name']:30s} {r['neuralspell_accuracy']*100:>10.1f}%  {r['aspell_accuracy']*100:>10.1f}%")


if __name__ == "__main__":
    main()
