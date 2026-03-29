#!/usr/bin/env python3
"""Sentence-level spell correction benchmark: NeuralSpell vs aspell.

Tests both systems on full corrupted sentences — the actual use case.
Aspell can only correct isolated words without context, so this shows
where a contextual model adds value.

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python eval/sentence_benchmarks.py \
        --model checkpoints/pretrain/step_70000.pt \
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
from corruption.engine import CorruptionEngine

BOS_ID = 2
EOS_ID = 3

# Sentences covering different domains and lengths
TEST_SENTENCES = [
    "The quick brown fox jumped over the lazy dog.",
    "She received the package in the mail yesterday afternoon.",
    "He was definitely going to the concert later today.",
    "The government announced new policies for education reform.",
    "I believe that separate rooms would be more efficient.",
    "The restaurant was beautiful and the food was excellent.",
    "Please be discreet when handling this sensitive information.",
    "The accommodation was convenient and reasonably priced overall.",
    "I went over there to see their house after work.",
    "You're the best person I know and I appreciate your help.",
    "It's going to rain today so bring its favorite umbrella.",
    "They're going to the store to buy some food for dinner.",
    "He knew the right answer all along but stayed quiet.",
    "The effect of the medicine was immediate and noticeable.",
    "The principal of the school gave a speech at graduation.",
    "The cat sat on the mat and looked outside the window.",
    "He went to the store to buy some groceries for dinner.",
    "She could not believe what had happened at the meeting.",
    "The weather was perfect for a long walk in the park today.",
    "I need to finish my work before the deadline tomorrow morning.",
    "The committee decided to postpone the meeting until next week.",
    "He had sufficient knowledge to pass the examination with ease.",
    "The environment requires immediate attention from the government.",
    "Markus built the prototype of the app while recruiting drivers.",
    "The religious spirit will be revivified because it will be in harmony.",
]


def load_model(model_path: Path, device: torch.device) -> NeuralSpellModel:
    model = NeuralSpellModel()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def neuralspell_correct_sentence(model, tokenizer, sentence: str, device, max_length=256) -> str:
    input_ids = tokenizer.Encode(sentence)[:max_length]
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


def aspell_correct_sentence(sentence: str) -> str:
    """Correct each word independently with aspell."""
    # Split preserving punctuation attached to words
    tokens = re.findall(r"[\w']+|[^\w\s]+|\s+", sentence)
    corrected = []
    for token in tokens:
        if re.match(r"[\w']+", token) and len(token) > 1:
            corrected.append(aspell_correct_word(token))
        else:
            corrected.append(token)
    return "".join(corrected)


def aspell_correct_word(word: str) -> str:
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
                suggestions = line.split(": ", 1)
                if len(suggestions) > 1:
                    return suggestions[1].split(",")[0].strip()
            elif line.startswith("*"):
                return word
        return word
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return word


def word_level_score(predicted: str, original: str) -> tuple[int, int]:
    """Count correct words (case-insensitive)."""
    pred_words = predicted.lower().split()
    orig_words = original.lower().split()
    correct = sum(1 for p, o in zip(pred_words, orig_words) if p == o)
    total = max(len(orig_words), 1)
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Sentence-level spell correction benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corruption-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.model, device)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(args.tokenizer))

    engine = CorruptionEngine(seed=args.seed)

    # Generate corrupted versions
    pairs = []
    for sentence in TEST_SENTENCES:
        corrupted = engine.corrupt_sentence(sentence, args.corruption_rate)
        if corrupted != sentence:  # skip if no corruption happened
            pairs.append((corrupted, sentence))

    print(f"\n{len(pairs)} test sentences, corruption rate {args.corruption_rate}")
    print(f"{'=' * 70}")

    ns_total_correct = 0
    ns_total_words = 0
    as_total_correct = 0
    as_total_words = 0

    for i, (corrupted, original) in enumerate(pairs):
        ns_pred = neuralspell_correct_sentence(model, tokenizer, corrupted, device)
        as_pred = aspell_correct_sentence(corrupted)

        ns_c, ns_t = word_level_score(ns_pred, original)
        as_c, as_t = word_level_score(as_pred, original)
        ns_total_correct += ns_c
        ns_total_words += ns_t
        as_total_correct += as_c
        as_total_words += as_t

        ns_pct = ns_c / ns_t * 100
        as_pct = as_c / as_t * 100
        winner = "NS" if ns_pct > as_pct else ("aspell" if as_pct > ns_pct else "tie")

        print(f"\n[{i+1}] {winner}")
        print(f"  Corrupted:    {corrupted}")
        print(f"  NeuralSpell:  {ns_pred}  ({ns_c}/{ns_t})")
        print(f"  Aspell:       {as_pred}  ({as_c}/{as_t})")
        print(f"  Original:     {original}")

    print(f"\n{'=' * 70}")
    print(f"SENTENCE-LEVEL RESULTS")
    print(f"{'=' * 70}")
    ns_acc = ns_total_correct / ns_total_words * 100
    as_acc = as_total_correct / as_total_words * 100
    print(f"  {'':35s} NeuralSpell    aspell")
    print(f"  {'Words correct:':35s} {ns_total_correct:>5d}/{ns_total_words:<5d}   {as_total_correct:>5d}/{as_total_words:<5d}")
    print(f"  {'Word accuracy:':35s} {ns_acc:>9.1f}%   {as_acc:>9.1f}%")


if __name__ == "__main__":
    main()
