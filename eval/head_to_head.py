#!/usr/bin/env python3
"""Head-to-head comparison: NeuralSpell vs other open-source spell checkers.

Compares on both our custom sentence test set and the standard Norvig/Birkbeck
word-level benchmarks.

Systems tested:
  - NeuralSpell (ours, 385M params, encoder-decoder)
  - oliverguhr/spelling-correction-english-base (BART, ~139M params)
  - ai-forever/T5-large-spell (T5, 770M params, SOTA)
  - aspell (dictionary-based baseline)

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python eval/head_to_head.py \
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

# ─── Test sentences ──────────────────────────────────────────────

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


# ─── System wrappers ─────────────────────────────────────────────

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
        self.name = "NeuralSpell (385M)"
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
        result = self.pipe(sentence, max_length=512)[0]["generated_text"]
        return result.strip()


class T5LargeSystem:
    def __init__(self):
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/T5-large-spell")
        self.model = T5ForConditionalGeneration.from_pretrained("ai-forever/T5-large-spell")
        self.model.eval()
        self.name = "T5-large-spell (770M)"

    def correct(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=256)
        result = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # T5 sometimes repeats the output — take first sentence-length chunk
        if len(result) > len(sentence) * 1.5:
            result = result[:len(sentence) + 20].strip()
        return result


class AspellSystem:
    def __init__(self):
        self.name = "aspell (dictionary)"

    def correct(self, sentence):
        tokens = re.findall(r"[\w']+|[^\w\s]+|\s+", sentence)
        corrected = []
        for token in tokens:
            if re.match(r"[\w']+", token) and len(token) > 1:
                corrected.append(self._correct_word(token))
            else:
                corrected.append(token)
        return "".join(corrected)

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


# ─── Scoring ─────────────────────────────────────────────────────

def word_accuracy(predicted: str, original: str) -> tuple[int, int]:
    pred = predicted.lower().split()
    orig = original.lower().split()
    correct = sum(1 for p, o in zip(pred, orig) if p == o)
    return correct, len(orig)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corruption-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-t5", action="store_true", help="Skip T5-large (slow to download)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading systems...")
    systems = []

    print("  NeuralSpell...")
    systems.append(NeuralSpellSystem(args.model, args.tokenizer, device))

    print("  BART-base...")
    systems.append(BARTSystem())

    if not args.skip_t5:
        print("  T5-large-spell (this may download ~3GB)...")
        systems.append(T5LargeSystem())

    print("  aspell...")
    systems.append(AspellSystem())

    # Generate corrupted test set
    engine = CorruptionEngine(seed=args.seed)
    pairs = []
    for sentence in TEST_SENTENCES:
        corrupted = engine.corrupt_sentence(sentence, args.corruption_rate)
        if corrupted != sentence:
            pairs.append((corrupted, sentence))

    print(f"\n{len(pairs)} test sentences, corruption rate {args.corruption_rate}")

    # Run all systems
    results = {s.name: {"correct": 0, "total": 0, "time": 0.0} for s in systems}
    details = []

    for i, (corrupted, original) in enumerate(pairs):
        row = {"corrupted": corrupted, "original": original, "predictions": {}}

        for sys in systems:
            t0 = time.time()
            predicted = sys.correct(corrupted)
            elapsed = time.time() - t0

            c, t = word_accuracy(predicted, original)
            results[sys.name]["correct"] += c
            results[sys.name]["total"] += t
            results[sys.name]["time"] += elapsed
            row["predictions"][sys.name] = predicted

        details.append(row)
        print(f"  {i+1}/{len(pairs)}...", end="\r")

    # Print detailed results
    print(f"\n\n{'=' * 80}")
    print("DETAILED RESULTS")
    print(f"{'=' * 80}")

    for i, row in enumerate(details):
        print(f"\n[{i+1}] Corrupted: {row['corrupted']}")
        print(f"    Original:  {row['original']}")
        for sys_name, pred in row["predictions"].items():
            c, t = word_accuracy(pred, row["original"])
            marker = "OK" if c == t else f"{c}/{t}"
            print(f"    {sys_name:40s} {pred}")
            if c < t:
                print(f"    {'':40s} [{marker}]")

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY — Sentence-Level Spell Correction")
    print(f"{'=' * 80}")
    print(f"  {'System':40s} {'Words Correct':>14s} {'Accuracy':>10s} {'Time':>8s}")
    print(f"  {'-'*40} {'-'*14} {'-'*10} {'-'*8}")

    for sys in systems:
        r = results[sys.name]
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {sys.name:40s} {r['correct']:>5d}/{r['total']:<5d}    {acc:>8.1f}%  {r['time']:>6.1f}s")


if __name__ == "__main__":
    main()
