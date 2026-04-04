#!/usr/bin/env python3
"""Quick BEA-60K eval for tagger v2 models. Runs on CPU, doesn't touch GPU.

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python eval/tagger_bea60k.py \
        --model checkpoints/tagger_v2/step_10000.pt \
        --tokenizer tokenizer/tokenizer.model
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import torch
import sentencepiece as spm

from model.tagger import SpellTagger
from eval.bea60k_benchmark import load_bea60k, word_correction_rate

KEEP_TAG = 0
DELETE_TAG = 1
REPLACE_OFFSET = 2


def tagger_correct(model, sp, sentence, device, keep_bias=0.0, iterations=3):
    """Apply tagger corrections iteratively."""
    result = sentence
    for _ in range(iterations):
        tokens = sp.Encode(result)
        if not tokens:
            return result
        tokens = tokens[:256]

        inp = torch.tensor([tokens], dtype=torch.long, device=device)
        mask = torch.ones_like(inp)

        with torch.no_grad():
            logits = model(inp, mask)
            if keep_bias > 0:
                logits[:, :, KEEP_TAG] += keep_bias
            preds = logits.argmax(dim=-1)[0].tolist()

        # Apply edits
        new_tokens = []
        changed = False
        for i, (tok, tag) in enumerate(zip(tokens, preds)):
            if tag == KEEP_TAG:
                new_tokens.append(tok)
            elif tag == DELETE_TAG:
                changed = True
            else:
                # REPLACE: tag = token_id + REPLACE_OFFSET
                new_tok = tag - REPLACE_OFFSET
                if 0 <= new_tok < 32000:
                    new_tokens.append(new_tok)
                    if new_tok != tok:
                        changed = True
                else:
                    new_tokens.append(tok)

        if not changed:
            break
        result = sp.Decode(new_tokens)

    return result


def aspell_correct(sentence):
    try:
        result = subprocess.run(
            ["aspell", "-a", "--lang=en"],
            input=sentence, capture_output=True, text=True, timeout=5,
        )
    except:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument("--keep-bias", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading model from {args.model}...")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    mcfg = ckpt.get("model_config", {})
    if "num_tags" not in mcfg:
        mcfg["num_tags"] = 32002
    model = SpellTagger(**mcfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    step = ckpt.get("step", "?")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    pairs = load_bea60k(max_sentences=args.max_sentences)
    print(f"BEA-60K: {len(pairs)} pairs, step {step}\n")

    # Aspell baseline
    as_results = [(n, c, aspell_correct(n)) for n, c in pairs]
    am = word_correction_rate(as_results)
    print(f"aspell:     {am['correction_rate']:5.1f}% corr, {am['wrong_corrections']:4d} FP")

    # Tagger at different bias values
    for bias in args.keep_bias:
        t0 = time.time()
        ns_results = [(n, c, tagger_correct(model, sp, n, device, keep_bias=bias))
                      for n, c in pairs]
        elapsed = time.time() - t0
        nm = word_correction_rate(ns_results)
        print(f"bias={bias:<4.1f}  {nm['correction_rate']:5.1f}% corr, {nm['wrong_corrections']:4d} FP  ({elapsed:.1f}s)")

    print(f"\n385M enc-dec: 64.4% corr, 699 FP")
    print(f"Published: aspell 48.7%, NeuSpell BERT 79.1%")


if __name__ == "__main__":
    main()
