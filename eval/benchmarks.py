"""Evaluation harness for NeuralSpell.

Runs the trained model against standard benchmarks:
  - Birkbeck spelling error corpus
  - Homophone test cases
  - Context-required corrections
"""

import argparse
from pathlib import Path

import torch
import sentencepiece as spm

from model.architecture import NeuralSpellModel
from training.eval import CorrectionMetrics


def load_model(model_path: Path, device: torch.device) -> NeuralSpellModel:
    model = NeuralSpellModel()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def correct_sentence(
    model: NeuralSpellModel,
    tokenizer: spm.SentencePieceProcessor,
    sentence: str,
    device: torch.device,
) -> str:
    """Run inference: correct a single sentence."""
    input_ids = tokenizer.Encode(sentence)
    input_ids = input_ids[:256]  # truncate

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor, attention_mask)
        predicted_ids = logits.argmax(dim=-1)[0].tolist()

    # Decode only up to the original length
    predicted_ids = predicted_ids[: len(input_ids)]
    return tokenizer.Decode(predicted_ids)


def eval_test_file(
    model: NeuralSpellModel,
    tokenizer: spm.SentencePieceProcessor,
    test_file: Path,
    device: torch.device,
) -> CorrectionMetrics:
    """Evaluate on a test file with format: corrupted<TAB>correct per line."""
    metrics = CorrectionMetrics()

    lines = test_file.read_text().strip().split("\n")
    for line in lines:
        if "\t" not in line:
            continue
        corrupted, correct = line.split("\t", 1)
        predicted = correct_sentence(model, tokenizer, corrupted, device)

        # Word-level comparison
        pred_words = predicted.split()
        corr_words = correct.split()
        inp_words = corrupted.split()

        for i in range(min(len(pred_words), len(corr_words), len(inp_words))):
            metrics.total_words += 1
            was_corrupted = inp_words[i] != corr_words[i]

            if was_corrupted:
                if pred_words[i] == corr_words[i]:
                    metrics.correct_corrections += 1
                else:
                    metrics.missed_errors += 1
            else:
                if pred_words[i] == corr_words[i]:
                    metrics.correct_passthrough += 1
                else:
                    metrics.false_positives += 1

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuralSpell")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("eval/test_sets"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(args.tokenizer))

    test_files = sorted(args.test_dir.glob("*.txt"))
    if not test_files:
        print(f"No test files found in {args.test_dir}")
        return

    for test_file in test_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmark: {test_file.name}")
        print(f"{'=' * 50}")
        metrics = eval_test_file(model, tokenizer, test_file, device)
        print(metrics)


if __name__ == "__main__":
    main()
