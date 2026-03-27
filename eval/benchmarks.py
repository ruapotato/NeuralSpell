"""Evaluation harness for NeuralSpell encoder-decoder.

Runs the trained model against benchmark test sets using
autoregressive greedy decoding. Uses difflib for word-level
alignment since the model can produce variable-length output.
"""

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import torch
import sentencepiece as spm

from model.architecture import NeuralSpellModel
from training.eval import CorrectionMetrics

BOS_ID = 2
EOS_ID = 3


def load_model(model_path: Path, device: torch.device) -> NeuralSpellModel:
    model = NeuralSpellModel()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def correct_sentence(
    model: NeuralSpellModel,
    tokenizer: spm.SentencePieceProcessor,
    sentence: str,
    device: torch.device,
    max_length: int = 256,
) -> str:
    """Run inference: correct a single sentence via autoregressive decoding."""
    input_ids = tokenizer.Encode(sentence)[:max_length]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    input_mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        encoder_output = model.encode(input_tensor, input_mask)

    # Greedy autoregressive decoding
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

    return tokenizer.Decode(generated[1:])  # skip BOS


def align_words(input_words: list[str], correct_words: list[str], predicted_words: list[str]):
    """Align words using SequenceMatcher for variable-length comparison.

    Returns list of (input_word, correct_word, predicted_word) tuples,
    where None indicates a gap (insertion/deletion).
    """
    # First align input with correct to identify which words were corrupted
    inp_corr = SequenceMatcher(None, input_words, correct_words)
    # Then align correct with predicted to see what the model got right
    corr_pred = SequenceMatcher(None, correct_words, predicted_words)

    return inp_corr, corr_pred


def eval_test_file(
    model: NeuralSpellModel,
    tokenizer: spm.SentencePieceProcessor,
    test_file: Path,
    device: torch.device,
    verbose: bool = False,
) -> CorrectionMetrics:
    """Evaluate on a test file with format: corrupted<TAB>correct per line."""
    metrics = CorrectionMetrics()

    lines = test_file.read_text().strip().split("\n")
    for line in lines:
        if "\t" not in line:
            continue
        corrupted, correct = line.split("\t", 1)
        predicted = correct_sentence(model, tokenizer, corrupted, device)

        inp_words = corrupted.split()
        corr_words = correct.split()
        pred_words = predicted.split()

        # Use SequenceMatcher to align predicted with correct
        # This handles cases where the model produces more/fewer words
        matcher = SequenceMatcher(None, corr_words, pred_words)
        inp_matcher = SequenceMatcher(None, inp_words, corr_words)

        # Build a set of correct-word indices that were corrupted
        corrupted_indices = set()
        for tag, i1, i2, j1, j2 in inp_matcher.get_opcodes():
            if tag != "equal":
                for j in range(j1, j2):
                    corrupted_indices.add(j)

        # Score based on alignment of predicted with correct
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    metrics.total_words += 1
                    if i in corrupted_indices:
                        metrics.correct_corrections += 1
                    else:
                        metrics.correct_passthrough += 1
            elif tag == "replace":
                for i in range(i1, i2):
                    metrics.total_words += 1
                    if i in corrupted_indices:
                        metrics.missed_errors += 1
                    else:
                        metrics.false_positives += 1
            elif tag == "delete":
                # Words in correct but not in predicted
                for i in range(i1, i2):
                    metrics.total_words += 1
                    if i in corrupted_indices:
                        metrics.missed_errors += 1
                    else:
                        metrics.false_positives += 1
            elif tag == "insert":
                # Extra words in predicted — count as false positives
                metrics.total_words += (j2 - j1)
                metrics.false_positives += (j2 - j1)

        if verbose:
            print(f"  Input:     {corrupted}")
            print(f"  Predicted: {predicted}")
            print(f"  Correct:   {correct}")
            print()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuralSpell")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("eval/test_sets"))
    parser.add_argument("--verbose", action="store_true", help="Print each prediction")
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
        metrics = eval_test_file(model, tokenizer, test_file, device, verbose=args.verbose)
        print(metrics)


if __name__ == "__main__":
    main()
