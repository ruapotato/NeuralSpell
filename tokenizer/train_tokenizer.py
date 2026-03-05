"""Train a SentencePiece BPE tokenizer on the cleaned corpus.

Target: 32000 vocab size, trained on a representative sample.
"""

import argparse
from pathlib import Path

import sentencepiece as spm


def collect_training_text(input_dir: Path, output_file: Path, max_lines: int = 10_000_000):
    """Collect text from cleaned shards into a single file for tokenizer training.

    SentencePiece trains from a single text file. We sample up to max_lines
    from the corpus to keep training fast while ensuring good coverage.
    """
    print(f"Collecting training text from {input_dir}...")
    shards = sorted(input_dir.glob("clean_*.txt"))
    if not shards:
        raise FileNotFoundError(f"No clean_*.txt files found in {input_dir}")

    lines_written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for shard in shards:
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        lines_written += 1
                        if lines_written >= max_lines:
                            break
            if lines_written >= max_lines:
                break

    print(f"Collected {lines_written:,} lines for tokenizer training")
    return output_file


def train_tokenizer(
    input_file: Path,
    output_prefix: str,
    vocab_size: int = 32000,
):
    """Train SentencePiece BPE tokenizer."""
    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,  # full coverage for English
        num_threads=8,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Reserve IDs for [MASK] and [CLS] and [SEP]
        user_defined_symbols=["[MASK]", "[CLS]", "[SEP]"],
        byte_fallback=True,  # handle any unicode
    )
    print(f"Tokenizer saved to {output_prefix}.model")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("tokenizer/tokenizer.model"))
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-lines", type=int, default=10_000_000)
    args = parser.parse_args()

    # Output prefix is model path without .model extension
    output_prefix = str(args.output).replace(".model", "")
    temp_file = Path(output_prefix + "_train_text.txt")

    collect_training_text(args.input_dir, temp_file, args.max_lines)
    train_tokenizer(temp_file, output_prefix, args.vocab_size)

    # Clean up temp file
    temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
