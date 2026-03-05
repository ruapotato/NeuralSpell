"""Clean and normalize downloaded text corpora.

Parallel processing: each worker handles a subset of input shards.
Two-pass approach:
  1. Workers extract+clean sentences in parallel (no cross-file dedup)
  2. Fast dedup pass merges results, removing duplicates via compact hashes

Streams files line-by-line to avoid loading multi-GB files into memory.
"""

import argparse
import hashlib
import os
import re
import struct
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

MIN_ALPHA_RATIO = 0.7
MIN_SENTENCE_CHARS = 20
MAX_SENTENCE_CHARS = 2000
MIN_WORDS = 4

_WHITESPACE = re.compile(r"[ \t]+")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SENT_BOUNDARY_FALLBACK = re.compile(r"(?<=[.!?])\s+")


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = line.replace("\u2018", "'").replace("\u2019", "'")
    line = line.replace("\u201c", '"').replace("\u201d", '"')
    line = line.replace("\u2013", "-").replace("\u2014", "--")
    line = _WHITESPACE.sub(" ", line)
    return line.strip()


def is_english(line: str) -> bool:
    if not line:
        return False
    alpha = sum(1 for c in line if c.isascii() and c.isalpha())
    total = len(line) - line.count(" ")
    if total == 0:
        return False
    return (alpha / total) >= MIN_ALPHA_RATIO


def extract_sentences(line: str) -> list[str]:
    parts = _SENT_BOUNDARY.split(line)
    sentences = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < MIN_SENTENCE_CHARS:
            continue
        if len(part) > MAX_SENTENCE_CHARS:
            for sp in _SENT_BOUNDARY_FALLBACK.split(part):
                sp = sp.strip()
                if MIN_SENTENCE_CHARS <= len(sp) <= MAX_SENTENCE_CHARS:
                    sentences.append(sp)
        else:
            sentences.append(part)
    return sentences


def compact_hash(text: str) -> int:
    """8-byte hash for memory-efficient deduplication."""
    digest = hashlib.md5(text.lower().encode()).digest()
    return struct.unpack("<Q", digest[:8])[0]


def stream_paragraphs(filepath: Path):
    """Yield paragraphs from a file, streaming line-by-line."""
    buf = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                if buf:
                    yield " ".join(buf)
                    buf = []
            else:
                buf.append(line.strip())
    if buf:
        yield " ".join(buf)


def process_shard(shard_path: str, output_path: str) -> tuple[int, int]:
    """Process a single shard file. Returns (sentences_written, total_paragraphs)."""
    shard_path = Path(shard_path)
    output_path = Path(output_path)
    sentences = []
    para_count = 0

    for paragraph in stream_paragraphs(shard_path):
        para_count += 1
        paragraph = normalize_line(paragraph)
        if not paragraph:
            continue
        for sent in extract_sentences(paragraph):
            if not is_english(sent):
                continue
            if len(sent.split()) < MIN_WORDS:
                continue
            sentences.append(sent)

    if sentences:
        output_path.write_text("\n".join(sentences), encoding="utf-8")

    return len(sentences), para_count


def dedup_pass(temp_dir: Path, output_dir: Path, batch_limit: int = 100_000):
    """Read all temp shard files and deduplicate using compact hashes."""
    print("Pass 2: Deduplicating...")
    seen: set[int] = set()
    total_sentences = 0
    total_dupes = 0
    out_idx = 0
    out_batch: list[str] = []

    temp_files = sorted(temp_dir.glob("temp_*.txt"))

    for temp_file in tqdm(temp_files, desc="Dedup"):
        with open(temp_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                h = compact_hash(line)
                if h in seen:
                    total_dupes += 1
                    continue
                seen.add(h)
                out_batch.append(line)
                total_sentences += 1

                if len(out_batch) >= batch_limit:
                    out_path = output_dir / f"clean_{out_idx:05d}.txt"
                    out_path.write_text("\n".join(out_batch), encoding="utf-8")
                    out_batch = []
                    out_idx += 1

    if out_batch:
        out_path = output_dir / f"clean_{out_idx:05d}.txt"
        out_path.write_text("\n".join(out_batch), encoding="utf-8")
        out_idx += 1

    return total_sentences, total_dupes, out_idx


def clean_corpus(input_dir: Path, output_dir: Path, num_workers: int = 8):
    """Two-pass parallel cleaning pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(input_dir.rglob("*.txt"))
    shard_files = [f for f in shard_files if f.name != "manifest.json"]
    print(f"Found {len(shard_files)} shard files to process with {num_workers} workers")

    # Pass 1: parallel sentence extraction
    print("Pass 1: Extracting and cleaning sentences...")
    futures = {}
    total_sentences_p1 = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, shard_path in enumerate(shard_files):
            temp_out = temp_dir / f"temp_{i:05d}.txt"
            future = executor.submit(process_shard, str(shard_path), str(temp_out))
            futures[future] = shard_path

        with tqdm(total=len(shard_files), desc="Extract") as pbar:
            for future in as_completed(futures):
                n_sents, n_paras = future.result()
                total_sentences_p1 += n_sents
                pbar.update(1)

    print(f"Pass 1 complete: {total_sentences_p1:,} sentences extracted")

    # Pass 2: sequential dedup (needs shared hash set)
    total_sentences, total_dupes, out_idx = dedup_pass(temp_dir, output_dir)

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\nCleaning complete:")
    print(f"  Sentences before dedup: {total_sentences_p1:,}")
    print(f"  Duplicates removed: {total_dupes:,}")
    print(f"  Final sentences: {total_sentences:,}")
    print(f"  Output shards: {out_idx}")
    print(f"  Output dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Clean and normalize corpora")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input directory {args.input_dir} does not exist.")
        print("Run 'python data/download.py' first.")
        raise SystemExit(1)

    clean_corpus(args.input_dir, args.output_dir, args.workers)
    print("\nNext step: python phonetics/build_ipa_db.py")


if __name__ == "__main__":
    main()
