"""Build IPA pronunciation database using espeak-ng.

Runs espeak-ng --ipa on every word in a frequency-sorted word list
to produce a JSON mapping of word -> IPA transcription.

Requires: espeak-ng (apt install espeak-ng)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# Default word list: top 100K English words by frequency.
# We'll build this from the training corpus if no external list is provided.
DEFAULT_WORDLIST = Path("data/processed/wordlist.txt")


def get_ipa(word: str) -> str | None:
    """Get IPA transcription of a word via espeak-ng."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-q", "--", word],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ipa = result.stdout.strip()
        # espeak-ng sometimes returns multiple pronunciations separated by newlines
        if ipa:
            return ipa.split("\n")[0].strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_ipa_batch(words: list[str], batch_size: int = 100) -> dict[str, str]:
    """Get IPA for many words efficiently by batching espeak-ng calls.

    Pipes words via stdin to avoid per-word process overhead.
    """
    ipa_map = {}

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        input_text = "\n".join(batch)
        try:
            result = subprocess.run(
                ["espeak-ng", "--ipa", "-q"],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = result.stdout.strip().split("\n")
            for word, ipa_line in zip(batch, lines):
                ipa = ipa_line.strip()
                if ipa:
                    ipa_map[word] = ipa
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fall back to one-by-one
            for word in batch:
                ipa = get_ipa(word)
                if ipa:
                    ipa_map[word] = ipa

    return ipa_map


def _count_shard(shard_path: str) -> dict[str, int]:
    """Count words in a single shard (for parallel execution)."""
    import re
    from collections import Counter

    counts: Counter[str] = Counter()
    with open(shard_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            words = re.findall(r"\b[a-zA-Z]+\b", line.lower())
            counts.update(words)
    return dict(counts)


def build_wordlist_from_corpus(corpus_dir: Path, output_path: Path, top_n: int = 100_000):
    """Build a frequency-sorted word list from a sample of the cleaned corpus."""
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("Building word list from corpus (sampled + parallel)...")
    shards = sorted(corpus_dir.glob("clean_*.txt"))

    # Sample every 10th shard — still millions of sentences, plenty for 100K words
    sample_step = max(1, len(shards) // 250)
    sampled = shards[::sample_step]
    print(f"  Sampling {len(sampled)}/{len(shards)} shards")

    word_counts: Counter[str] = Counter()
    num_workers = min(8, len(sampled))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_count_shard, str(s)): s for s in sampled}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Word count"):
            shard_counts = future.result()
            word_counts.update(shard_counts)

    # Filter: only words 2-30 chars, appearing at least 3 times in sample
    filtered = [
        (word, count)
        for word, count in word_counts.most_common()
        if 2 <= len(word) <= 30 and count >= 3
    ][:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(word for word, _ in filtered), encoding="utf-8"
    )
    print(f"Word list: {len(filtered)} words written to {output_path}")
    return [word for word, _ in filtered]


def main():
    parser = argparse.ArgumentParser(description="Build IPA database via espeak-ng")
    parser.add_argument("--wordlist", type=Path, default=DEFAULT_WORDLIST)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("phonetics/ipa_db.json"))
    parser.add_argument("--top-n", type=int, default=100_000)
    args = parser.parse_args()

    # Check espeak-ng is installed
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: espeak-ng not found. Install with: apt install espeak-ng")
        sys.exit(1)

    # Get or build word list
    if args.wordlist.exists():
        words = args.wordlist.read_text().strip().split("\n")
        words = [w.strip() for w in words if w.strip()]
        print(f"Loaded {len(words)} words from {args.wordlist}")
    else:
        words = build_wordlist_from_corpus(args.corpus_dir, args.wordlist, args.top_n)

    # Build IPA database
    print(f"Getting IPA for {len(words)} words via espeak-ng...")
    ipa_db = {}
    batch_size = 200

    for i in tqdm(range(0, len(words), batch_size), desc="IPA lookup"):
        batch = words[i : i + batch_size]
        batch_result = get_ipa_batch(batch, batch_size=batch_size)
        ipa_db.update(batch_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ipa_db, indent=1), encoding="utf-8")
    print(f"\nIPA database: {len(ipa_db)} entries written to {args.output}")
    print(f"Coverage: {len(ipa_db)}/{len(words)} ({100*len(ipa_db)/len(words):.1f}%)")


if __name__ == "__main__":
    main()
