#!/usr/bin/env python3
"""Build edit vocabulary for GECToR-style sequence tagging.

Generates millions of (corrupted, clean) pairs using the corruption engine,
aligns tokens, and extracts the most common edit operations.

Edit types:
  $KEEP           - don't change this token
  $DELETE         - remove this token
  $REPLACE_{tok}  - replace with specific token
  $APPEND_{tok}   - insert token after this position
  $MERGE          - merge with next token (fix split words)
  $CASE_LOWER     - lowercase this token
  $CASE_UPPER     - uppercase this token
  $CASE_TITLE     - title-case this token

Usage:
    PYTHONPATH=. python training/build_edit_vocab.py \
        --data-dir data/processed \
        --output checkpoints/tagger/edit_vocab.json \
        --num-sentences 500000
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from corruption.engine import CorruptionEngine


def _similar(a: str, b: str) -> bool:
    """Check if two words are similar enough to be a REPLACE pair.
    Uses character-level edit distance ratio.
    """
    if a.lower() == b.lower():
        return True
    la, lb = a.lower(), b.lower()
    if abs(len(la) - len(lb)) > max(len(la), len(lb)) // 2:
        return False
    # Quick Levenshtein
    if len(la) > len(lb):
        la, lb = lb, la
    prev = list(range(len(la) + 1))
    for j in range(1, len(lb) + 1):
        curr = [j] + [0] * len(la)
        for i in range(1, len(la) + 1):
            curr[i] = min(prev[i] + 1, curr[i-1] + 1,
                         prev[i-1] + (0 if la[i-1] == lb[j-1] else 1))
        prev = curr
    dist = prev[len(la)]
    max_len = max(len(la), len(lb))
    return dist <= max(1, max_len // 3)  # within ~33% edit distance


def align_and_extract_edits(corrupted: str, clean: str) -> list[tuple[str, str]]:
    """Align corrupted and clean words, extract edit operations.

    Uses similarity matching so character-level corruptions (quikc->quick)
    are captured as REPLACE operations, not DELETE+APPEND.

    Returns list of (corrupted_word, edit_tag) pairs.
    """
    c_words = corrupted.split()
    t_words = clean.split()

    if not c_words or not t_words:
        return []

    # DP alignment with similarity matching
    m, n = len(c_words), len(t_words)
    # Cost matrix: 0=match, 1=substitute(similar), 2=substitute(different), INF=gap
    INF = 999
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if c_words[i-1].lower() == t_words[j-1].lower():
                cost = 0
            elif _similar(c_words[i-1], t_words[j-1]):
                cost = 0.5  # similar words prefer alignment over gaps
            else:
                cost = 1.5
            dp[i][j] = min(
                dp[i-1][j-1] + cost,  # align/replace
                dp[i-1][j] + 1,       # delete from corrupted
                dp[i][j-1] + 1,       # insert from clean
            )

    # Backtrack
    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cw, tw = c_words[i-1], t_words[j-1]
            if cw.lower() == tw.lower():
                cost = 0
            elif _similar(cw, tw):
                cost = 0.5
            else:
                cost = 1.5

            if dp[i][j] == dp[i-1][j-1] + cost:
                # Aligned pair
                if cw == tw:
                    aligned.append((cw, "$KEEP"))
                elif cw.lower() == tw.lower():
                    # Case change
                    if tw.islower():
                        aligned.append((cw, "$CASE_LOWER"))
                    elif tw.isupper():
                        aligned.append((cw, "$CASE_UPPER"))
                    elif tw[0].isupper() and len(tw) > 1 and tw[1:].islower():
                        aligned.append((cw, "$CASE_TITLE"))
                    else:
                        aligned.append((cw, f"$REPLACE_{tw}"))
                else:
                    aligned.append((cw, f"$REPLACE_{tw}"))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            aligned.append((c_words[i-1], "$DELETE"))
            i -= 1
        elif j > 0:
            # Insert — attach as APPEND to previous aligned word
            if aligned:
                prev_word, prev_tag = aligned[-1]
                aligned[-1] = (prev_word, f"$APPEND_{t_words[j-1]}")
            else:
                aligned.append(("", f"$APPEND_{t_words[j-1]}"))
            j -= 1
        else:
            break

    aligned.reverse()
    return aligned


def build_vocab(data_dir: Path, num_sentences: int, seed: int = 42) -> dict:
    """Generate corruption pairs and extract edit vocabulary."""
    import random

    engine = CorruptionEngine(
        confusion_db_path=None, homophone_db_path=None, seed=seed
    )
    rng = random.Random(seed)

    edit_counts = Counter()
    total_tokens = 0
    total_edits = 0

    shards = sorted(data_dir.glob("clean_*.txt"))
    sentences_processed = 0

    for shard in shards:
        if sentences_processed >= num_sentences:
            break
        with open(shard) as f:
            for line in f:
                if sentences_processed >= num_sentences:
                    break
                line = line.strip()
                if not line or len(line.split()) < 4:
                    continue

                # Generate corrupted version
                rate = rng.uniform(0.05, 0.30)
                corrupted = engine.corrupt_sentence(line, rate)

                if corrupted == line:
                    # Identity — all KEEP
                    for w in line.split():
                        edit_counts["$KEEP"] += 1
                        total_tokens += 1
                else:
                    edits = align_and_extract_edits(corrupted, line)
                    for _, tag in edits:
                        edit_counts[tag] += 1
                        total_tokens += 1
                        if tag != "$KEEP":
                            total_edits += 1

                sentences_processed += 1
                if sentences_processed % 50000 == 0:
                    print(f"  {sentences_processed:,} sentences, {len(edit_counts):,} unique edits")

    print(f"\nTotal: {sentences_processed:,} sentences, {total_tokens:,} tokens, {total_edits:,} edits")
    print(f"Unique edit tags: {len(edit_counts):,}")
    print(f"KEEP rate: {edit_counts['$KEEP'] / total_tokens * 100:.1f}%")

    # Build vocabulary: special tags + top N edits
    special_tags = ["$KEEP", "$DELETE", "$CASE_LOWER", "$CASE_UPPER", "$CASE_TITLE", "$MERGE"]
    replace_tags = sorted(
        [(tag, count) for tag, count in edit_counts.items()
         if tag.startswith("$REPLACE_") and count >= 3],
        key=lambda x: -x[1]
    )
    append_tags = sorted(
        [(tag, count) for tag, count in edit_counts.items()
         if tag.startswith("$APPEND_") and count >= 3],
        key=lambda x: -x[1]
    )

    # Cap at reasonable vocab size
    max_replace = 3000
    max_append = 1000

    vocab = special_tags.copy()
    vocab += [tag for tag, _ in replace_tags[:max_replace]]
    vocab += [tag for tag, _ in append_tags[:max_append]]

    tag2id = {tag: i for i, tag in enumerate(vocab)}

    print(f"\nFinal vocab: {len(vocab)} tags")
    print(f"  Special: {len(special_tags)}")
    print(f"  Replace: {min(len(replace_tags), max_replace)}")
    print(f"  Append:  {min(len(append_tags), max_append)}")

    # Top 20 edits
    print("\nTop 20 non-KEEP edits:")
    for tag, count in edit_counts.most_common(25):
        if tag != "$KEEP":
            print(f"  {tag:30s} {count:>8,}")

    return {
        "vocab": vocab,
        "tag2id": tag2id,
        "stats": {
            "sentences": sentences_processed,
            "total_tokens": total_tokens,
            "total_edits": total_edits,
            "unique_edits": len(edit_counts),
            "keep_rate": edit_counts["$KEEP"] / total_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build edit vocabulary")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tagger/edit_vocab.json"))
    parser.add_argument("--num-sentences", type=int, default=500000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Building edit vocabulary...")
    result = build_vocab(args.data_dir, args.num_sentences)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
