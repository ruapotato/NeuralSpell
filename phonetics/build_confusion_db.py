"""Build phonetic confusion database from IPA transcriptions.

For each word, find other words with IPA edit distance <= 1.
These are phonetically confusable words — the kind of errors
humans make when spelling by sound.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def ipa_edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance on IPA phoneme strings.

    Operates on characters since espeak IPA output is already
    segmented enough for our purposes.
    """
    if abs(len(a) - len(b)) > 2:
        return 999  # early exit for very different lengths
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    # Optimized single-row DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[n]


def build_confusion_db(
    ipa_db: dict[str, str], max_distance: int = 1
) -> dict[str, list[str]]:
    """Find phonetically confusable words.

    For efficiency, first group by IPA length (words with very different
    IPA lengths can't be within edit distance 1). Then compare within
    nearby length buckets.
    """
    # Group words by IPA string length for efficient comparison
    by_length: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for word, ipa in ipa_db.items():
        by_length[len(ipa)].append((word, ipa))

    confusions: dict[str, list[str]] = {}
    lengths = sorted(by_length.keys())

    for length in tqdm(lengths, desc="Building confusion sets"):
        # Compare words of this length with words of length-1, length, length+1
        candidates = []
        for dl in range(max_distance + 1):
            if (length + dl) in by_length:
                candidates.extend(by_length[length + dl])
            if dl > 0 and (length - dl) in by_length:
                candidates.extend(by_length[length - dl])

        for word, ipa in by_length[length]:
            similar = []
            for other_word, other_ipa in candidates:
                if other_word == word:
                    continue
                if ipa_edit_distance(ipa, other_ipa) <= max_distance:
                    similar.append(other_word)
            if similar:
                confusions[word] = similar

    return confusions


def main():
    parser = argparse.ArgumentParser(description="Build phonetic confusion database")
    parser.add_argument(
        "--ipa-db", type=Path, default=Path("phonetics/ipa_db.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("phonetics/confusion_db.json")
    )
    parser.add_argument("--max-distance", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading IPA database from {args.ipa_db}...")
    ipa_db = json.loads(args.ipa_db.read_text())
    print(f"  {len(ipa_db)} entries")

    print(f"Finding confusable pairs (edit distance <= {args.max_distance})...")
    confusions = build_confusion_db(ipa_db, args.max_distance)

    args.output.write_text(json.dumps(confusions, indent=1), encoding="utf-8")
    print(f"\nConfusion database: {len(confusions)} words with confusables")
    print(f"Written to {args.output}")

    # Print some examples
    print("\nExamples:")
    count = 0
    for word, similar in confusions.items():
        if len(similar) >= 2:
            print(f"  {word} -> {similar[:5]}")
            count += 1
            if count >= 10:
                break


if __name__ == "__main__":
    main()
