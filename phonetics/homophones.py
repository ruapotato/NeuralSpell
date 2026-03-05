"""Extract exact homophone sets from IPA database.

Homophones are words with identical IPA transcriptions.
These are the errors that context-free spell checkers cannot fix.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_homophone_sets(ipa_db: dict[str, str]) -> list[list[str]]:
    """Group words by exact IPA match."""
    by_ipa: dict[str, list[str]] = defaultdict(list)
    for word, ipa in ipa_db.items():
        # Normalize IPA: strip stress marks and whitespace for matching
        normalized = ipa.replace("ˈ", "").replace("ˌ", "").replace(" ", "")
        by_ipa[normalized].append(word)

    # Only keep groups with 2+ words
    sets = [sorted(words) for words in by_ipa.values() if len(words) >= 2]
    sets.sort(key=lambda s: (-len(s), s[0]))
    return sets


def build_homophone_lookup(sets: list[list[str]]) -> dict[str, list[str]]:
    """Build word -> [homophones] lookup for the corruption engine."""
    lookup: dict[str, list[str]] = {}
    for group in sets:
        for word in group:
            others = [w for w in group if w != word]
            if others:
                lookup[word] = others
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Extract homophone sets")
    parser.add_argument(
        "--ipa-db", type=Path, default=Path("phonetics/ipa_db.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("phonetics/homophone_sets.json")
    )
    args = parser.parse_args()

    ipa_db = json.loads(args.ipa_db.read_text())
    sets = build_homophone_sets(ipa_db)
    lookup = build_homophone_lookup(sets)

    output = {"sets": sets, "lookup": lookup}
    args.output.write_text(json.dumps(output, indent=1), encoding="utf-8")

    print(f"Found {len(sets)} homophone sets ({len(lookup)} words total)")
    print("\nLargest sets:")
    for group in sets[:15]:
        print(f"  {group}")


if __name__ == "__main__":
    main()
