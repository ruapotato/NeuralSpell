#!/usr/bin/env python3
"""Analyze error patterns in C4_200M to improve our corruption engine.

Reads C4_200M sentence pairs, aligns words, categorizes each error,
and produces a detailed report of error type frequencies.

The goal: reverse-engineer the error patterns so our corruption engine
can generate equivalent training data from scratch.

Usage:
    PYTHONPATH=. python tools/analyze_c4_errors.py \
        --input data/c4_200m/sentence_pairs.tsv \
        --max-pairs 100000 \
        --output docs/c4_error_analysis.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def categorize_edit(corrupted_word: str, clean_word: str) -> dict:
    """Categorize a single word-level edit into error types.

    Returns dict with category, subcategory, and details.
    """
    cw = corrupted_word
    tw = clean_word
    cl = cw.lower()
    tl = tw.lower()

    # Identical
    if cw == tw:
        return {"category": "KEEP", "subcategory": "identical"}

    # Case-only change
    if cl == tl:
        if tw.islower():
            return {"category": "CASE", "subcategory": "to_lower", "from": cw, "to": tw}
        elif tw.isupper():
            return {"category": "CASE", "subcategory": "to_upper", "from": cw, "to": tw}
        elif tw[0].isupper():
            return {"category": "CASE", "subcategory": "to_title", "from": cw, "to": tw}
        return {"category": "CASE", "subcategory": "other", "from": cw, "to": tw}

    # Character-level analysis
    len_diff = len(cl) - len(tl)

    # Single character deletion (target is longer)
    if len_diff == -1:
        for i in range(len(tl)):
            if cl == tl[:i] + tl[i+1:]:
                return {"category": "SPELLING", "subcategory": "char_insertion_needed",
                        "detail": f"missing '{tl[i]}' at pos {i}", "from": cw, "to": tw}

    # Single character insertion (corrupted is longer)
    if len_diff == 1:
        for i in range(len(cl)):
            if tl == cl[:i] + cl[i+1:]:
                return {"category": "SPELLING", "subcategory": "char_deletion_needed",
                        "detail": f"extra '{cl[i]}' at pos {i}", "from": cw, "to": tw}

    # Transposition
    if len(cl) == len(tl) and len(cl) >= 2:
        for i in range(len(cl) - 1):
            swapped = list(cl)
            swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
            if "".join(swapped) == tl:
                return {"category": "SPELLING", "subcategory": "transposition",
                        "detail": f"swap pos {i},{i+1}", "from": cw, "to": tw}

    # Single substitution
    if len(cl) == len(tl):
        diffs = [(i, cl[i], tl[i]) for i in range(len(cl)) if cl[i] != tl[i]]
        if len(diffs) == 1:
            pos, cfrom, cto = diffs[0]
            return {"category": "SPELLING", "subcategory": "char_substitution",
                    "detail": f"'{cfrom}'->'{cto}' at pos {pos}", "from": cw, "to": tw}

    # Double consonant errors
    for ch in "bcdfgklmnprst":
        if ch*2 in tl and ch*2 not in cl and tl.replace(ch*2, ch, 1) == cl:
            return {"category": "SPELLING", "subcategory": "missing_double",
                    "detail": f"need '{ch}{ch}'", "from": cw, "to": tw}
        if ch*2 in cl and ch*2 not in tl and cl.replace(ch*2, ch, 1) == tl:
            return {"category": "SPELLING", "subcategory": "extra_double",
                    "detail": f"extra '{ch}{ch}'", "from": cw, "to": tw}

    # Homophone / real-word error (both are valid English words)
    # We can't check dictionary here, but common patterns:
    homophones = {
        ("their", "there"), ("there", "their"), ("they're", "their"), ("their", "they're"),
        ("your", "you're"), ("you're", "your"), ("its", "it's"), ("it's", "its"),
        ("to", "too"), ("too", "to"), ("then", "than"), ("than", "then"),
        ("affect", "effect"), ("effect", "affect"), ("accept", "except"),
        ("loose", "lose"), ("lose", "loose"), ("quiet", "quite"), ("quite", "quiet"),
        ("where", "were"), ("were", "where"), ("hear", "here"), ("here", "hear"),
    }
    if (cl, tl) in homophones:
        return {"category": "REAL_WORD", "subcategory": "homophone",
                "from": cw, "to": tw}

    # Suffix confusion
    suffix_pairs = [
        ("ence", "ance"), ("ance", "ence"), ("ent", "ant"), ("ant", "ent"),
        ("ible", "able"), ("able", "ible"), ("tion", "sion"), ("sion", "tion"),
        ("ous", "ious"), ("ious", "ous"),
    ]
    for s1, s2 in suffix_pairs:
        if cl.endswith(s1) and tl.endswith(s2) and cl[:-len(s1)] == tl[:-len(s2)]:
            return {"category": "SPELLING", "subcategory": "suffix_confusion",
                    "detail": f"-{s1} vs -{s2}", "from": cw, "to": tw}

    # Phonetic confusion (ph/f, ck/k, etc)
    phonetic_subs = [("ph", "f"), ("f", "ph"), ("ck", "k"), ("k", "ck"),
                     ("ght", "t"), ("t", "ght"), ("tion", "shun"), ("shun", "tion")]
    for p1, p2 in phonetic_subs:
        if p1 in cl and cl.replace(p1, p2, 1) == tl:
            return {"category": "SPELLING", "subcategory": "phonetic",
                    "detail": f"{p1}->{p2}", "from": cw, "to": tw}

    # Vowel confusion
    vowels = "aeiou"
    if len(cl) == len(tl):
        diffs = [(i, cl[i], tl[i]) for i in range(len(cl)) if cl[i] != tl[i]]
        if all(c in vowels and t in vowels for _, c, t in diffs) and len(diffs) <= 2:
            return {"category": "SPELLING", "subcategory": "vowel_confusion",
                    "detail": str(diffs), "from": cw, "to": tw}

    # Grammar: verb form changes
    if (cl.endswith("ed") and not tl.endswith("ed")) or \
       (not cl.endswith("ed") and tl.endswith("ed")):
        if cl.rstrip("ed")[:3] == tl.rstrip("ed")[:3]:
            return {"category": "GRAMMAR", "subcategory": "verb_tense",
                    "from": cw, "to": tw}

    if (cl.endswith("ing") and not tl.endswith("ing")) or \
       (not cl.endswith("ing") and tl.endswith("ing")):
        return {"category": "GRAMMAR", "subcategory": "verb_form",
                "from": cw, "to": tw}

    # Plural/singular
    if cl + "s" == tl or cl + "es" == tl:
        return {"category": "GRAMMAR", "subcategory": "missing_plural",
                "from": cw, "to": tw}
    if tl + "s" == cl or tl + "es" == cl:
        return {"category": "GRAMMAR", "subcategory": "extra_plural",
                "from": cw, "to": tw}

    # Multi-edit (Levenshtein > 2)
    # Quick Levenshtein
    m, n = len(cl), len(tl)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1,
                         prev[j-1] + (0 if cl[i-1] == tl[j-1] else 1))
        prev = curr
    edit_dist = prev[n]

    if edit_dist <= 2:
        return {"category": "SPELLING", "subcategory": "multi_char",
                "detail": f"edit_dist={edit_dist}", "from": cw, "to": tw}

    # Completely different word (real-word substitution or major rewrite)
    return {"category": "WORD_CHOICE", "subcategory": "different_word",
            "detail": f"edit_dist={edit_dist}", "from": cw, "to": tw}


def align_words(corrupted: str, clean: str) -> list[tuple[str, str, str]]:
    """Align corrupted and clean sentences at word level.

    Returns list of (corrupted_word, clean_word, operation) triples.
    Operation is: KEEP, REPLACE, DELETE, INSERT.
    """
    cw = corrupted.split()
    tw = clean.split()

    if not cw or not tw:
        return []

    # DP alignment
    m, n = len(cw), len(tw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cw[i-1].lower() == tw[j-1].lower():
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)

    # Backtrack
    result = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if cw[i-1].lower() == tw[j-1].lower() else 1):
            if cw[i-1] == tw[j-1]:
                result.append((cw[i-1], tw[j-1], "KEEP"))
            else:
                result.append((cw[i-1], tw[j-1], "REPLACE"))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            result.append((cw[i-1], "", "DELETE"))
            i -= 1
        elif j > 0:
            result.append(("", tw[j-1], "INSERT"))
            j -= 1
        else:
            break

    result.reverse()
    return result


def analyze_file(input_path: Path, max_pairs: int) -> dict:
    """Analyze error patterns in a TSV file of sentence pairs."""
    category_counts = Counter()
    subcategory_counts = Counter()
    edit_examples = defaultdict(list)
    total_pairs = 0
    total_words = 0
    total_edits = 0
    operation_counts = Counter()

    with open(input_path) as f:
        for line_num, line in enumerate(f):
            if total_pairs >= max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            corrupted, clean = parts
            if corrupted == clean:
                continue

            alignment = align_words(corrupted, clean)
            total_pairs += 1

            for cword, tword, operation in alignment:
                total_words += 1
                operation_counts[operation] += 1

                if operation == "KEEP":
                    category_counts["KEEP"] += 1
                    continue

                if operation == "DELETE":
                    category_counts["DELETE"] += 1
                    subcategory_counts["DELETE.word_deletion"] += 1
                    total_edits += 1
                    if len(edit_examples["DELETE"]) < 5:
                        edit_examples["DELETE"].append({"word": cword})
                    continue

                if operation == "INSERT":
                    category_counts["INSERT"] += 1
                    subcategory_counts[f"INSERT.word_{tword.lower()}"] += 1
                    total_edits += 1
                    if len(edit_examples["INSERT"]) < 5:
                        edit_examples["INSERT"].append({"word": tword})
                    continue

                # REPLACE — categorize the error
                info = categorize_edit(cword, tword)
                cat = info["category"]
                subcat = f"{cat}.{info['subcategory']}"
                category_counts[cat] += 1
                subcategory_counts[subcat] += 1
                total_edits += 1

                if len(edit_examples.get(subcat, [])) < 3:
                    if subcat not in edit_examples:
                        edit_examples[subcat] = []
                    edit_examples[subcat].append({"from": cword, "to": tword})

            if (total_pairs) % 10000 == 0:
                print(f"  {total_pairs:,} pairs, {total_edits:,} edits...", flush=True)

    return {
        "total_pairs": total_pairs,
        "total_words": total_words,
        "total_edits": total_edits,
        "keep_rate": (total_words - total_edits) / total_words if total_words > 0 else 0,
        "category_counts": dict(category_counts.most_common()),
        "subcategory_counts": dict(subcategory_counts.most_common(100)),
        "operation_counts": dict(operation_counts.most_common()),
        "examples": {k: v for k, v in edit_examples.items()},
    }


def print_report(analysis: dict):
    """Print human-readable analysis report."""
    print(f"\n{'=' * 70}")
    print(f"C4_200M Error Pattern Analysis")
    print(f"{'=' * 70}")
    print(f"Total pairs analyzed: {analysis['total_pairs']:,}")
    print(f"Total words: {analysis['total_words']:,}")
    print(f"Total edits: {analysis['total_edits']:,}")
    print(f"KEEP rate: {analysis['keep_rate']*100:.1f}%")

    print(f"\n--- Top-Level Categories ---")
    for cat, count in sorted(analysis['category_counts'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_edits'] * 100 if analysis['total_edits'] > 0 else 0
        print(f"  {cat:25s} {count:>8,}  ({pct:5.1f}%)")

    print(f"\n--- Subcategories (top 40) ---")
    for subcat, count in list(analysis['subcategory_counts'].items())[:40]:
        pct = count / analysis['total_edits'] * 100 if analysis['total_edits'] > 0 else 0
        examples = analysis['examples'].get(subcat, [])
        ex_str = ""
        if examples:
            ex = examples[0]
            if "from" in ex:
                ex_str = f"  e.g. {ex['from']} -> {ex['to']}"
            elif "word" in ex:
                ex_str = f"  e.g. {ex['word']}"
        print(f"  {subcat:40s} {count:>8,}  ({pct:5.1f}%){ex_str}")


def main():
    parser = argparse.ArgumentParser(description="Analyze C4_200M error patterns")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Analyzing {args.input}...")
    analysis = analyze_file(args.input, args.max_pairs)
    print_report(analysis)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
