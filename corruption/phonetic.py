"""Phonetic substitution corruption using on-the-fly espeak-ng lookups.

Instead of prebuilding a massive confusion DB, we:
1. Look up IPA for the target word via espeak-ng (cached)
2. Check a small local cache of known confusable words
3. Fall back to character-level phonetic rewrites
"""

import functools
import random
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=50000)
def get_ipa(word: str) -> str | None:
    """Get IPA transcription via espeak-ng, cached."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-q", "--", word],
            capture_output=True,
            text=True,
            timeout=2,
        )
        ipa = result.stdout.strip()
        if ipa:
            return ipa.split("\n")[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


class PhoneticCorruptor:
    """Substitutes words with phonetically similar alternatives.

    Uses the homophone_sets.json if available (exact IPA matches),
    otherwise falls back to character-level phonetic rewrites.
    """

    def __init__(self, confusion_db_path: Path | None = None):
        # We no longer require a prebuilt confusion DB.
        # The engine's homophone corruptor handles exact matches,
        # and phonetic_rewrite handles character-level swaps.
        self.confusion_db: dict[str, list[str]] = {}
        if confusion_db_path is not None and confusion_db_path.exists():
            import json
            self.confusion_db = json.loads(confusion_db_path.read_text())

    def corrupt(self, word: str, rng: random.Random | None = None) -> str | None:
        """Replace word with a phonetically similar alternative.

        Returns None if no phonetic substitute is available.
        """
        rng = rng or random.Random()
        lower = word.lower()

        # Try prebuilt DB first
        candidates = self.confusion_db.get(lower)
        if candidates:
            replacement = rng.choice(candidates)
            if word.isupper():
                return replacement.upper()
            if word[0].isupper():
                return replacement.capitalize()
            return replacement

        # Fall back to character-level phonetic rewrite
        return phonetic_rewrite(word, rng)


# Common phonetic misspelling patterns that don't need espeak.
# These are applied at the character level within words.
PHONETIC_REWRITES = [
    # /f/ <-> ph
    ("ph", "f"),
    ("f", "ph"),
    # /k/ <-> ck, c
    ("ck", "k"),
    ("k", "ck"),
    # Silent letters dropped
    ("kn", "n"),
    ("wr", "r"),
    ("gn", "n"),
    ("mb", "m"),
    # Double/single confusion
    ("ss", "s"),
    ("ll", "l"),
    ("tt", "t"),
    ("ff", "f"),
    ("rr", "r"),
    ("nn", "n"),
    ("mm", "m"),
    ("pp", "p"),
    ("dd", "d"),
    ("gg", "g"),
    ("cc", "c"),
    # Vowel confusions
    ("ei", "ie"),
    ("ie", "ei"),
    ("ea", "ee"),
    ("ee", "ea"),
    # -tion/-sion
    ("tion", "sion"),
    ("sion", "tion"),
    # -ible/-able
    ("ible", "able"),
    ("able", "ible"),
    # -ent/-ant
    ("ent", "ant"),
    ("ant", "ent"),
    # -ence/-ance
    ("ence", "ance"),
    ("ance", "ence"),
]


def phonetic_rewrite(word: str, rng: random.Random | None = None) -> str | None:
    """Apply a character-level phonetic rewrite to a word.

    Returns None if no rewrite is applicable.
    """
    rng = rng or random.Random()
    lower = word.lower()

    applicable = []
    for old, new in PHONETIC_REWRITES:
        idx = lower.find(old)
        if idx != -1:
            applicable.append((idx, old, new))

    if not applicable:
        return None

    idx, old, new = rng.choice(applicable)
    # Apply the rewrite preserving case of first char
    result = lower[:idx] + new + lower[idx + len(old) :]

    if word[0].isupper():
        result = result[0].upper() + result[1:]
    if word.isupper():
        result = result.upper()

    return result
