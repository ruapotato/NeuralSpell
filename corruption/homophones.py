"""Homophone swap corruption using the homophone database."""

import json
import random
from pathlib import Path


class HomophoneCorruptor:
    """Swaps words with their homophones — errors only context can fix."""

    def __init__(
        self,
        homophone_db_path: Path = Path("phonetics/homophone_sets.json"),
    ):
        if homophone_db_path is not None and homophone_db_path.exists():
            data = json.loads(homophone_db_path.read_text())
            self.lookup: dict[str, list[str]] = data.get("lookup", {})
        else:
            # Fallback: hard-coded high-frequency homophones
            self.lookup = {}
        # Always include these high-value pairs even if DB isn't built yet
        self._add_known_homophones()

    def _add_known_homophones(self):
        """Ensure high-frequency homophone pairs are always present."""
        known = [
            ["their", "there", "they're"],
            ["to", "too", "two"],
            ["your", "you're"],
            ["its", "it's"],
            ["then", "than"],
            ["affect", "effect"],
            ["accept", "except"],
            ["principal", "principle"],
            ["stationary", "stationery"],
            ["complement", "compliment"],
            ["council", "counsel"],
            ["discreet", "discrete"],
            ["loose", "lose"],
            ["peace", "piece"],
            ["weather", "whether"],
            ["whose", "who's"],
            ["hear", "here"],
            ["know", "no"],
            ["knew", "new"],
            ["write", "right"],
            ["sight", "site", "cite"],
            ["bare", "bear"],
            ["brake", "break"],
            ["fair", "fare"],
            ["flour", "flower"],
            ["hair", "hare"],
            ["heal", "heel"],
            ["hole", "whole"],
            ["hour", "our"],
            ["mail", "male"],
            ["meat", "meet"],
            ["pair", "pear", "pare"],
            ["plain", "plane"],
            ["pole", "poll"],
            ["rain", "reign", "rein"],
            ["road", "rode"],
            ["role", "roll"],
            ["sail", "sale"],
            ["scene", "seen"],
            ["sea", "see"],
            ["sole", "soul"],
            ["some", "sum"],
            ["son", "sun"],
            ["stair", "stare"],
            ["steal", "steel"],
            ["tail", "tale"],
            ["through", "threw"],
            ["waist", "waste"],
            ["wait", "weight"],
            ["weak", "week"],
            ["wear", "where"],
            ["wood", "would"],
        ]
        for group in known:
            for word in group:
                others = [w for w in group if w != word]
                if word not in self.lookup:
                    self.lookup[word] = others
                else:
                    # Merge with existing
                    existing = set(self.lookup[word])
                    existing.update(others)
                    self.lookup[word] = list(existing)

    def corrupt(self, word: str, rng: random.Random | None = None) -> str | None:
        """Replace word with a homophone.

        Returns None if no homophone is available.
        """
        rng = rng or random.Random()
        lower = word.lower()
        candidates = self.lookup.get(lower)
        if not candidates:
            return None

        replacement = rng.choice(candidates)

        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement
