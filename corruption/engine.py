"""Corruption engine: generates realistic human spelling errors.

Takes clean sentences and produces (corrupted, original) pairs for
training the spell corrector. Implements 11 corruption types with
configurable weights matching real-world error frequency.
"""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from corruption.keyboard import fat_finger_word
from corruption.phonetic import PhoneticCorruptor, phonetic_rewrite
from corruption.homophones import HomophoneCorruptor
from corruption.grammar import corrupt_grammar


@dataclass
class CorruptionWeights:
    """Relative weights for each corruption type.

    Weights are normalized to probabilities at runtime.
    Higher weight = more likely to be selected.
    """
    keyboard_adjacency: float = 20.0
    transposition: float = 15.0
    insertion: float = 10.0
    deletion: float = 10.0
    phonetic_word: float = 5.0    # whole-word phonetic swap (from DB)
    phonetic_rewrite: float = 10.0 # character-level phonetic rewrite
    homophone: float = 10.0
    capitalization: float = 5.0
    double_letter: float = 8.0
    suffix_confusion: float = 7.0
    missing_space: float = 10.0   # join adjacent words


class CorruptionEngine:
    """Generate realistic human spelling errors from clean text."""

    def __init__(
        self,
        confusion_db_path: Path | None = Path("phonetics/confusion_db.json"),
        homophone_db_path: Path | None = Path("phonetics/homophone_sets.json"),
        weights: CorruptionWeights | None = None,
        seed: int | None = None,
    ):
        self.weights = weights or CorruptionWeights()
        self.rng = random.Random(seed)
        self.phonetic = PhoneticCorruptor(confusion_db_path)
        self.homophones = HomophoneCorruptor(homophone_db_path)

        # Build normalized weight distribution for word-level corruptions
        self._word_types = [
            ("keyboard_adjacency", self.weights.keyboard_adjacency),
            ("transposition", self.weights.transposition),
            ("insertion", self.weights.insertion),
            ("deletion", self.weights.deletion),
            ("phonetic_word", self.weights.phonetic_word),
            ("phonetic_rewrite", self.weights.phonetic_rewrite),
            ("homophone", self.weights.homophone),
            ("capitalization", self.weights.capitalization),
            ("double_letter", self.weights.double_letter),
            ("suffix_confusion", self.weights.suffix_confusion),
        ]
        total = sum(w for _, w in self._word_types)
        self._word_probs = [(t, w / total) for t, w in self._word_types]

    def _pick_word_corruption(self) -> str:
        """Pick a corruption type based on weights."""
        r = self.rng.random()
        cumulative = 0.0
        for ctype, prob in self._word_probs:
            cumulative += prob
            if r <= cumulative:
                return ctype
        return self._word_probs[-1][0]

    def corrupt_word(self, word: str) -> str:
        """Apply a random corruption to a single word."""
        if len(word) < 3:
            return word

        ctype = self._pick_word_corruption()
        result = self._apply_word_corruption(word, ctype)
        # If the chosen type didn't produce a change, try others
        if result == word:
            for backup_type, _ in self._word_types:
                if backup_type != ctype:
                    result = self._apply_word_corruption(word, backup_type)
                    if result != word:
                        break
        return result

    def _apply_word_corruption(self, word: str, ctype: str) -> str:
        """Apply a specific corruption type to a word."""
        if ctype == "keyboard_adjacency":
            return fat_finger_word(word, self.rng)

        elif ctype == "transposition":
            return self._transpose(word)

        elif ctype == "insertion":
            return self._insert_char(word)

        elif ctype == "deletion":
            return self._delete_char(word)

        elif ctype == "phonetic_word":
            result = self.phonetic.corrupt(word, self.rng)
            return result if result else word

        elif ctype == "phonetic_rewrite":
            result = phonetic_rewrite(word, self.rng)
            return result if result else word

        elif ctype == "homophone":
            result = self.homophones.corrupt(word, self.rng)
            return result if result else word

        elif ctype == "capitalization":
            return self._corrupt_case(word)

        elif ctype == "double_letter":
            return self._double_letter(word)

        elif ctype == "suffix_confusion":
            result = phonetic_rewrite(word, self.rng)  # suffix handling is in phonetic_rewrite
            return result if result else word

        return word

    def _transpose(self, word: str) -> str:
        """Swap two adjacent characters."""
        if len(word) < 3:
            return word
        pos = self.rng.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)

    def _insert_char(self, word: str) -> str:
        """Insert a random adjacent-key character."""
        from corruption.keyboard import ADJACENCY, get_adjacent_key

        pos = self.rng.randint(0, len(word) - 1)
        char = word[pos]
        adj = get_adjacent_key(char)
        if adj:
            return word[:pos] + adj + word[pos:]
        # Fallback: double the character
        return word[:pos] + char + word[pos:]

    def _delete_char(self, word: str) -> str:
        """Delete a random character (not first or last)."""
        if len(word) <= 3:
            return word
        pos = self.rng.randint(1, len(word) - 2)
        return word[:pos] + word[pos + 1 :]

    def _corrupt_case(self, word: str) -> str:
        """Apply a capitalization error."""
        choice = self.rng.randint(0, 3)
        if choice == 0:
            return word.lower()
        elif choice == 1:
            return word.upper()
        elif choice == 2:
            return word.capitalize() if word[0].islower() else word.lower()
        else:
            # MiXeD CaSe
            return "".join(
                c.upper() if self.rng.random() > 0.5 else c.lower() for c in word
            )

    def _double_letter(self, word: str) -> str:
        """Add or remove a doubled letter."""
        if len(word) < 3:
            return word

        # Find existing doubles and remove one
        for i in range(len(word) - 1):
            if word[i] == word[i + 1] and word[i].isalpha():
                if self.rng.random() < 0.5:
                    return word[:i] + word[i + 1 :]

        # Or double a random consonant
        consonants = [
            i
            for i, c in enumerate(word)
            if c.lower() in "bcdfghjklmnpqrstvwxyz"
        ]
        if consonants:
            pos = self.rng.choice(consonants)
            return word[:pos] + word[pos] + word[pos:]

        return word

    def corrupt_sentence(
        self, sentence: str, word_corruption_rate: float = 0.15
    ) -> str:
        """Corrupt a sentence, returning the corrupted version.

        Each word independently has word_corruption_rate chance of corruption.
        May also apply sentence-level corruptions (grammar, missing spaces).
        """
        # Tokenize preserving whitespace and punctuation
        tokens = re.findall(r"\S+|\s+", sentence)
        result_tokens = []
        words_corrupted = 0

        for token in tokens:
            if token.isspace():
                result_tokens.append(token)
                continue

            # Separate word from trailing punctuation
            match = re.match(r"^([A-Za-z'-]+)(.*)$", token)
            if match and len(match.group(1)) >= 3:
                word, rest = match.group(1), match.group(2)
                if self.rng.random() < word_corruption_rate:
                    word = self.corrupt_word(word)
                    words_corrupted += 1
                result_tokens.append(word + rest)
            else:
                result_tokens.append(token)

        result = "".join(result_tokens)

        # Sentence-level corruptions

        # Grammar corruption (independent chance)
        if self.rng.random() < 0.10:
            grammar_result = corrupt_grammar(result, self.rng)
            if grammar_result:
                result = grammar_result

        # Missing space corruption: join two adjacent words
        if self.rng.random() < (self.weights.missing_space / 100.0):
            result = self._remove_space(result)

        return result

    def _remove_space(self, sentence: str) -> str:
        """Remove a random space between two words to create a joined-words error."""
        # Find spaces between words (not at boundaries)
        space_positions = []
        tokens = sentence.split(" ")
        if len(tokens) < 3:
            return sentence

        # Pick a space between two alphabetic words
        for i in range(1, len(tokens) - 1):
            left = tokens[i - 1] if tokens[i - 1] else ""
            curr = tokens[i] if tokens[i] else ""
            # Both sides should be word-like
            if (
                left
                and curr
                and left[-1].isalpha()
                and curr[0].isalpha()
                and len(left) >= 2
                and len(curr) >= 2
            ):
                space_positions.append(i)

        if not space_positions:
            return sentence

        join_at = self.rng.choice(space_positions)
        # Join tokens[join_at-1] and tokens[join_at]
        tokens[join_at - 1] = tokens[join_at - 1] + tokens[join_at]
        del tokens[join_at]
        return " ".join(tokens)

    def generate_pair(
        self, clean_sentence: str, corruption_rate: float = 0.15
    ) -> tuple[str, str]:
        """Generate a (corrupted, original) training pair."""
        corrupted = self.corrupt_sentence(clean_sentence, corruption_rate)
        return corrupted, clean_sentence

    def build_dataset(
        self,
        clean_sentences: list[str],
        variants_per_sentence: int = 3,
        min_corruption_rate: float = 0.05,
        max_corruption_rate: float = 0.30,
    ) -> list[tuple[str, str]]:
        """Generate corruption pairs from clean sentences.

        Each clean sentence generates `variants_per_sentence` corrupted
        versions with varying corruption rates.
        """
        pairs = []
        for sentence in clean_sentences:
            for _ in range(variants_per_sentence):
                rate = self.rng.uniform(min_corruption_rate, max_corruption_rate)
                corrupted, original = self.generate_pair(sentence, rate)
                if corrupted != original:  # only keep if actually corrupted
                    pairs.append((corrupted, original))
        return pairs
