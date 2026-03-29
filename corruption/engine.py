"""Corruption engine: generates realistic human writing errors.

Takes clean sentences and produces (corrupted, original) pairs for
training the spell corrector. Error types and weights are based on
real human error statistics from BEA-2019, NUCLE, FCE, Birkbeck,
and GCSE studies.

Error categories (research-based relative frequencies):
  - Non-word spelling (typos): ~35-40%
  - Homophones / confusables: ~10-15%
  - Determiners / articles: ~10-16%
  - Prepositions: ~7-11%
  - Verb tense: ~6-7%
  - Punctuation: ~5-10%
  - Noun number: ~3-8%
  - Word form: ~3-5%
  - Missing / extra words: ~5-8%
  - Agreement: ~2-3%
  - Missing / extra spaces: ~2-3%
  - Contractions: ~1-2%
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path

from corruption.keyboard import fat_finger_word
from corruption.phonetic import PhoneticCorruptor, phonetic_rewrite
from corruption.homophones import HomophoneCorruptor
from corruption.grammar import corrupt_grammar


@dataclass
class CorruptionWeights:
    """Relative weights for each corruption type.

    Weights are normalized to probabilities at runtime.
    Calibrated against human error research (BEA-2019, NUCLE, Birkbeck).

    Within non-word spelling, the research breakdown is:
      deletion ~40%, substitution ~30%, insertion ~20%, transposition ~10%
    """
    # Non-word spelling (typos): ~35-40% total
    keyboard_adjacency: float = 12.0     # fat-finger substitution
    deletion: float = 12.0               # missing character (most common typo)
    insertion: float = 6.0               # extra character
    transposition: float = 4.0           # swapped adjacent chars
    double_letter: float = 4.0           # add/remove doubled letter
    phonetic_rewrite: float = 5.0        # character-level phonetic (ph->f etc)
    suffix_confusion: float = 3.0        # -ible/-able, -ie/-ei

    # Real-word errors: ~10-15% total
    homophone: float = 8.0              # their/there/they're etc
    phonetic_word: float = 3.0          # whole-word phonetic swap

    # Grammar errors: ~25-35% total (dispatched to grammar.py)
    grammar: float = 25.0              # determiners, prepositions, tense, etc

    # Sentence-level: ~10-15% total
    missing_word: float = 6.0          # drop an article/preposition/aux verb
    extra_word: float = 3.0            # repeat a word or insert filler
    punctuation: float = 5.0           # comma, apostrophe, period errors
    missing_space: float = 3.0         # join adjacent words
    split_word: float = 2.0            # add space inside a word
    capitalization: float = 3.0        # case errors


# Common small words that get dropped in human writing
DROPPABLE_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "must", "shall",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between",
    "and", "but", "or", "so", "yet", "not", "that", "which", "who",
    "it", "he", "she", "they", "we", "you", "I",
}

# Compound words that people sometimes split
COMPOUND_WORDS = [
    "cannot", "sometimes", "everyone", "everything", "nothing",
    "something", "anything", "anyone", "already", "always",
    "myself", "yourself", "himself", "herself", "themselves",
    "ourselves", "itself", "another", "however", "whatever",
    "whenever", "wherever", "otherwise", "nevertheless",
    "furthermore", "therefore", "meanwhile", "throughout",
    "inside", "outside", "without", "within", "into",
    "today", "tonight", "tomorrow", "yesterday",
    "somewhere", "everywhere", "nowhere",
    "maybe", "also", "upon", "onto", "nearby",
]


class CorruptionEngine:
    """Generate realistic human writing errors from clean text."""

    def __init__(
        self,
        confusion_db_path: Path | None = Path("phonetics/confusion_db.json"),
        homophone_db_path: Path | None = Path("phonetics/homophone_sets.json"),
        weights: CorruptionWeights | None = None,
        seed: int | None = None,
    ):
        self.weights = weights or CorruptionWeights()
        self.seed = seed
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

        # Sentence-level corruption weights
        self._sentence_types = [
            ("grammar", self.weights.grammar),
            ("missing_word", self.weights.missing_word),
            ("extra_word", self.weights.extra_word),
            ("punctuation", self.weights.punctuation),
            ("missing_space", self.weights.missing_space),
            ("split_word", self.weights.split_word),
        ]

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
            result = phonetic_rewrite(word, self.rng)
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
        from corruption.keyboard import get_adjacent_key

        pos = self.rng.randint(0, len(word) - 1)
        char = word[pos]
        adj = get_adjacent_key(char)
        if adj:
            return word[:pos] + adj + word[pos:]
        return word[:pos] + char + word[pos:]

    def _delete_char(self, word: str) -> str:
        """Delete a random character (not first or last)."""
        if len(word) <= 3:
            return word
        pos = self.rng.randint(1, len(word) - 2)
        return word[:pos] + word[pos + 1:]

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
            return "".join(
                c.upper() if self.rng.random() > 0.5 else c.lower() for c in word
            )

    def _double_letter(self, word: str) -> str:
        """Add or remove a doubled letter."""
        if len(word) < 3:
            return word
        for i in range(len(word) - 1):
            if word[i] == word[i + 1] and word[i].isalpha():
                if self.rng.random() < 0.5:
                    return word[:i] + word[i + 1:]
        consonants = [
            i for i, c in enumerate(word)
            if c.lower() in "bcdfghjklmnpqrstvwxyz"
        ]
        if consonants:
            pos = self.rng.choice(consonants)
            return word[:pos] + word[pos] + word[pos:]
        return word

    # ─── Sentence-level corruptions ──────────────────────────────

    def _drop_word(self, sentence: str) -> str:
        """Drop a random small word (article, preposition, auxiliary)."""
        words = sentence.split()
        candidates = [i for i, w in enumerate(words)
                      if w.lower().rstrip(",.;:!?") in DROPPABLE_WORDS and i > 0]
        if not candidates:
            return sentence
        idx = self.rng.choice(candidates)
        words.pop(idx)
        return " ".join(words)

    def _repeat_word(self, sentence: str) -> str:
        """Repeat a random word ("I went went to the store")."""
        words = sentence.split()
        if len(words) < 4:
            return sentence
        idx = self.rng.randint(1, len(words) - 2)
        words.insert(idx, words[idx])
        return " ".join(words)

    def _corrupt_punctuation(self, sentence: str) -> str:
        """Corrupt punctuation: drop comma, add comma, swap period/comma."""
        choice = self.rng.randint(0, 3)

        if choice == 0:
            # Drop a comma
            comma_positions = [i for i, c in enumerate(sentence) if c == ","]
            if comma_positions:
                pos = self.rng.choice(comma_positions)
                # Remove comma and following space
                if pos + 1 < len(sentence) and sentence[pos + 1] == " ":
                    return sentence[:pos] + sentence[pos + 2:]
                return sentence[:pos] + sentence[pos + 1:]

        elif choice == 1:
            # Insert a random comma after a word
            words = sentence.split()
            if len(words) > 4:
                idx = self.rng.randint(2, len(words) - 3)
                if not words[idx].endswith((",", ".", "!", "?")):
                    words[idx] = words[idx] + ","
                    return " ".join(words)

        elif choice == 2:
            # Drop period at end
            if sentence.endswith("."):
                return sentence[:-1]

        else:
            # Drop apostrophe from possessives
            possessive = re.search(r"(\w+)'s\b", sentence)
            if possessive:
                return sentence[:possessive.start()] + possessive.group(1) + "s" + sentence[possessive.end():]

        return sentence

    def _split_compound(self, sentence: str) -> str:
        """Split a compound word with a space ("cannot" -> "can not")."""
        words = sentence.split()
        candidates = []
        for i, w in enumerate(words):
            clean = re.sub(r"[^\w]", "", w).lower()
            if clean in COMPOUND_WORDS and len(clean) > 5:
                candidates.append(i)

        if not candidates:
            # Fallback: split a long word at a random point
            for i, w in enumerate(words):
                clean = re.sub(r"[^\w]", "", w)
                if len(clean) > 7:
                    candidates.append(i)

            if not candidates:
                return sentence

            idx = self.rng.choice(candidates)
            word = words[idx]
            clean = re.sub(r"([^\w]+)$", "", word)
            suffix = word[len(clean):]
            split_pos = self.rng.randint(3, len(clean) - 3)
            words[idx] = clean[:split_pos] + " " + clean[split_pos:] + suffix
            return " ".join(words)

        idx = self.rng.choice(candidates)
        word = words[idx]
        clean = re.sub(r"([^\w]+)$", "", word).lower()
        suffix = word[len(re.sub(r"([^\w]+)$", "", word)):]

        # Known split points for compound words
        known_splits = {
            "cannot": "can not", "sometimes": "some times",
            "everyone": "every one", "everything": "every thing",
            "nothing": "no thing", "something": "some thing",
            "anything": "any thing", "anyone": "any one",
            "already": "all ready", "always": "all ways",
            "myself": "my self", "yourself": "your self",
            "himself": "him self", "herself": "her self",
            "themselves": "them selves", "ourselves": "our selves",
            "itself": "it self", "another": "an other",
            "inside": "in side", "outside": "out side",
            "without": "with out", "within": "with in",
            "into": "in to", "onto": "on to",
            "maybe": "may be", "nearby": "near by",
        }

        if clean in known_splits:
            replacement = known_splits[clean]
            if word[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            words[idx] = replacement + suffix
        else:
            # Generic split
            split_pos = len(clean) // 2
            words[idx] = clean[:split_pos] + " " + clean[split_pos:] + suffix

        return " ".join(words)

    def _remove_space(self, sentence: str) -> str:
        """Remove a random space between two words to create a joined-words error."""
        tokens = sentence.split(" ")
        if len(tokens) < 3:
            return sentence

        space_positions = []
        for i in range(1, len(tokens) - 1):
            left = tokens[i - 1] if tokens[i - 1] else ""
            curr = tokens[i] if tokens[i] else ""
            if (left and curr and left[-1].isalpha() and curr[0].isalpha()
                    and len(left) >= 2 and len(curr) >= 2):
                space_positions.append(i)

        if not space_positions:
            return sentence

        join_at = self.rng.choice(space_positions)
        tokens[join_at - 1] = tokens[join_at - 1] + tokens[join_at]
        del tokens[join_at]
        return " ".join(tokens)

    # ─── Main API ────────────────────────────────────────────────

    def corrupt_sentence(
        self, sentence: str, word_corruption_rate: float = 0.15
    ) -> str:
        """Corrupt a sentence, returning the corrupted version.

        Each word independently has word_corruption_rate chance of corruption.
        Also applies sentence-level corruptions (grammar, punctuation, etc).
        """
        tokens = re.findall(r"\S+|\s+", sentence)
        result_tokens = []

        for token in tokens:
            if token.isspace():
                result_tokens.append(token)
                continue

            match = re.match(r"^([A-Za-z'-]+)(.*)$", token)
            if match and len(match.group(1)) >= 3:
                word, rest = match.group(1), match.group(2)
                if self.rng.random() < word_corruption_rate:
                    word = self.corrupt_word(word)
                result_tokens.append(word + rest)
            else:
                result_tokens.append(token)

        result = "".join(result_tokens)

        # Sentence-level corruptions — each applied independently
        # with probability proportional to its weight
        total_sent_weight = sum(w for _, w in self._sentence_types)

        # Grammar corruption (includes determiners, prepositions, tense, etc)
        if self.rng.random() < (self.weights.grammar / total_sent_weight) * 0.5:
            grammar_result = corrupt_grammar(result, self.rng)
            if grammar_result:
                result = grammar_result

        # Missing word
        if self.rng.random() < (self.weights.missing_word / total_sent_weight) * 0.3:
            result = self._drop_word(result)

        # Extra/repeated word
        if self.rng.random() < (self.weights.extra_word / total_sent_weight) * 0.2:
            result = self._repeat_word(result)

        # Punctuation
        if self.rng.random() < (self.weights.punctuation / total_sent_weight) * 0.3:
            result = self._corrupt_punctuation(result)

        # Missing space
        if self.rng.random() < (self.weights.missing_space / total_sent_weight) * 0.15:
            result = self._remove_space(result)

        # Split compound word
        if self.rng.random() < (self.weights.split_word / total_sent_weight) * 0.1:
            result = self._split_compound(result)

        return result

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
        """Generate corruption pairs from clean sentences."""
        pairs = []
        for sentence in clean_sentences:
            for _ in range(variants_per_sentence):
                rate = self.rng.uniform(min_corruption_rate, max_corruption_rate)
                corrupted, original = self.generate_pair(sentence, rate)
                if corrupted != original:
                    pairs.append((corrupted, original))
        return pairs
