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
    Calibrated against BEA-60K actual error distribution:
      char_insertion_needed (missing letter): 29.6%
      char_substitution: 22.9%
      char_deletion_needed (extra letter): 17.7%
      multi_char (edit dist 2+): 13.9%
      different_word: 6.8%
      transposition: 6.1%
      verb/grammar: 2-3%
    """
    # Calibrated against BOTH BEA-60K (spelling) AND C4_200M (GEC).
    # C4_200M distribution: 31% word_choice, 19% delete, 17% insert,
    # 9% missing_char, 7% multi_char, 6% case, 4% extra_char, 3% sub, 3% grammar

    # Character-level spelling (~24% of C4 errors)
    deletion: float = 15.0               # missing character
    keyboard_adjacency: float = 8.0      # char substitution
    insertion: float = 8.0               # extra character
    transposition: float = 3.0           # swapped adjacent chars
    double_letter: float = 3.0           # add/remove doubled letter
    phonetic_rewrite: float = 2.0        # character-level phonetic (ph->f etc)
    suffix_confusion: float = 2.0        # -ible/-able, -ie/-ei
    vowel_swap: float = 2.0             # vowel confusion

    # Real-word errors
    homophone: float = 3.0              # their/there/they're etc
    phonetic_word: float = 1.0          # whole-word phonetic swap

    # Grammar/word-level errors (~35% of C4 errors - THE BIG GAP)
    grammar: float = 30.0              # determiners, prepositions, tense, word_form

    # Sentence-level (~35% of C4 errors — word deletion/insertion dominate)
    missing_word: float = 20.0         # drop a word (18.8% of C4 errors!)
    extra_word: float = 8.0            # repeat/insert a word
    punctuation: float = 3.0           # comma, apostrophe, period errors
    missing_space: float = 2.0         # join adjacent words
    split_word: float = 1.0            # add space inside a word
    capitalization: float = 6.0        # case errors (5.6% of C4)


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
            ("deletion", self.weights.deletion),
            ("keyboard_adjacency", self.weights.keyboard_adjacency),
            ("insertion", self.weights.insertion),
            ("transposition", self.weights.transposition),
            ("double_letter", self.weights.double_letter),
            ("phonetic_rewrite", self.weights.phonetic_rewrite),
            ("suffix_confusion", self.weights.suffix_confusion),
            ("vowel_swap", self.weights.vowel_swap),
            ("homophone", self.weights.homophone),
            ("phonetic_word", self.weights.phonetic_word),
            ("capitalization", self.weights.capitalization),
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

    def corrupt_word(self, word: str, multi_edit: bool = False) -> str:
        """Apply a random corruption to a single word.

        If multi_edit is True, apply 2-3 corruptions to simulate severe
        misspellings (accounts for ~50% of BEA-60K failures).
        """
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

        # Multi-edit: stack 1-2 more corruptions on top
        if multi_edit and result != word and len(result) >= 3:
            extra = self.rng.randint(1, 2)
            for _ in range(extra):
                ctype2 = self._pick_word_corruption()
                result2 = self._apply_word_corruption(result, ctype2)
                if result2 != result and len(result2) >= 2:
                    result = result2

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

        elif ctype == "vowel_swap":
            return self._vowel_swap(word)

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
        """Delete a random character. Prefers interior but allows any position."""
        if len(word) <= 2:
            return word
        if len(word) <= 3:
            pos = self.rng.randint(0, len(word) - 1)
        elif self.rng.random() < 0.8:
            pos = self.rng.randint(1, len(word) - 2)  # interior (80%)
        else:
            pos = self.rng.randint(0, len(word) - 1)  # any position (20%)
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

    def _vowel_swap(self, word: str) -> str:
        """Swap a vowel for a different vowel (e.g. 'a' -> 'e')."""
        if len(word) < 3:
            return word
        vowels = "aeiou"
        positions = [i for i in range(1, len(word) - 1)
                    if word[i].lower() in vowels]
        if not positions:
            return word
        pos = self.rng.choice(positions)
        old = word[pos].lower()
        alternatives = [v for v in vowels if v != old]
        new = self.rng.choice(alternatives)
        if word[pos].isupper():
            new = new.upper()
        return word[:pos] + new + word[pos + 1:]

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
        ~20% of corrupted words get multi-edit (2-3 stacked corruptions) to
        simulate severe misspellings seen in real human writing.
        Also applies sentence-level corruptions (grammar, punctuation, etc).
        """
        # Case variation: 5% chance to convert sentence to ALL CAPS or
        # random case before corrupting
        case_mode = None
        if self.rng.random() < 0.05:
            case_choice = self.rng.randint(0, 2)
            if case_choice == 0:
                sentence = sentence.upper()
                case_mode = "upper"
            elif case_choice == 1:
                # Random words uppercased
                words = sentence.split()
                words = [w.upper() if self.rng.random() < 0.3 else w for w in words]
                sentence = " ".join(words)
                case_mode = "mixed"

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
                    # 40% of corruptions are multi-edit (BEA-60K: 14% of errors are multi-char)
                    multi = self.rng.random() < 0.40
                    word = self.corrupt_word(word, multi_edit=multi)
                result_tokens.append(word + rest)
            else:
                result_tokens.append(token)

        result = "".join(result_tokens)

        # Sentence-level corruptions — apply multiple per sentence
        # C4_200M has ~19% word deletion and ~17% word insertion, so these
        # need to fire frequently. Each fires independently.

        # Grammar corruption (determiners, prepositions, tense, word_form)
        # Apply 1-2 grammar corruptions per sentence
        for _ in range(self.rng.randint(0, 2)):
            if self.rng.random() < 0.4:
                grammar_result = corrupt_grammar(result, self.rng)
                if grammar_result:
                    result = grammar_result

        # Missing word — drop 1-3 words per sentence (C4: 18.8%)
        num_drops = self.rng.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
        for _ in range(num_drops):
            result = self._drop_word(result)

        # Extra/repeated word (C4: ~5% combined with insert)
        if self.rng.random() < 0.15:
            result = self._repeat_word(result)

        # Punctuation
        if self.rng.random() < 0.10:
            result = self._corrupt_punctuation(result)

        # Missing space
        if self.rng.random() < 0.05:
            result = self._remove_space(result)

        # Split compound word
        if self.rng.random() < 0.03:
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
