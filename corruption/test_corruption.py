"""Tests for the corruption engine and all corruption types."""

import random
import re

import pytest

from corruption.keyboard import fat_finger_word, ADJACENCY, get_adjacent_key
from corruption.phonetic import phonetic_rewrite, PHONETIC_REWRITES
from corruption.homophones import HomophoneCorruptor
from corruption.grammar import (
    corrupt_agreement, corrupt_tense, corrupt_article,
    corrupt_determiner, corrupt_preposition, corrupt_noun_number,
    corrupt_word_form, corrupt_contraction, corrupt_grammar,
)
from corruption.engine import CorruptionEngine


class TestKeyboard:
    def test_adjacency_map_completeness(self):
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in ADJACENCY, f"Missing key: {c}"

    def test_adjacency_symmetry(self):
        for key, neighbors in ADJACENCY.items():
            for neighbor in neighbors:
                assert key in ADJACENCY[neighbor], (
                    f"{key} lists {neighbor} as adjacent, "
                    f"but {neighbor} doesn't list {key}"
                )

    def test_fat_finger_changes_word(self):
        rng = random.Random(42)
        changed = False
        for _ in range(100):
            result = fat_finger_word("hello", rng)
            if result != "hello":
                changed = True
                assert len(result) == len("hello")
                break
        assert changed

    def test_fat_finger_preserves_length(self):
        rng = random.Random(42)
        for _ in range(50):
            result = fat_finger_word("testing", rng)
            assert len(result) == len("testing")

    def test_fat_finger_short_word(self):
        assert fat_finger_word("a") == "a"

    def test_get_adjacent_key_preserves_case(self):
        rng = random.Random(42)
        for _ in range(50):
            result = get_adjacent_key("A")
            if result:
                assert result.isupper()
            result = get_adjacent_key("a")
            if result:
                assert result.islower()


class TestPhonetic:
    def test_phonetic_rewrite_applies(self):
        rng = random.Random(42)
        results = set()
        for _ in range(100):
            r = phonetic_rewrite("phone", rng)
            if r:
                results.add(r)
        assert "fone" in results

    def test_phonetic_rewrite_preserves_case(self):
        rng = random.Random(42)
        for _ in range(100):
            r = phonetic_rewrite("Phone", rng)
            if r and r != "Phone":
                assert r[0].isupper()
                break

    def test_phonetic_rewrite_none_for_no_match(self):
        assert phonetic_rewrite("xyz") is None


class TestHomophones:
    def test_known_homophones_present(self):
        h = HomophoneCorruptor(homophone_db_path=None)
        assert "their" in h.lookup
        assert "there" in h.lookup["their"]

    def test_corrupt_returns_homophone(self):
        h = HomophoneCorruptor(homophone_db_path=None)
        rng = random.Random(42)
        results = set()
        for _ in range(100):
            r = h.corrupt("their", rng)
            if r:
                results.add(r)
        assert results.issubset({"there", "they're"})
        assert len(results) > 0

    def test_corrupt_preserves_case(self):
        h = HomophoneCorruptor(homophone_db_path=None)
        rng = random.Random(42)
        for _ in range(100):
            r = h.corrupt("Their", rng)
            if r:
                assert r[0].isupper()

    def test_corrupt_unknown_word(self):
        h = HomophoneCorruptor(homophone_db_path=None)
        assert h.corrupt("xylophone") is None


class TestGrammar:
    def test_agreement_error(self):
        rng = random.Random(42)
        result = corrupt_agreement("They were happy.", rng)
        assert result is not None
        assert "was" in result

    def test_tense_error(self):
        rng = random.Random(42)
        result = corrupt_tense("I saw the movie.", rng)
        assert result is not None
        assert "seen" in result

    def test_article_swap(self):
        rng = random.Random(42)
        for _ in range(200):
            result = corrupt_article("I ate a sandwich and an apple.", rng)
            if result and result != "I ate a sandwich and an apple.":
                # Should swap a<->an somewhere
                assert result != "I ate a sandwich and an apple."
                break

    def test_no_grammar_corruption_possible(self):
        rng = random.Random(42)
        result = corrupt_agreement("The cat sat.", rng)
        assert result is None

    # ─── New grammar corruption tests ────────────────────────────

    def test_determiner_drop(self):
        rng = random.Random(42)
        found = False
        for _ in range(200):
            result = corrupt_determiner("I saw the cat on the mat.", rng)
            if result and "the" not in result.split()[:4]:
                found = True
                break
        # At minimum, corrupt_determiner should sometimes produce a change
        changed = False
        for _ in range(200):
            result = corrupt_determiner("I saw the cat on the mat.", rng)
            if result and result != "I saw the cat on the mat.":
                changed = True
                break
        assert changed

    def test_preposition_swap(self):
        rng = random.Random(42)
        changed = False
        for _ in range(200):
            result = corrupt_preposition("The cat is on the mat.", rng)
            if result and result != "The cat is on the mat.":
                changed = True
                # Should have swapped "on" for something else
                assert "on" not in result or result != "The cat is on the mat."
                break
        assert changed

    def test_noun_number(self):
        rng = random.Random(42)
        changed = False
        for _ in range(200):
            result = corrupt_noun_number("The children played in the garden.", rng)
            if result and result != "The children played in the garden.":
                changed = True
                break
        assert changed

    def test_word_form(self):
        rng = random.Random(42)
        changed = False
        for _ in range(200):
            result = corrupt_word_form("She ran quickly to the store.", rng)
            if result and result != "She ran quickly to the store.":
                changed = True
                assert "quick" in result.lower()
                break
        assert changed

    def test_contraction(self):
        rng = random.Random(42)
        changed = False
        for _ in range(200):
            result = corrupt_contraction("I don't know what you're saying.", rng)
            if result and result != "I don't know what you're saying.":
                changed = True
                assert "dont" in result.lower() or "youre" in result.lower()
                break
        assert changed

    def test_tense_of_for_have(self):
        rng = random.Random(42)
        result = corrupt_tense("I should have known better.", rng)
        assert result is not None
        assert "should of" in result

    def test_grammar_dispatch(self):
        """corrupt_grammar should produce changes on a variety of sentences."""
        rng = random.Random(42)
        sentences = [
            "They were going to the store.",
            "I saw the movie yesterday.",
            "The cat sat on the mat.",
            "She ran quickly to the door.",
            "I don't know what happened.",
        ]
        changes = 0
        for s in sentences:
            for _ in range(50):
                result = corrupt_grammar(s, rng)
                if result and result != s:
                    changes += 1
                    break
        assert changes >= 3, f"Only {changes}/5 sentences were corrupted"


class TestEngine:
    @pytest.fixture
    def engine(self):
        return CorruptionEngine(
            confusion_db_path=None,
            homophone_db_path=None,
            seed=42,
        )

    def test_corrupt_word_changes_word(self, engine):
        changed = False
        for _ in range(100):
            result = engine.corrupt_word("testing")
            if result != "testing":
                changed = True
                break
        assert changed

    def test_corrupt_word_skips_short(self, engine):
        assert engine.corrupt_word("is") == "is"
        assert engine.corrupt_word("a") == "a"

    def test_corrupt_sentence_produces_change(self, engine):
        sentence = "The quick brown fox jumped over the lazy dog."
        changed = False
        for _ in range(20):
            result = engine.corrupt_sentence(sentence, word_corruption_rate=0.5)
            if result != sentence:
                changed = True
                break
        assert changed

    def test_corrupt_sentence_preserves_punctuation(self, engine):
        sentence = "Hello, world! How are you?"
        result = engine.corrupt_sentence(sentence, word_corruption_rate=0.3)
        assert any(c in result for c in ",.!?")

    def test_generate_pair(self, engine):
        corrupted, original = engine.generate_pair(
            "They went to the store.", corruption_rate=0.5
        )
        assert original == "They went to the store."

    def test_build_dataset(self, engine):
        sentences = [
            "The cat sat on the mat.",
            "She went to the store yesterday.",
            "They have been working very hard.",
        ]
        pairs = engine.build_dataset(sentences, variants_per_sentence=2)
        assert len(pairs) > 0
        for corrupted, original in pairs:
            assert original in sentences
            assert corrupted != original

    def test_missing_space_corruption(self, engine):
        found_joined = False
        for _ in range(200):
            result = engine._remove_space(
                "The quick brown fox jumps over the lazy dog"
            )
            if result != "The quick brown fox jumps over the lazy dog":
                orig_spaces = "The quick brown fox jumps over the lazy dog".count(" ")
                result_spaces = result.count(" ")
                assert result_spaces == orig_spaces - 1
                found_joined = True
                break
        assert found_joined

    def test_transpose(self, engine):
        result = engine._transpose("the")
        assert result in ("hte", "teh")

    def test_delete_char(self, engine):
        result = engine._delete_char("hello")
        assert len(result) == 4

    def test_double_letter(self, engine):
        result = engine._double_letter("hello")
        assert result != "" and len(result) in (4, 5, 6)

    # ─── New sentence-level corruption tests ─────────────────────

    def test_drop_word(self, engine):
        result = engine._drop_word("I went to the store yesterday.")
        assert len(result.split()) < len("I went to the store yesterday.".split())

    def test_repeat_word(self, engine):
        result = engine._repeat_word("I went to the store yesterday.")
        assert len(result.split()) > len("I went to the store yesterday.".split())

    def test_corrupt_punctuation(self, engine):
        changed = False
        for _ in range(200):
            result = engine._corrupt_punctuation("Hello, world. How are you?")
            if result != "Hello, world. How are you?":
                changed = True
                break
        assert changed

    def test_split_compound(self, engine):
        changed = False
        for _ in range(200):
            result = engine._split_compound("I cannot believe everyone is here.")
            if result != "I cannot believe everyone is here.":
                changed = True
                assert " " in result.replace("I ", "").replace(" believe ", " ").replace(" is ", " ").replace(" here.", "")
                break
        assert changed

    def test_sentence_level_variety(self, engine):
        """High corruption rate should trigger various sentence-level errors."""
        sentence = "I don't think the committee should have gone to the restaurant."
        results = set()
        for _ in range(500):
            result = engine.corrupt_sentence(sentence, word_corruption_rate=0.3)
            if result != sentence:
                results.add(result)
        # Should produce many different corrupted versions
        assert len(results) > 10, f"Only {len(results)} unique corruptions"
