"""Rule-based grammar error injection.

Generates real-word errors that require sentence context to detect:
  - Tense errors (I seen -> I saw)
  - Agreement errors (They was -> They were)
  - Article errors (a apple -> an apple)
"""

import random
import re

# Subject-verb agreement corruptions.
# Maps (subject_pattern, correct_verb) -> wrong_verb
AGREEMENT_ERRORS = [
    # "they were" -> "they was"
    (r"\b(they|we)\s+were\b", r"\1 was"),
    (r"\b(they|we)\s+are\b", r"\1 is"),
    (r"\b(they|we)\s+have\b", r"\1 has"),
    (r"\b(they|we)\s+do\b", r"\1 does"),
    # "he/she was" -> "he/she were"
    (r"\b(he|she|it)\s+was\b", r"\1 were"),
    (r"\b(he|she|it)\s+is\b", r"\1 are"),
    (r"\b(he|she|it)\s+has\b", r"\1 have"),
    (r"\b(he|she|it)\s+does\b", r"\1 do"),
    # "I am" -> "I is"
    (r"\bI\s+am\b", "I is"),
    (r"\bI\s+was\b", "I were"),
]

# Tense corruptions
TENSE_ERRORS = [
    # Past participle used as simple past
    (r"\bI\s+saw\b", "I seen"),
    (r"\bI\s+went\b", "I gone"),
    (r"\bI\s+did\b", "I done"),
    (r"\bI\s+ran\b", "I run"),
    (r"\bI\s+came\b", "I come"),
    (r"\bI\s+ate\b", "I eat"),
    (r"\bI\s+gave\b", "I give"),
    (r"\bI\s+took\b", "I took"),  # already same
    (r"\bhe\s+saw\b", "he seen"),
    (r"\bhe\s+went\b", "he gone"),
    (r"\bhe\s+did\b", "he done"),
    (r"\bshe\s+saw\b", "she seen"),
    (r"\bshe\s+went\b", "she gone"),
    (r"\bthey\s+saw\b", "they seen"),
    (r"\bthey\s+went\b", "they gone"),
    (r"\bwe\s+saw\b", "we seen"),
    (r"\bwe\s+went\b", "we gone"),
]

# Article errors: a/an swap
ARTICLE_PATTERN_A_TO_AN = re.compile(r"\ba\s+([aeiouAEIOU]\w+)")
ARTICLE_PATTERN_AN_TO_A = re.compile(r"\ban\s+([^aeiouAEIOU\s]\w+)")


def corrupt_agreement(sentence: str, rng: random.Random | None = None) -> str | None:
    """Apply a subject-verb agreement error if possible."""
    rng = rng or random.Random()
    applicable = []
    for pattern, replacement in AGREEMENT_ERRORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            applicable.append((pattern, replacement))

    if not applicable:
        return None

    pattern, replacement = rng.choice(applicable)
    return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)


def corrupt_tense(sentence: str, rng: random.Random | None = None) -> str | None:
    """Apply a tense error if possible."""
    rng = rng or random.Random()
    applicable = []
    for pattern, replacement in TENSE_ERRORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            applicable.append((pattern, replacement))

    if not applicable:
        return None

    pattern, replacement = rng.choice(applicable)
    return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)


def corrupt_article(sentence: str, rng: random.Random | None = None) -> str | None:
    """Swap a/an incorrectly."""
    rng = rng or random.Random()

    # Find all a->an and an->a opportunities
    a_matches = list(ARTICLE_PATTERN_A_TO_AN.finditer(sentence))
    an_matches = list(ARTICLE_PATTERN_AN_TO_A.finditer(sentence))

    candidates = []
    for m in a_matches:
        candidates.append(("a_to_an", m))
    for m in an_matches:
        candidates.append(("an_to_a", m))

    if not candidates:
        return None

    kind, match = rng.choice(candidates)
    start, end = match.start(), match.end()
    if kind == "a_to_an":
        # "a apple" -> already wrong, we want "an apple" -> "a apple"
        # Wait — we want to corrupt, so: find correct "a <consonant>" and make it "an <consonant>"
        # Actually: find "a <vowel-word>" (which is wrong in text) — no.
        # Let's be precise: find correct usage "an <vowel>" and break it,
        # or find correct usage "a <consonant>" and break it.
        # "a <vowel-word>" means the text has a mistake (or we should make one)
        # For corruption: change "a <consonant-word>" to "an <consonant-word>"
        # The regex matched "a <vowel-word>" which is already correct -> skip
        pass
    if kind == "an_to_a":
        # "an <consonant>" -> text has mistake. We want to CREATE mistakes.
        pass

    # Simpler approach: just swap a<->an at a random article position
    article_positions = [(m.start(), m.group()) for m in re.finditer(r"\b(an?)\s", sentence)]
    if not article_positions:
        return None

    pos, article = rng.choice(article_positions)
    if article == "a":
        new = sentence[:pos] + "an" + sentence[pos + 1 :]
    else:  # "an"
        new = sentence[:pos] + "a" + sentence[pos + 2 :]

    return new


def corrupt_grammar(sentence: str, rng: random.Random | None = None) -> str | None:
    """Apply a random grammar corruption to the sentence.

    Returns None if no grammar corruption is applicable.
    """
    rng = rng or random.Random()
    corruptors = [corrupt_agreement, corrupt_tense, corrupt_article]
    rng.shuffle(corruptors)

    for fn in corruptors:
        result = fn(sentence, rng)
        if result is not None and result != sentence:
            return result

    return None
