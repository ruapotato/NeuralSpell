"""Rule-based grammar and real-word error injection.

Generates errors that require sentence context to detect, based on
real human error statistics from BEA-2019, NUCLE, FCE, and GCSE studies.

Error types and approximate frequencies in human writing:
  - Determiner/article errors: 10-16% of all errors (BEA-2019)
  - Preposition errors: 7-11% (NUCLE, FCE)
  - Verb tense errors: 6-7% (CoNLL-2014)
  - Noun number (plural/singular): 3-8% (NUCLE)
  - Word form (adj/adv, noun/verb): 3-5% (BEA-2019)
  - Subject-verb agreement: 2-3% (CoNLL-2014)
  - Contraction errors: 1-2%
  - "of" for "have" pattern
"""

import random
import re


# ─── Determiner / Article Errors ─────────────────────────────────
# 10-16% of all human errors (largest single category in NUCLE)

def corrupt_determiner(sentence: str, rng: random.Random) -> str | None:
    """Corrupt articles and determiners: drop, insert, or swap."""
    choice = rng.randint(0, 3)

    if choice == 0:
        # Drop an article
        pattern = re.compile(r"\b(the|a|an)\s+", re.IGNORECASE)
        matches = list(pattern.finditer(sentence))
        if matches:
            m = rng.choice(matches)
            return sentence[:m.start()] + sentence[m.end():]

    elif choice == 1:
        # Insert a wrong article before a noun-like word
        # Find words after verbs or prepositions where an article might go
        pattern = re.compile(r"\b(is|are|was|were|have|has|in|on|at|for|with|of|to)\s+([a-z])", re.IGNORECASE)
        matches = list(pattern.finditer(sentence))
        if matches:
            m = rng.choice(matches)
            article = rng.choice(["the ", "a ", "an "])
            insert_pos = m.start(2)
            return sentence[:insert_pos] + article + sentence[insert_pos:]

    elif choice == 2:
        # Swap the/a/an
        pattern = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
        matches = list(pattern.finditer(sentence))
        if matches:
            m = rng.choice(matches)
            original = m.group()
            replacements = {"the": "a", "a": "the", "an": "the",
                          "The": "A", "A": "The", "An": "The"}
            new = replacements.get(original, "the")
            return sentence[:m.start()] + new + sentence[m.end():]

    else:
        # "a" <-> "an" swap (original behavior)
        return corrupt_article(sentence, rng)

    return None


# ─── Preposition Errors ──────────────────────────────────────────
# 7-11% of all errors (FCE, NUCLE)

PREPOSITION_CONFUSIONS = {
    "in": ["on", "at", "into", "to"],
    "on": ["in", "at", "onto", "to"],
    "at": ["in", "on", "to"],
    "to": ["for", "at", "into", "in"],
    "for": ["to", "of", "with"],
    "of": ["for", "from", "in"],
    "with": ["by", "from", "of"],
    "by": ["with", "from", "through"],
    "from": ["of", "by", "out of"],
    "about": ["on", "of", "around"],
    "into": ["in", "to", "onto"],
    "through": ["by", "across", "over"],
    "between": ["among", "in"],
    "during": ["in", "while", "for"],
    "since": ["for", "from", "after"],
    "until": ["to", "till", "by"],
}


def corrupt_preposition(sentence: str, rng: random.Random) -> str | None:
    """Swap a preposition with a commonly confused alternative."""
    choice = rng.randint(0, 2)

    if choice <= 1:
        # Swap a preposition
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in PREPOSITION_CONFUSIONS) + r")\b",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(sentence))
        if matches:
            m = rng.choice(matches)
            original = m.group().lower()
            if original in PREPOSITION_CONFUSIONS:
                replacement = rng.choice(PREPOSITION_CONFUSIONS[original])
                # Preserve case
                if m.group()[0].isupper():
                    replacement = replacement.capitalize()
                return sentence[:m.start()] + replacement + sentence[m.end():]
    else:
        # Drop a preposition
        pattern = re.compile(
            r"\b(in|on|at|to|for|of|with|by|from)\s+",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(sentence))
        if matches:
            m = rng.choice(matches)
            return sentence[:m.start()] + sentence[m.start() + len(m.group(1)) + 1:]

    return None


# ─── Noun Number Errors ──────────────────────────────────────────
# 3-8% of all errors (NUCLE reports 8.4%)

IRREGULAR_PLURALS = {
    "children": "child", "child": "children",
    "people": "person", "person": "people",
    "men": "man", "man": "men",
    "women": "woman", "woman": "women",
    "teeth": "tooth", "tooth": "teeth",
    "feet": "foot", "foot": "feet",
    "mice": "mouse", "mouse": "mice",
    "geese": "goose", "goose": "geese",
    "lives": "life", "life": "lives",
    "knives": "knife", "knife": "knives",
    "wives": "wife", "wife": "wives",
    "leaves": "leaf", "leaf": "leaves",
    "shelves": "shelf", "shelf": "shelves",
    "criteria": "criterion", "criterion": "criteria",
    "phenomena": "phenomenon", "phenomenon": "phenomena",
    "analyses": "analysis", "analysis": "analyses",
    "data": "datum", "datum": "data",
}


def corrupt_noun_number(sentence: str, rng: random.Random) -> str | None:
    """Corrupt noun number: add/remove plural -s, swap irregular plurals."""
    words = sentence.split()
    if len(words) < 3:
        return None

    # Find potential nouns (words that aren't at the start, not too short)
    candidates = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^\w]", "", w)
        if len(clean) < 3 or i == 0:
            continue
        # Check for irregular plurals
        if clean.lower() in IRREGULAR_PLURALS:
            candidates.append((i, "irregular"))
        # Check for regular plural -s/-es
        elif clean.lower().endswith("s") and not clean.lower().endswith("ss") and len(clean) > 3:
            candidates.append((i, "remove_s"))
        # Words that could take a plural -s
        elif clean.lower()[-1] not in "sxz" and clean[0].islower():
            candidates.append((i, "add_s"))

    if not candidates:
        return None

    idx, action = rng.choice(candidates)
    word = words[idx]
    clean = re.sub(r"([^\w]+)$", "", word)
    suffix = word[len(clean):]

    if action == "irregular":
        replacement = IRREGULAR_PLURALS.get(clean.lower(), clean)
        if clean[0].isupper():
            replacement = replacement.capitalize()
        words[idx] = replacement + suffix
    elif action == "remove_s":
        # Remove trailing s/es
        if clean.lower().endswith("ies") and len(clean) > 4:
            words[idx] = clean[:-3] + "y" + suffix
        elif clean.lower().endswith("es") and len(clean) > 3:
            words[idx] = clean[:-2] + suffix
        else:
            words[idx] = clean[:-1] + suffix
    elif action == "add_s":
        if clean.lower().endswith("y") and clean[-2].lower() not in "aeiou":
            words[idx] = clean[:-1] + "ies" + suffix
        elif clean.lower()[-1] in "shx" or clean.lower().endswith("ch"):
            words[idx] = clean + "es" + suffix
        else:
            words[idx] = clean + "s" + suffix

    return " ".join(words)


# ─── Verb Form Errors ────────────────────────────────────────────
# 3-5% of all errors (BEA-2019)

VERB_FORM_ERRORS = [
    # Gerund/infinitive confusion
    (r"\bto\s+(go|come|make|take|get|see|give|run|eat)\b", lambda m, rng: f"to {m.group(1)}ing"),
    (r"\b(going|coming|making|taking|getting|seeing|giving|running|eating)\b",
     lambda m, rng: m.group(1).replace("ing", "") if not m.group(1).endswith("eing") else m.group(1)[:-3]),
    # Modal + past participle (should went -> should go)
    (r"\b(should|would|could|will|can|might|must)\s+(went|came|saw|took|got|made|gave|ran|ate)\b",
     lambda m, rng: f"{m.group(1)} " + {"went": "go", "came": "come", "saw": "see", "took": "take",
      "got": "get", "made": "make", "gave": "give", "ran": "run", "ate": "eat"}.get(m.group(2), m.group(2))),
    # Missing -ed on past tense
    (r"\b(want|need|start|help|seem|happen|work|walk|talk|look)ed\b",
     lambda m, rng: m.group(1)),
    # Extra -ed (already past)
    (r"\b(went|came|saw|took|got|made|gave|ran|ate|thought|brought|caught|felt|found)\b",
     lambda m, rng: m.group(1) + "ed" if rng.random() < 0.3 else m.group(0)),
]


def corrupt_verb_form(sentence: str, rng: random.Random) -> str | None:
    """Corrupt verb forms: gerund/infinitive, modal+past, missing -ed."""
    applicable = []
    for pattern, replacement_fn in VERB_FORM_ERRORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            applicable.append((pattern, replacement_fn))

    if not applicable:
        return None

    pattern, replacement_fn = rng.choice(applicable)
    result = re.sub(pattern, lambda m: replacement_fn(m, rng), sentence, count=1, flags=re.IGNORECASE)
    return result if result != sentence else None


# ─── Word Form Errors ────────────────────────────────────────────
# 3-5% of all errors (adjective/adverb, noun/verb confusions)

WORD_FORM_SWAPS = {
    # Adjective <-> Adverb
    "quickly": "quick", "quick": "quickly",
    "slowly": "slow", "slow": "slowly",
    "badly": "bad", "bad": "badly",
    "easily": "easy", "easy": "easily",
    "happily": "happy", "happy": "happily",
    "carefully": "careful", "careful": "carefully",
    "completely": "complete", "complete": "completely",
    "really": "real", "real": "really",
    "hardly": "hard", "hard": "hardly",
    "lately": "late", "late": "lately",
    "highly": "high", "high": "highly",
    "deeply": "deep", "deep": "deeply",
    "widely": "wide", "wide": "widely",
    "clearly": "clear", "clear": "clearly",
    "properly": "proper", "proper": "properly",
    "seriously": "serious", "serious": "seriously",
    "differently": "different", "different": "differently",
    "specifically": "specific", "specific": "specifically",
    # Noun <-> Verb
    "advice": "advise", "advise": "advice",
    "practice": "practise", "practise": "practice",
    "effect": "affect", "affect": "effect",
    "breath": "breathe", "breathe": "breath",
    "choice": "choose", "choose": "choice",
    "loss": "lose", "lose": "loss",
    "belief": "believe", "believe": "belief",
    "proof": "prove", "prove": "proof",
    # Adjective <-> Noun
    "strength": "strong", "strong": "strength",
    "length": "long", "long": "length",
    "height": "high",
    "width": "wide",
    "depth": "deep",
    "truth": "true", "true": "truth",
    "success": "successful", "successful": "success",
    "importance": "important", "important": "importance",
    "difference": "different",
    "appearance": "apparent",
    # Irregular verb form confusions (BEA-60K common failures)
    "chose": "choose", "choose": "chose", "chosen": "choosen",
    "become": "became", "became": "become", "becomes": "becames",
    "taken": "took", "took": "taken",
    "written": "wrote", "wrote": "written",
    "spoken": "spoke", "spoke": "spoken",
    "broken": "broke", "broke": "broken",
    "forgotten": "forgot", "forgot": "forgotten",
    "driven": "drove", "drove": "driven",
    "risen": "rose", "rose": "risen",
    "fallen": "fell", "fell": "fallen",
    "hidden": "hid", "hid": "hidden",
    "begun": "began", "began": "begun",
    # -ness / -ment / -tion confusions
    "happiness": "happyness", "sadness": "sadnes",
    "comfortable": "confortable", "comfortably": "confortably",
    "necessary": "nessesary", "necessarily": "nessesarily",
    "definitely": "definately", "separate": "seperate",
    "environment": "enviroment", "government": "goverment",
    "accommodation": "accomodation", "occurrence": "occurence",
    "millennium": "millenium", "committee": "commitee",
    "embarrass": "embarass", "harassment": "harasment",
    "maintenance": "maintanence", "independence": "independance",
    "existence": "existance", "resistance": "resistence",
    "experience": "experiance", "audience": "audiance",
    "fascinated": "facinated", "fascinating": "facinating",
    "technology": "tecnology", "technique": "tecnique",
    "demonstration": "demostration", "certificate": "certificat",
    "mystery": "mistery", "mysteries": "misteries",
    "passive": "pasive", "aggressive": "agressive",
    "possession": "posession", "professional": "proffesional",
    "recommend": "recomend", "recommendation": "recomendation",
}


def corrupt_word_form(sentence: str, rng: random.Random) -> str | None:
    """Swap a word with a commonly confused form (adj/adv, noun/verb)."""
    words = sentence.split()
    candidates = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^\w']", "", w).lower()
        if clean in WORD_FORM_SWAPS:
            candidates.append(i)

    if not candidates:
        return None

    idx = rng.choice(candidates)
    word = words[idx]
    clean = re.sub(r"([^\w']+)$", "", word)
    suffix = word[len(clean):]
    replacement = WORD_FORM_SWAPS.get(clean.lower(), clean)
    if clean[0].isupper():
        replacement = replacement.capitalize()
    words[idx] = replacement + suffix
    return " ".join(words)


# ─── Subject-Verb Agreement (expanded) ───────────────────────────

AGREEMENT_ERRORS = [
    (r"\b(they|we)\s+were\b", r"\1 was"),
    (r"\b(they|we)\s+are\b", r"\1 is"),
    (r"\b(they|we)\s+have\b", r"\1 has"),
    (r"\b(they|we)\s+do\b", r"\1 does"),
    (r"\b(he|she|it)\s+was\b", r"\1 were"),
    (r"\b(he|she|it)\s+is\b", r"\1 are"),
    (r"\b(he|she|it)\s+has\b", r"\1 have"),
    (r"\b(he|she|it)\s+does\b", r"\1 do"),
    (r"\bI\s+am\b", "I is"),
    (r"\bI\s+was\b", "I were"),
    # Expanded: third person singular present
    (r"\b(he|she|it)\s+go\b", r"\1 goes"),
    (r"\b(he|she|it)\s+goes\b", r"\1 go"),
    (r"\b(they|we|I)\s+goes\b", r"\1 go"),
    (r"\b(he|she|it)\s+need\b", r"\1 needs"),
    (r"\b(he|she|it)\s+needs\b", r"\1 need"),
    (r"\b(he|she|it)\s+want\b", r"\1 wants"),
    (r"\b(he|she|it)\s+wants\b", r"\1 want"),
    (r"\b(he|she|it)\s+make\b", r"\1 makes"),
    (r"\b(he|she|it)\s+makes\b", r"\1 make"),
    # "there is/are" confusion
    (r"\bthere\s+are\b", "there is"),
    (r"\bthere\s+is\b", "there are"),
]


def corrupt_agreement(sentence: str, rng: random.Random | None = None) -> str | None:
    rng = rng or random.Random()
    applicable = []
    for pattern, replacement in AGREEMENT_ERRORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            applicable.append((pattern, replacement))
    if not applicable:
        return None
    pattern, replacement = rng.choice(applicable)
    return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)


# ─── Tense Errors (expanded) ─────────────────────────────────────

TENSE_ERRORS = [
    (r"\bI\s+saw\b", "I seen"),
    (r"\bI\s+went\b", "I gone"),
    (r"\bI\s+did\b", "I done"),
    (r"\bI\s+ran\b", "I run"),
    (r"\bI\s+came\b", "I come"),
    (r"\bI\s+ate\b", "I eat"),
    (r"\bI\s+gave\b", "I give"),
    (r"\bhe\s+saw\b", "he seen"),
    (r"\bhe\s+went\b", "he gone"),
    (r"\bhe\s+did\b", "he done"),
    (r"\bshe\s+saw\b", "she seen"),
    (r"\bshe\s+went\b", "she gone"),
    (r"\bthey\s+saw\b", "they seen"),
    (r"\bthey\s+went\b", "they gone"),
    (r"\bwe\s+saw\b", "we seen"),
    (r"\bwe\s+went\b", "we gone"),
    # Expanded: present/past confusion
    (r"\bhas\s+been\b", "has being"),
    (r"\bhave\s+been\b", "have being"),
    (r"\bwould\s+have\b", "would of"),  # very common error
    (r"\bcould\s+have\b", "could of"),
    (r"\bshould\s+have\b", "should of"),
    (r"\bmight\s+have\b", "might of"),
    (r"\bmust\s+have\b", "must of"),
    # Past participle without auxiliary
    (r"\bhas\s+gone\b", "has went"),
    (r"\bhave\s+gone\b", "have went"),
    (r"\bhas\s+done\b", "has did"),
    (r"\bhave\s+done\b", "have did"),
    (r"\bhas\s+seen\b", "has saw"),
    (r"\bhave\s+seen\b", "have saw"),
]


def corrupt_tense(sentence: str, rng: random.Random | None = None) -> str | None:
    rng = rng or random.Random()
    applicable = []
    for pattern, replacement in TENSE_ERRORS:
        if re.search(pattern, sentence, re.IGNORECASE):
            applicable.append((pattern, replacement))
    if not applicable:
        return None
    pattern, replacement = rng.choice(applicable)
    return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)


# ─── Article a/an swap ───────────────────────────────────────────

def corrupt_article(sentence: str, rng: random.Random | None = None) -> str | None:
    """Swap a/an incorrectly."""
    rng = rng or random.Random()
    article_positions = [(m.start(), m.group()) for m in re.finditer(r"\b(an?)\s", sentence)]
    if not article_positions:
        return None
    pos, article = rng.choice(article_positions)
    if article == "a":
        new = sentence[:pos] + "an" + sentence[pos + 1:]
    else:
        new = sentence[:pos] + "a" + sentence[pos + 2:]
    return new


# ─── Contraction Errors ──────────────────────────────────────────

CONTRACTION_EXPAND = {
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt",
    "can't": "cant", "won't": "wont", "wouldn't": "wouldnt",
    "couldn't": "couldnt", "shouldn't": "shouldnt",
    "isn't": "isnt", "aren't": "arent", "wasn't": "wasnt",
    "weren't": "werent", "hasn't": "hasnt", "haven't": "havent",
    "hadn't": "hadnt", "it's": "its", "I'm": "Im",
    "I'll": "Ill", "I've": "Ive", "I'd": "Id",
    "you're": "youre", "you'll": "youll", "you've": "youve",
    "we're": "were", "we'll": "well", "we've": "weve",
    "they're": "theyre", "they'll": "theyll", "they've": "theyve",
    "that's": "thats", "there's": "theres", "here's": "heres",
    "what's": "whats", "who's": "whos", "let's": "lets",
}


def corrupt_contraction(sentence: str, rng: random.Random) -> str | None:
    """Remove apostrophe from contractions (dont, cant, etc)."""
    candidates = []
    for contraction, wrong in CONTRACTION_EXPAND.items():
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        matches = list(pattern.finditer(sentence))
        if matches:
            candidates.append((matches, contraction, wrong))

    if not candidates:
        return None

    matches, contraction, wrong = rng.choice(candidates)
    m = rng.choice(matches)
    # Preserve case
    original = m.group()
    if original[0].isupper() and wrong[0].islower():
        wrong = wrong[0].upper() + wrong[1:]
    return sentence[:m.start()] + wrong + sentence[m.end():]


# ─── Master dispatch ─────────────────────────────────────────────

# Grammar corruption types with weights based on research
GRAMMAR_TYPES = [
    ("determiner", corrupt_determiner, 30),     # 10-16% of all errors
    ("preposition", corrupt_preposition, 20),    # 7-11%
    ("noun_number", corrupt_noun_number, 15),    # 3-8%
    ("verb_tense", corrupt_tense, 15),           # 6-7%
    ("agreement", corrupt_agreement, 8),         # 2-3%
    ("word_form", corrupt_word_form, 8),         # 3-5%
    ("contraction", corrupt_contraction, 4),     # 1-2%
]


def corrupt_grammar(sentence: str, rng: random.Random | None = None) -> str | None:
    """Apply a random grammar corruption weighted by real-world frequency.

    Returns None if no corruption is applicable.
    """
    rng = rng or random.Random()

    # Weighted random selection
    total = sum(w for _, _, w in GRAMMAR_TYPES)
    r = rng.random() * total
    cumulative = 0.0
    selected = []
    for name, fn, weight in GRAMMAR_TYPES:
        cumulative += weight
        if r <= cumulative:
            selected = [(name, fn)]
            break

    # Try selected type first, then fall back to others
    remaining = [(n, f) for n, f, _ in GRAMMAR_TYPES if (n, f) not in selected]
    rng.shuffle(remaining)
    all_types = selected + remaining

    for name, fn in all_types:
        try:
            result = fn(sentence, rng)
            if result is not None and result != sentence:
                return result
        except Exception:
            continue

    return None
