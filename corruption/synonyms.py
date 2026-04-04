"""Synonym and word-level swap rules for content word corruption.

Based on analysis of C4_200M's most common word-level rewrites.
All patterns are rule-based — no lookup against test data.

Categories:
  1. Common synonym pairs (give/provide, get/receive, make/create)
  2. Number words <-> digits (3/three, 2/two)
  3. Quantifier swaps (many/much, every/each/all)
  4. Conditional/temporal (when/if, while/although)
  5. Degree/comparison (more/further/better/greater)
  6. Register shifts (like/as, till/until, got/received)
"""

import random

# Bidirectional synonym groups — any word can become any other in its group
SYNONYM_GROUPS = [
    # Verbs of giving/providing
    ["give", "provide", "offer", "supply"],
    ["get", "receive", "obtain", "acquire"],
    ["make", "create", "build", "construct", "produce"],
    ["show", "demonstrate", "display", "reveal", "indicate"],
    ["help", "assist", "support", "aid"],
    ["start", "begin", "commence", "initiate"],
    ["end", "finish", "complete", "conclude"],
    ["use", "utilize", "employ", "apply"],
    ["buy", "purchase", "acquire"],
    ["sell", "market"],
    ["keep", "maintain", "retain", "preserve"],
    ["find", "discover", "locate", "identify"],
    ["watch", "see", "observe", "view"],
    ["think", "believe", "consider", "feel"],
    ["say", "state", "mention", "note", "indicate"],
    ["tell", "inform", "notify"],
    ["ask", "request", "inquire"],
    ["try", "attempt", "endeavor"],
    ["need", "require"],
    ["want", "desire", "wish"],
    ["change", "modify", "alter", "adjust"],
    ["grow", "increase", "expand", "rise"],
    ["reduce", "decrease", "lower", "cut", "diminish"],
    ["stop", "cease", "halt", "discontinue"],
    ["allow", "permit", "enable", "let"],
    ["prevent", "avoid", "stop"],
    ["choose", "select", "pick"],
    ["join", "connect", "link", "combine"],
    ["fix", "repair", "mend"],
    ["learn", "study", "discover"],
    ["teach", "instruct", "educate", "train"],
    ["send", "deliver", "transmit"],
    ["hold", "contain", "include"],
    ["lead", "guide", "direct"],
    ["follow", "pursue", "track"],
    ["move", "relocate", "transfer", "shift"],
    ["talk", "speak", "discuss", "converse"],
    ["happen", "occur", "take place"],
    ["suggest", "recommend", "propose"],

    # Adjectives
    ["big", "large", "huge", "enormous", "massive"],
    ["small", "little", "tiny", "minor"],
    ["good", "great", "excellent", "fine", "wonderful"],
    ["bad", "poor", "terrible", "awful"],
    ["important", "significant", "crucial", "essential", "vital", "key"],
    ["different", "various", "diverse"],
    ["hard", "difficult", "challenging", "tough"],
    ["easy", "simple", "straightforward"],
    ["fast", "quick", "rapid", "swift"],
    ["slow", "gradual"],
    ["old", "ancient", "former", "previous"],
    ["new", "modern", "recent", "latest", "current"],
    ["whole", "entire", "complete", "full"],
    ["main", "primary", "chief", "principal", "major"],
    ["right", "correct", "proper", "appropriate"],
    ["wrong", "incorrect", "improper"],
    ["sure", "certain", "confident"],
    ["happy", "glad", "pleased", "delighted"],
    ["sad", "unhappy", "upset"],
    ["rich", "wealthy", "affluent"],
    ["enough", "sufficient", "adequate"],
    ["clear", "obvious", "evident", "apparent"],

    # Nouns
    ["area", "region", "zone", "district"],
    ["place", "location", "site", "spot"],
    ["way", "method", "approach", "technique", "manner"],
    ["kind", "type", "sort", "category"],
    ["part", "section", "portion", "segment"],
    ["problem", "issue", "challenge", "difficulty"],
    ["answer", "response", "reply", "solution"],
    ["idea", "concept", "notion", "thought"],
    ["goal", "aim", "objective", "target"],
    ["result", "outcome", "consequence", "effect"],
    ["reason", "cause", "factor"],
    ["chance", "opportunity", "possibility"],
    ["job", "work", "position", "role", "occupation"],
    ["home", "house", "residence"],
    ["money", "funds", "cash", "capital"],
    ["power", "authority", "control", "influence"],
    ["price", "cost", "fee", "charge"],
    ["trip", "journey", "travel", "voyage"],
    ["center", "centre"],
    ["color", "colour"],
    ["favorite", "favourite"],

    # Adverbs/discourse
    ["also", "additionally", "furthermore", "moreover"],
    ["however", "nevertheless", "nonetheless", "yet"],
    ["therefore", "thus", "consequently", "hence"],
    ["especially", "particularly", "specifically"],
    ["usually", "typically", "generally", "normally"],
    ["often", "frequently", "regularly"],
    ["only", "just", "merely", "simply"],
    ["almost", "nearly", "practically", "virtually"],
    ["really", "truly", "actually", "genuinely"],
    ["quite", "rather", "fairly", "somewhat"],
]

# Quantifier swaps
QUANTIFIER_GROUPS = [
    ["many", "numerous", "several", "various"],
    ["much", "considerable", "substantial"],
    ["every", "each", "all"],
    ["some", "certain", "several", "a few"],
]

# Number word <-> digit
NUMBER_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "15": "fifteen", "20": "twenty",
    "100": "hundred", "1000": "thousand",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Register / formality shifts
REGISTER_SWAPS = {
    "like": "as", "as": "like",
    "till": "until", "until": "till",
    "got": "received", "received": "got",
    "kids": "children", "children": "kids",
    "buy": "purchase", "purchase": "buy",
    "big": "large", "large": "big",
    "small": "little", "little": "small",
    "begin": "start", "start": "begin",
    "finish": "end", "end": "finish",
    "maybe": "perhaps", "perhaps": "maybe",
    "enough": "sufficient", "sufficient": "enough",
    "about": "approximately", "approximately": "about",
}

# Conditional/temporal
CONDITIONAL_SWAPS = {
    "when": "if", "if": "when",
    "while": "although", "although": "while",
    "because": "since", "since": "because",
    "before": "until", "until": "before",
}


def _build_lookup(groups):
    """Build word -> alternatives lookup from synonym groups."""
    lookup = {}
    for group in groups:
        for word in group:
            lookup[word] = [w for w in group if w != word]
    return lookup

_SYNONYM_LOOKUP = _build_lookup(SYNONYM_GROUPS)
_QUANTIFIER_LOOKUP = _build_lookup(QUANTIFIER_GROUPS)


def synonym_swap(word: str, rng: random.Random) -> str | None:
    """Swap a word with a synonym/alternative.

    Returns None if no swap is available.
    """
    lower = word.lower().rstrip(".,;:!?")
    suffix = word[len(lower):]

    # Try each category
    alternatives = None

    if lower in _SYNONYM_LOOKUP:
        alternatives = _SYNONYM_LOOKUP[lower]
    elif lower in _QUANTIFIER_LOOKUP:
        alternatives = _QUANTIFIER_LOOKUP[lower]
    elif lower in NUMBER_WORDS:
        alternatives = [NUMBER_WORDS[lower]]
    elif lower in REGISTER_SWAPS:
        alternatives = [REGISTER_SWAPS[lower]]
    elif lower in CONDITIONAL_SWAPS:
        alternatives = [CONDITIONAL_SWAPS[lower]]

    if not alternatives:
        return None

    replacement = rng.choice(alternatives)

    # Preserve case
    if word[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    if word.isupper():
        replacement = replacement.upper()

    return replacement + suffix
