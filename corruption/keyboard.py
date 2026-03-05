"""QWERTY keyboard adjacency map for fat-finger error simulation."""

# Each key maps to its physically adjacent keys on a US QWERTY layout.
ADJACENCY = {
    "q": ["w", "a", "s"],
    "w": ["q", "e", "a", "s", "d"],
    "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"],
    "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"],
    "u": ["y", "i", "h", "j", "k"],
    "i": ["u", "o", "j", "k", "l"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z", "x"],
    "s": ["q", "w", "e", "a", "d", "z", "x", "c"],
    "d": ["w", "e", "r", "s", "f", "x", "c", "v"],
    "f": ["e", "r", "t", "d", "g", "c", "v", "b"],
    "g": ["r", "t", "y", "f", "h", "v", "b", "n"],
    "h": ["t", "y", "u", "g", "j", "b", "n", "m"],
    "j": ["y", "u", "i", "h", "k", "n", "m"],
    "k": ["u", "i", "o", "j", "l", "m"],
    "l": ["i", "o", "p", "k"],
    "z": ["a", "s", "x"],
    "x": ["a", "s", "d", "z", "c"],
    "c": ["s", "d", "f", "x", "v"],
    "v": ["d", "f", "g", "c", "b"],
    "b": ["f", "g", "h", "v", "n"],
    "n": ["g", "h", "j", "b", "m"],
    "m": ["h", "j", "k", "n"],
}


def get_adjacent_key(char: str) -> str | None:
    """Return a random adjacent key for the given character.

    Preserves case of the original character.
    """
    import random

    lower = char.lower()
    if lower not in ADJACENCY:
        return None
    adj = random.choice(ADJACENCY[lower])
    return adj.upper() if char.isupper() else adj


def fat_finger_word(word: str, rng: "random.Random | None" = None) -> str:
    """Apply a single fat-finger substitution to a word."""
    import random

    rng = rng or random.Random()
    if len(word) < 2:
        return word

    # Pick a random position (avoid first char to preserve recognition)
    positions = [i for i in range(len(word)) if word[i].lower() in ADJACENCY]
    if not positions:
        return word

    pos = rng.choice(positions)
    replacement = get_adjacent_key(word[pos])
    if replacement is None:
        return word

    return word[:pos] + replacement + word[pos + 1 :]
