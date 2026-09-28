"""Word lists ("word sets") and the numpy encodings the agents use.

A word set is a list of possible answers plus the words you may guess:
  wordle5                 the official Wordle lists (2,315 answers, 12,972 guesses); the default
  common3 ... common10    common English words of that length, built by `python -m wordle.wordlists`
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANSWERS_PATH = DATA_DIR / "answers.txt"
ALLOWED_GUESSES_PATH = DATA_DIR / "allowed_guesses.txt"
LISTS_DIR = DATA_DIR / "lists"
DEFAULT_WORD_SET = "wordle5"
LENGTHS = range(3, 11)


@dataclass(frozen=True)
class WordSet:
    name: str
    length: int
    answers: tuple   # the possible secret words
    guesses: tuple   # every word you may guess, answers first (so answer i is guess i)

    @property
    def official(self):
        return self.name == "wordle5"

    @property
    def label(self):
        return "Official Wordle (5 letters)" if self.official else f"Common {self.length}-letter words"


def word_set_name(length=5, official=True):
    return "wordle5" if official and length == 5 else f"common{length}"


def load_words(path=ANSWERS_PATH, length=5):
    """Load a list of lowercase words of one length, one per line."""
    words = Path(path).read_text().split()
    return [w.lower() for w in words if len(w) == length and w.isalpha()]


@lru_cache(maxsize=None)
def load_word_set(name=DEFAULT_WORD_SET):
    if name == "wordle5":
        answers = load_words(ANSWERS_PATH)
        extra = sorted(set(load_words(ALLOWED_GUESSES_PATH)) - set(answers))
        return WordSet(name, 5, tuple(answers), tuple(answers + extra))
    if not name.startswith("common") or not name[6:].isdigit() or int(name[6:]) not in LENGTHS:
        raise ValueError(f"unknown word set '{name}'")
    length = int(name[6:])
    answers_path = LISTS_DIR / f"{name}_answers.txt"
    if not answers_path.exists():
        raise FileNotFoundError(f"{answers_path} is missing: run `python -m wordle.wordlists`")
    answers = load_words(answers_path, length)
    guesses = load_words(LISTS_DIR / f"{name}_guesses.txt", length)
    extra = [w for w in guesses if w not in set(answers)]
    return WordSet(name, length, tuple(answers), tuple(answers + extra))


def load_valid_guesses(name=DEFAULT_WORD_SET):
    """Every word the game accepts as a guess."""
    return set(load_word_set(name).guesses)


def encode(words):
    """Encode same-length words as numpy arrays.

    Returns (letters, counts):
      letters: N x L array of letter indices 0-25 (a=0 ... z=25)
      counts:  N x 26 array, how many times each letter appears in each word
    """
    letters = np.array([[ord(c) - ord("a") for c in w] for w in words], dtype=np.int64)
    counts = np.zeros((len(words), 26), dtype=np.int64)
    rows = np.arange(len(words))
    for pos in range(letters.shape[1]):
        np.add.at(counts, (rows, letters[:, pos]), 1)
    return letters, counts
