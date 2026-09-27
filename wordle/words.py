"""Word lists and the numpy encodings the agents use."""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANSWERS_PATH = DATA_DIR / "answers.txt"
ALLOWED_GUESSES_PATH = DATA_DIR / "allowed_guesses.txt"


def load_words(path=ANSWERS_PATH):
    """Load a list of 5-letter lowercase words, one per line."""
    words = Path(path).read_text().split()
    return [w.lower() for w in words if len(w) == 5 and w.isalpha()]


def load_valid_guesses():
    """Every word the game accepts as a guess: answers + extra allowed guesses."""
    words = set(load_words(ANSWERS_PATH))
    if ALLOWED_GUESSES_PATH.exists():
        words.update(load_words(ALLOWED_GUESSES_PATH))
    return words


def encode(words):
    """Encode words as numpy arrays.

    Returns (letters, counts):
      letters: N x 5 array of letter indices 0-25 (a=0 ... z=25)
      counts:  N x 26 array, how many times each letter appears in each word
    """
    letters = np.array([[ord(c) - ord("a") for c in w] for w in words], dtype=np.int64)
    counts = np.zeros((len(words), 26), dtype=np.int64)
    rows = np.arange(len(words))
    for pos in range(5):
        np.add.at(counts, (rows, letters[:, pos]), 1)
    return letters, counts
