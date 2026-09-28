"""Every (guess, answer) color pattern, precomputed, for one word set.

For the official Wordle lists that's 12,972 guesses x 2,315 answers = 30 million
patterns. Computing them once and caching them on disk makes questions like
"how would this guess split the words that are still possible?" fast enough to
ask for every word, every turn.

A pattern is stored as one number: sum(color[i] * 3**i) with GREY=0, YELLOW=1,
GREEN=2. For L-letter words that's 0 .. 3**L - 1, and 3**L - 1 means all green.
Up to 5 letters that fits in a uint8; longer words use uint16 (3**8 = 6,561).
"""
from functools import lru_cache

import numpy as np

from wordle.game import GREEN, GREY, YELLOW
from wordle.words import DATA_DIR, DEFAULT_WORD_SET, encode, load_word_set

N_PATTERNS = 3 ** 5          # for the default 5-letter set
ALL_GREEN = N_PATTERNS - 1
CACHE_PATH = DATA_DIR / "patterns.npz"  # the default set's cache


def n_patterns(length):
    return 3 ** length


def cache_path(name):
    return CACHE_PATH if name == DEFAULT_WORD_SET else DATA_DIR / f"patterns_{name}.npz"


def guess_list(name=DEFAULT_WORD_SET):
    """Every valid guess, answers first (so answer i is guess i)."""
    return list(load_word_set(name).guesses)


def encode_pattern(feedback):
    return sum(color * 3 ** i for i, color in enumerate(feedback))


def decode_pattern(code, length=5):
    return tuple((int(code) // 3 ** i) % 3 for i in range(length))


def compute_patterns(guesses, answers):
    """The pattern for every (guess, answer) pair, with Wordle's duplicate-letter rule.

    Same rule as score_guess: greens first; then, left to right, the k-th
    non-green copy of a letter in the guess is yellow only if the answer has
    more than k copies of that letter outside its green spots.
    """
    guess_letters, _ = encode(guesses)
    answer_letters, _ = encode(answers)
    length = guess_letters.shape[1]
    dtype = np.uint8 if n_patterns(length) <= 256 else np.uint16
    table = np.empty((len(guesses), len(answers)), dtype=dtype)
    a = answer_letters[None, :, :]                              # 1 x N x L
    chunk = max(16, 12_000_000 // (len(answers) * length))     # guesses per block (bounded memory)
    for start in range(0, len(guesses), chunk):
        g = guess_letters[start:start + chunk]                  # B x L
        green = g[:, None, :] == a                              # B x N x L
        code = np.zeros(green.shape[:2], dtype=np.int32)
        for i in range(length):
            letter = g[:, i][:, None, None]                     # B x 1 x 1
            # copies of this letter in the answer that no green used up
            left_in_answer = ((a == letter) & ~green).sum(axis=2)
            # earlier non-green copies of this letter in the guess (they get first dibs)
            same_before = (g[:, :i] == g[:, i:i + 1])[:, None, :] & ~green[:, :, :i]
            earlier = same_before.sum(axis=2)
            yellow = ~green[:, :, i] & (earlier < left_in_answer)
            color = np.where(green[:, :, i], GREEN, np.where(yellow, YELLOW, GREY))
            code += color * 3 ** i
        table[start:start + chunk] = code
    return table


@lru_cache(maxsize=2)
def pattern_table(name=DEFAULT_WORD_SET):
    """(guesses, answers, table) for a word set; built on first use and cached on disk."""
    word_set = load_word_set(name)
    guesses, answers = list(word_set.guesses), list(word_set.answers)
    path = cache_path(name)
    if path.exists():
        data = np.load(path)
        if list(data["guesses"]) == guesses and list(data["answers"]) == answers:
            return guesses, answers, data["table"]
    table = compute_patterns(guesses, answers)
    np.savez(path, table=table, guesses=np.array(guesses), answers=np.array(answers))
    return guesses, answers, table
