"""Every (guess, answer) color pattern, precomputed.

12,972 valid guesses x 2,315 possible answers = 30 million patterns. Computing
them once and caching them on disk makes questions like "how would this guess
split the words that are still possible?" fast enough to ask for every word,
every turn.

A pattern is stored as one number: sum(color[i] * 3**i) with GREY=0,
YELLOW=1, GREEN=2, so 0..242 and 242 means all green.
"""
from functools import lru_cache

import numpy as np

from wordle.game import GREEN, GREY, YELLOW
from wordle.words import ALLOWED_GUESSES_PATH, ANSWERS_PATH, DATA_DIR, encode, load_words

N_PATTERNS = 3 ** 5
ALL_GREEN = N_PATTERNS - 1
CACHE_PATH = DATA_DIR / "patterns.npz"


def guess_list():
    """Every valid guess: the 2,315 answers first (so answer i is guess i), then the rest."""
    answers = load_words(ANSWERS_PATH)
    extra = sorted(set(load_words(ALLOWED_GUESSES_PATH)) - set(answers))
    return answers + extra


def encode_pattern(feedback):
    return sum(color * 3 ** i for i, color in enumerate(feedback))


def decode_pattern(code):
    return tuple((code // 3 ** i) % 3 for i in range(5))


def compute_patterns(guesses, answers, chunk=400):
    """The pattern for every (guess, answer) pair, with Wordle's duplicate-letter rule.

    Same rule as score_guess: greens first; then, left to right, the k-th
    non-green copy of a letter in the guess is yellow only if the answer has
    more than k copies of that letter outside its green spots.
    """
    guess_letters, _ = encode(guesses)
    answer_letters, _ = encode(answers)
    table = np.empty((len(guesses), len(answers)), dtype=np.uint8)
    a = answer_letters[None, :, :]                              # 1 x N x 5
    for start in range(0, len(guesses), chunk):
        g = guess_letters[start:start + chunk]                  # B x 5
        green = g[:, None, :] == a                              # B x N x 5
        code = np.zeros(green.shape[:2], dtype=np.int32)
        for i in range(5):
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


@lru_cache(maxsize=1)
def pattern_table():
    """(guesses, answers, table), built on first use and cached in data/patterns.npz."""
    guesses, answers = guess_list(), load_words(ANSWERS_PATH)
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        if list(data["guesses"]) == guesses and list(data["answers"]) == answers:
            return guesses, answers, data["table"]
    table = compute_patterns(guesses, answers)
    np.savez(CACHE_PATH, table=table, guesses=np.array(guesses), answers=np.array(answers))
    return guesses, answers, table
