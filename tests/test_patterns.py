import numpy as np

from wordle.game import score_guess
from wordle.patterns import (ALL_GREEN, compute_patterns, decode_pattern, encode_pattern,
                             guess_list, pattern_table)
from wordle.words import load_words

TRICKY = [("babes", "abbey"), ("kebab", "abbey"), ("eerie", "crane"), ("speed", "abide"),
          ("floor", "robot"), ("abide", "speed"), ("eerie", "keeps"), ("llama", "hello")]


def test_encode_decode_round_trip():
    for code in range(243):
        assert encode_pattern(decode_pattern(code)) == code


def test_duplicate_letter_cases_match_score_guess():
    guesses = [g for g, _ in TRICKY]
    answers = sorted({a for _, a in TRICKY})
    table = compute_patterns(guesses, answers)
    for i, (g, a) in enumerate(TRICKY):
        assert decode_pattern(table[i, answers.index(a)]) == score_guess(g, a)


def test_table_matches_score_guess_on_random_pairs():
    guesses, answers, table = pattern_table()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        g, a = rng.integers(len(guesses)), rng.integers(len(answers))
        assert decode_pattern(table[g, a]) == score_guess(guesses[g], answers[a])


def test_answers_come_first_in_the_guess_list():
    guesses, answers, table = pattern_table()
    assert guesses[:len(answers)] == answers == load_words()
    assert len(guesses) == len(set(guesses)) == len(guess_list())
    assert np.all(table[np.arange(len(answers)), np.arange(len(answers))] == ALL_GREEN)
