"""Other word lengths: the same rules, checked against plain score_guess."""
from collections import Counter

import numpy as np
import pytest

from agents.base import play_game
from agents.features import Featurizer
from agents.planning_agent import PlanningAgent
from agents.split_features import SplitFeaturizer
from wordle.game import WordleGame, score_guess
from wordle.patterns import decode_pattern, pattern_table
from wordle.wordlists import is_simple_plural
from wordle.words import load_word_set, word_set_name


@pytest.mark.parametrize("length", [3, 4, 6, 7, 8])
def test_word_sets_load(length):
    ws = load_word_set(word_set_name(length, official=False))
    assert ws.length == length and not ws.official
    assert all(len(w) == length for w in ws.guesses)
    assert list(ws.guesses[:len(ws.answers)]) == list(ws.answers)
    assert len(set(ws.guesses)) == len(ws.guesses)
    assert load_word_set(word_set_name(5)).official


def test_score_guess_any_length():
    G, Y, X = 2, 1, 0
    assert score_guess("banana", "bandit") == (G, G, G, X, X, X)
    assert score_guess("eel", "lee") == (Y, G, Y)
    assert score_guess("aaabbb", "bbbaaa") == (Y, Y, Y, Y, Y, Y)
    game = WordleGame("planets", max_guesses=None)
    with pytest.raises(ValueError):
        game.guess("plane")
    game.guess("planets")
    assert game.won


@pytest.mark.parametrize("length", [3, 6, 8])
def test_pattern_table_matches_score_guess(length):
    guesses, answers, table = pattern_table(word_set_name(length, official=False))
    rng = np.random.default_rng(length)
    for _ in range(3000):
        g, a = rng.integers(len(guesses)), rng.integers(len(answers))
        assert decode_pattern(table[g, a], length) == score_guess(guesses[g], answers[a])


@pytest.mark.parametrize("length", [3, 6, 8])
def test_rule_logic_matches_score_guess(length):
    """Both ways of tracking "still possible" agree with plain Wordle scoring."""
    name = word_set_name(length, official=False)
    words = list(load_word_set(name).answers)
    possible_mode, any_mode = Featurizer(words), SplitFeaturizer(name)
    rng = np.random.default_rng(0)
    for _ in range(15):
        secret = words[rng.integers(len(words))]
        history = [(g, score_guess(g, secret)) for g in rng.choice(words, size=rng.integers(1, 4), replace=False)]
        truth = np.flatnonzero([all(score_guess(g, w) == fb for g, fb in history) for w in words])
        assert np.array_equal(possible_mode.candidates(history), truth)
        assert np.array_equal(any_mode.candidates(history), truth)


def test_planner_cost_matches_brute_force_six_letters():
    featurizer = SplitFeaturizer("common6")
    agent = PlanningAgent(seed=0, featurizer=featurizer)
    agent.coef = np.array([1.0, 0.6, 0.0])
    history = [("stares", score_guess("stares", "planet"))]
    possible = featurizer.candidates(history)
    words = [featurizer.pool[i] for i in possible]
    cost = agent.expected_cost(possible)
    v = agent.value_table()
    for guess in [words[0], featurizer.pool[-1], "planet", "caning"]:
        groups = Counter(score_guess(guess, answer) for answer in words)
        if len(groups) == 1 and guess not in words:  # tells it nothing: never an option
            assert cost[featurizer.index[guess]] == np.inf
            continue
        expected = sum(n * v[n] for p, n in groups.items() if p != (2,) * 6) / len(words)
        assert cost[featurizer.index[guess]] == pytest.approx(expected)
    assert play_game(agent.frozen_copy(), "planet").won


@pytest.mark.parametrize("name", ["common3", "wordle5", "common8"])
def test_group_sizes_same_for_few_and_many_words(name):
    """Few words left are compared pairwise, many are counted over all patterns: same sizes either way."""
    featurizer = SplitFeaturizer(name)
    rng = np.random.default_rng(1)
    for n in [1, 2, 7, 30, 120]:  # both sides of the switch-over for every length
        possible = np.sort(rng.choice(featurizer.n_answers, n, replace=False))
        sizes = np.concatenate([s for _, _, s in featurizer.group_size_blocks(possible)], axis=1)
        counted = np.concatenate([c[cells] for _, _, c, cells in featurizer._blocks(possible)], axis=1)
        assert np.array_equal(sizes, counted)
        j = rng.integers(len(featurizer.pool))  # and against the words themselves, for one guess
        codes = featurizer.by_answer[possible, j]
        assert list(sizes[:, j]) == [np.sum(codes == c) for c in codes]


def test_simple_plurals():
    dictionary = {"cat", "cats", "box", "boxes", "city", "cities", "glass", "news"}
    assert is_simple_plural("cats", dictionary) and is_simple_plural("boxes", dictionary)
    assert is_simple_plural("cities", dictionary)
    assert not is_simple_plural("glass", dictionary)  # -ss isn't a plural
    assert not is_simple_plural("news", dictionary)   # "new" isn't in this dictionary
