from collections import Counter

import numpy as np
import pytest

import agents.split_features as split_features
from agents.base import play_game
from agents.planning_agent import PlanningAgent
from agents.split_features import SplitFeaturizer
from wordle.game import score_guess
from wordle.words import load_words

WORDS = load_words()


@pytest.fixture(scope="module")
def featurizer():
    return SplitFeaturizer()


def sensible(featurizer):
    """A planner with a reasonable belief: V(m) = 1 + 0.6*log2(m)."""
    agent = PlanningAgent(seed=0, featurizer=featurizer)
    agent.coef = np.array([1.0, 0.6, 0.0])
    return agent


def brute_force_cost(agent, possible_words, guess):
    v = agent.value_table()
    groups = Counter(score_guess(guess, answer) for answer in possible_words)
    total = sum(size * v[size] for pattern, size in groups.items() if pattern != (2, 2, 2, 2, 2))
    return total / len(possible_words)


@pytest.mark.parametrize("block_cells", [20_000, 10 ** 8])  # many small blocks of guesses, or one big one
def test_expected_cost_matches_brute_force(featurizer, monkeypatch, block_cells):
    monkeypatch.setattr(split_features, "BLOCK_CELLS", block_cells)
    agent = sensible(featurizer)
    history = [("stare", score_guess("stare", "nymph"))]
    possible = featurizer.candidates(history)
    possible_words = [featurizer.pool[i] for i in possible]
    cost = agent.expected_cost(possible)
    for guess in ["nymph", "lymph", "cloud", "zzzzz" if "zzzzz" in featurizer.index else "fjord"]:
        assert cost[featurizer.index[guess]] == pytest.approx(brute_force_cost(agent, possible_words, guess))


def test_goes_for_the_win_with_two_left(featurizer):
    agent = sensible(featurizer)
    history = [("stare", score_guess("stare", "abbey")), ("could", score_guess("could", "abbey"))]
    possible = [featurizer.pool[i] for i in featurizer.candidates(history)]
    while len(possible) > 2:  # narrow it to exactly two for the test
        history.append((possible[0], score_guess(possible[0], "abbey")))
        possible = [featurizer.pool[i] for i in featurizer.candidates(history)]
    if len(possible) == 2:
        assert agent.choose(history) in possible


def test_probes_the_ight_family(featurizer):
    agent = sensible(featurizer)
    history = [("light", score_guess("light", "night"))]
    possible = [featurizer.pool[i] for i in featurizer.candidates(history)]
    assert len(possible) >= 6 and all(w.endswith("ight") for w in possible)
    guess = agent.choose(history)
    assert guess not in possible  # a probe: can't win, but tests several first letters at once
    tested = {c for c in guess if any(w[0] == c for w in possible)}
    assert len(tested) >= 3


def test_blank_agent_plays_random_valid_words_and_learns(featurizer):
    agent = PlanningAgent(seed=1, featurizer=featurizer)
    assert not agent.coef.any()
    agent.training = True
    for secret in ["crane", "nymph", "fuzzy"]:
        assert play_game(agent, secret).won
    v = dict(agent.value_curve())
    assert v[1] < v[20] < v[1000]  # learned: more words left means more guesses to go


def test_tested_play_is_deterministic(featurizer):
    """Same beliefs + same word = same game, whatever the seed or the order words are played in."""
    agent = sensible(featurizer)
    a, b = agent.frozen_copy(seed=1), agent.frozen_copy(seed=99)
    words = ["nymph", "foyer", "catch", "night"]
    first = [[g for g, _ in play_game(a, w).history] for w in words]
    second = [[g for g, _ in play_game(b, w).history] for w in reversed(words)]
    assert first == list(reversed(second))


def test_frozen_copy_does_not_learn(featurizer, tmp_path):
    agent = sensible(featurizer)
    frozen = agent.frozen_copy()
    play_game(frozen, "crane")
    assert np.array_equal(frozen.coef, agent.coef)
    agent.save(tmp_path / "planner.npz")
    loaded = PlanningAgent.load(tmp_path / "planner.npz")
    assert np.array_equal(loaded.coef, agent.coef)
