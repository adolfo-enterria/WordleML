import numpy as np
import pytest

from agents.base import play_game
from agents.lookahead import LookaheadAgent
from agents.planning_agent import PlanningAgent
from agents.split_features import SplitFeaturizer
from wordle.game import score_guess


@pytest.fixture(scope="module")
def base():
    agent = PlanningAgent(seed=0, featurizer=SplitFeaturizer())
    agent.coef = np.array([1.0, 0.45, -0.02])  # a trained-looking belief
    return agent


@pytest.mark.parametrize("secret", ["mound", "catch", "night"])
def test_never_worse_than_the_planner(base, secret):
    """From a mid-game position, the look-ahead's exact expected cost <= the planner's."""
    la = LookaheadAgent(base, width=5)
    history = [("reast", score_guess("reast", secret))]
    possible = la.featurizer.candidates(history)
    assert la.expected_total(history) <= la.base_value(possible) + 1e-9


def test_exact_value_matches_playing_it_out(base):
    la = LookaheadAgent(base, width=5)
    history = [("reast", score_guess("reast", "catch"))]
    words = [la.pool[i] for i in la.featurizer.candidates(history)]
    played = []
    for secret in words:  # continue from the same position for every possible answer
        h = [("reast", score_guess("reast", secret))]
        while h[-1][0] != secret:
            guess = la.choose(h)
            h.append((guess, score_guess(guess, secret)))
        played.append(len(h) - 1)
    assert np.mean(played) == pytest.approx(la.expected_total(history))  # guesses from this position on


def test_deterministic_and_solves(base):
    a, b = LookaheadAgent(base, width=5), LookaheadAgent(base, width=5)
    history = [("reast", score_guess("reast", "night"))]
    assert a.choose(history) == b.choose(history)
    game = play_game(a, "night")
    assert game.won
