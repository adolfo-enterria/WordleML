import numpy as np
import pytest

from agents.features import FEATURE_NAMES, Featurizer
from agents.learning_agent import LearningAgent, standardize
from agents.base import play_game
from wordle.game import score_guess
from wordle.words import load_words

WORDS = load_words()


@pytest.fixture(scope="module")
def featurizer():
    return Featurizer(WORDS)


def test_first_turn_everything_is_possible(featurizer):
    candidates, facts = featurizer.features([])
    assert len(candidates) == len(WORDS)
    assert facts.shape == (len(WORDS), len(FEATURE_NAMES))
    distinct = dict(zip(WORDS, facts[:, FEATURE_NAMES.index("distinct_letters")]))
    assert distinct["crane"] == 5 and distinct["llama"] == 3


def test_possible_matches_the_rules(featurizer):
    """The agent's rule logic agrees with plain Wordle scoring on random boards."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        secret = WORDS[rng.integers(len(WORDS))]
        history = []
        for guess in rng.choice(WORDS, size=rng.integers(1, 5), replace=False):
            history.append((guess, score_guess(guess, secret)))
        possible = np.array([all(score_guess(g, w) == fb for g, fb in history) for w in WORDS])
        assert np.array_equal(featurizer.possible(history), possible)
        assert possible[WORDS.index(secret)]


def test_guessed_words_are_no_longer_possible(featurizer):
    history = [("crane", score_guess("crane", "abbey"))]
    candidates, _ = featurizer.features(history)
    assert WORDS.index("crane") not in candidates
    assert WORDS.index("abbey") in candidates


def test_duplicate_letters(featurizer):
    # Secret ABBEY, guess BABES (allowed guess, not an answer): yellow yellow green green grey.
    history = [("babes", score_guess("babes", "abbey"))]
    candidates, _ = featurizer.features(history)
    possible = {WORDS[i] for i in candidates}
    assert "abbey" in possible
    assert "baker" not in possible  # B and A back in the spots they were yellow
    assert "saute" not in possible  # S is grey


def test_standardize():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    z = standardize(x)
    assert np.allclose(z[:, 0], [-1, 1])
    assert np.all(z[:, 1] == 0)  # every word ties: no information, no preference


def test_untrained_agent_always_solves():
    agent = LearningAgent(WORDS, seed=0)
    agent.training = True
    for secret in ["crane", "nymph", "fuzzy", "mamma"]:
        game = play_game(agent, secret)
        assert game.won
        assert all(fb != (2, 2, 2, 2, 2) for _, fb in game.history[:-1])
