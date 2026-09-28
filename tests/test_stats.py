import numpy as np
import pytest

from agents.base import play_game
from agents.split_features import SplitFeaturizer
from analysis.stats import games_to_tell_apart, mean_ci, ols, paired_bootstrap
from analysis.strategies import ForcedOpener, GreedySplitAgent


def test_mean_ci():
    mean, ci = mean_ci([3.4, 3.5, 3.6])
    assert mean == pytest.approx(3.5)
    assert ci == pytest.approx(4.303 * 0.1 / np.sqrt(3))  # t(2) = 4.303


def test_paired_bootstrap_contains_the_true_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(3.5, 1, 2000)
    b = a + 0.1 + rng.normal(0, 0.5, 2000)  # truly 0.1 worse on average, with noise
    diff, lo, hi = paired_bootstrap(a, b)
    assert lo <= -0.1 <= hi and hi - lo < 0.1


def test_games_to_tell_apart():
    # 2 * (1.96 + 0.84)^2 * 1^2 / 0.1^2 = 1568
    assert games_to_tell_apart(1.0, 0.1) == 1568


def test_ols_recovers_a_line():
    x = np.linspace(0, 10, 50)[:, None]
    rows, r2 = ols(x, 2 + 3 * x[:, 0], ["x"])
    assert rows[1][1] == pytest.approx(3) and r2 == pytest.approx(1)


@pytest.fixture(scope="module")
def featurizer():
    return SplitFeaturizer()


def test_greedy_heuristic_and_forced_opener(featurizer):
    greedy = GreedySplitAgent(featurizer, "entropy")
    game = play_game(greedy, "nymph")
    assert game.won and game.guesses_used <= 5
    forced = play_game(ForcedOpener(greedy, "salet"), "nymph")
    assert forced.history[0][0] == "salet" and forced.won
