"""Proven bounds: exact search over every guess agrees with brute force, and bounds never pass a real total."""
import numpy as np
import pytest

from agents.split_features import SplitFeaturizer
from analysis.limits import NEVER, EvenSplit, ExactSearch, one_level
from test_search import brute_force, family


@pytest.fixture(scope="module")
def featurizer():
    return SplitFeaturizer()


@pytest.mark.parametrize("pattern", ["?ight", "?atch", "?ound", "s?ale", "?ill?", "?aste"])
def test_exact_matches_brute_force(featurizer, pattern):
    possible = family(featurizer, pattern)
    assert len(possible) >= 3
    assert ExactSearch(featurizer).solve(possible, np.inf) == brute_force(featurizer, possible)


def test_budget_gives_a_lower_bound_then_the_exact_value(featurizer):
    possible = family(featurizer, "?ight")
    search = ExactSearch(featurizer)
    exact = brute_force(featurizer, possible)
    assert exact - 1 <= search.solve(possible, exact - 1)     # can't beat exact - 1: says so, proves no more
    assert search.solve(possible, np.inf) == exact            # the remembered bound doesn't spoil the answer


def test_one_level_is_a_lower_bound_for_every_guess(featurizer):
    rng = np.random.default_rng(0)
    possible = np.sort(rng.choice(featurizer.n_answers, 9, replace=False))
    bounds, k = one_level(featurizer, possible, EvenSplit(featurizer))
    search = ExactSearch(featurizer)
    all_green = featurizer.n_patterns - 1
    for g in rng.choice(len(featurizer.pool), 40, replace=False):
        codes = featurizer.by_answer[possible, g]
        if bounds[g] == NEVER:
            assert len(np.unique(codes)) == 1 and g not in possible
            continue
        total = len(possible) + sum(search.solve(possible[codes == c], np.inf) for c in np.unique(codes)
                                    if c != all_green)
        assert bounds[g] <= total
        assert len(np.unique(codes)) - (g in possible) <= k    # nobody makes more groups than k


def test_root_bounds_bracket_a_real_total():
    featurizer = SplitFeaturizer("common10")
    known = 3290   # the search's total (agents/search.py, width 20): some strategy reaches it
    result = ExactSearch(featurizer).prove(known)
    assert result["lower"] <= result["upper"] <= known
    stopped = ExactSearch(featurizer).prove(known, seconds=0)    # out of time at once: still a valid bound
    assert stopped["one_level"] <= stopped["lower"] <= known


@pytest.mark.parametrize("name, n", [("common3", 150), ("common4", 60), ("wordle5", 40), ("common7", 12)])
def test_one_level_matches_the_definition(name, n):
    """Many words (counted per color pattern) or few (compared pairwise): the same sums as by hand."""
    featurizer = SplitFeaturizer(name)
    even = EvenSplit(featurizer)
    possible = np.sort(np.random.default_rng(2).choice(featurizer.n_answers, n, replace=False))
    bounds, k = one_level(featurizer, possible, even)
    codes = featurizer.by_answer[possible]                       # answers x guesses
    all_green = featurizer.n_patterns - 1
    groups = [np.unique(codes[:, g], return_counts=True) for g in range(len(featurizer.pool))]
    assert k == max(int((c != all_green).sum()) for c, _ in groups)
    f = even(min(k, n - 1))
    for g in np.random.default_rng(3).choice(len(featurizer.pool), 60, replace=False):
        c, sizes = groups[g]
        if len(c) == 1 and c[0] != all_green:
            assert bounds[g] == NEVER
        else:
            assert bounds[g] == n + sum(int(f[s]) for code, s in zip(c, sizes) if code != all_green)
