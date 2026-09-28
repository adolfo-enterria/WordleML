"""Search: exact within its candidates, never worse than look-ahead, and it plays what it computed."""
import numpy as np
import pytest

from agents.base import play_game
from agents.features import cheapest
from agents.lookahead import LookaheadAgent
from agents.planning_agent import PlanningAgent
from agents.search import SearchAgent, StrategyAgent
from agents.split_features import SplitFeaturizer
from wordle.game import score_guess


@pytest.fixture(scope="module")
def base():
    agent = PlanningAgent(seed=0, featurizer=SplitFeaturizer())
    agent.coef = np.array([1.0, 0.45, -0.02])  # a trained-looking belief
    return agent


def brute_force(featurizer, possible, memo=None):
    """Best total over EVERY valid guess, no bounds, no planner: the plain definition, for checking."""
    memo = {} if memo is None else memo
    n = len(possible)
    if n <= 2:
        return 2 * n - 1
    key = possible.tobytes()
    if key not in memo:
        all_green = featurizer.n_patterns - 1
        best = np.inf
        for column in np.unique(featurizer.by_answer[possible].T, axis=0):  # guesses that split alike count once
            codes = np.unique(column)
            if len(codes) == 1 and codes[0] != all_green:
                continue  # tells nothing
            total = n + sum(brute_force(featurizer, possible[column == c], memo) for c in codes if c != all_green)
            best = min(best, total)
        memo[key] = best
    return memo[key]


def family(featurizer, pattern):
    """Answers that fit a pattern like '?ight' (the hard look-alike families)."""
    return np.array([i for i, w in enumerate(featurizer.pool[:featurizer.n_answers])
                     if all(p in ("?", c) for p, c in zip(pattern, w))])


@pytest.mark.parametrize("pattern", ["?ight", "?atch", "?ound", "s?ale"])
def test_matches_brute_force_with_every_guess(base, pattern):
    possible = family(base.featurizer, pattern)
    assert 3 <= len(possible) <= 12
    search = SearchAgent(base, width=len(base.pool))
    assert search.solve(possible) == brute_force(base.featurizer, possible)


@pytest.mark.parametrize("secret", ["mound", "catch", "night", "vivid"])
def test_never_worse_than_lookahead_same_width(base, secret):
    history = [("reast", score_guess("reast", secret))]
    search, ahead = SearchAgent(base, width=5), LookaheadAgent(base, width=5)
    possible = search.featurizer.candidates(history)
    assert search.solve(possible) <= round(len(possible) * ahead.expected_total(history))


def test_total_matches_playing_it_out(base):
    search = SearchAgent(base, width=5)
    history = [("reast", score_guess("reast", "catch"))]
    possible = search.featurizer.candidates(history)
    played = 0
    for secret in [search.pool[i] for i in possible]:
        h = [("reast", score_guess("reast", secret))]
        while h[-1][0] != secret:
            guess = search.choose(h)
            h.append((guess, score_guess(guess, secret)))
        played += len(h) - 1
    assert played == search.solve(possible)


def test_saved_plan_replays_the_same_games(base, tmp_path):
    search = SearchAgent(base, width=3, root_width=2)
    data = search.save(tmp_path / "plan.json")
    assert data["total"] == search.total()
    replay = StrategyAgent.load(tmp_path / "plan.json")
    for secret in ["night", "catch", "vivid", "crate"]:
        assert [g for g, _ in play_game(replay, secret).history] == data["paths"][secret]


def test_cheapest_puts_the_planners_choice_first(base):
    possible = base.featurizer.candidates([("reast", score_guess("reast", "night"))])
    cost = base.expected_cost(possible)
    top = cheapest(cost, 8, base.featurizer.tie_rank)
    assert base.pool[top[0]] == base.frozen_copy().choose([("reast", score_guess("reast", "night"))])
    assert np.all(np.diff(cost[top]) >= -1e-9) and np.isfinite(cost[top]).all()
