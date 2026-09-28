"""The work the benchmark farms out to parallel worker processes.

Each worker loads the word lists and pattern table once, then runs tasks:
train one agent, or play a chunk of secret words with some strategy.
Everything is seeded, so the same settings always give the same numbers.

Strategy kinds:
  planner       planning agent (any valid word): learns V(m), plans one guess ahead
  pg-any        policy gradient (REINFORCE) on the 4 split facts, any valid word
  pg-possible   policy gradient on 3 word facts, only words that could be the answer
  greedy        hand-made: best split by a fixed rule ("entropy" or "avg_left")
  random        no strategy: a random word that could still be the answer
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # workers run side by side: one thread each

import time  # noqa: E402

import numpy as np  # noqa: E402

from agents.base import play_game  # noqa: E402
from agents.consistent_agent import ConsistentAgent  # noqa: E402
from agents.features import Featurizer  # noqa: E402
from agents.learning_agent import LearningAgent  # noqa: E402
from agents.planning_agent import PlanningAgent  # noqa: E402
from agents.lookahead import LookaheadAgent  # noqa: E402
from agents.split_features import SplitFeaturizer  # noqa: E402
from analysis.difficulty import TIE_SEED  # noqa: E402
from analysis.strategies import FactSubsetAgent, ForcedOpener, GreedySplitAgent  # noqa: E402
from wordle.patterns import pattern_table  # noqa: E402
from wordle.words import DEFAULT_WORD_SET, load_word_set  # noqa: E402

_word_set = None
_words = None
_featurizers = {}


def init_worker(word_set=DEFAULT_WORD_SET):
    use_words(word_set)


def use_words(word_set):
    """Point this worker at a word set (tasks for different word lengths can share one pool)."""
    global _word_set, _words
    if word_set != _word_set:
        _word_set, _words = word_set, list(load_word_set(word_set).answers)
        for key in [k for k in _featurizers if k[0] != word_set]:  # keep memory bounded
            del _featurizers[key]
        pattern_table.cache_clear()  # featurizers keep their own copy; don't hold old tables twice


def _featurizer(kind):
    key = (_word_set, "possible" if kind == "pg-possible" else "any")
    if key not in _featurizers:
        _featurizers[key] = Featurizer(_words) if key[1] == "possible" else SplitFeaturizer(_word_set)
    return _featurizers[key]


def make(kind, state=None, seed=TIE_SEED, options=None):
    """A learner of this kind; with `state` it gets those learned numbers."""
    options = options or {}
    if kind == "planner":
        agent = PlanningAgent(seed=seed, featurizer=_featurizer(kind),
                              forget=options.get("forget", 0.98), curve=options.get("curve", "curved"))
        if state is not None:
            agent.coef = np.array(state, float)
        return agent
    mode = "any" if kind == "pg-any" else "possible"
    if options.get("keep"):
        agent = FactSubsetAgent(_words, options["keep"], seed=seed, featurizer=_featurizer(kind), mode=mode)
    else:
        agent = LearningAgent(_words, seed=seed, featurizer=_featurizer(kind), mode=mode)
    if state is not None:
        agent.weights = np.array(state, float)
    return agent


def learned_state(agent):
    return (agent.coef if isinstance(agent, PlanningAgent) else agent.weights).tolist()


def build_strategy(spec):
    kind = spec["kind"]
    if kind == "greedy":
        agent = GreedySplitAgent(_featurizer("any"), spec["criterion"])
    elif kind == "random":
        agent = ConsistentAgent(_words, seed=spec["seed"])
    else:
        agent = make(kind, spec.get("state"), options=spec.get("options"))
    if spec.get("opener"):
        agent = ForcedOpener(agent, spec["opener"])
    return agent


def _turn_records(featurizer, game):
    """Per real decision: (words possible before the guess, was it a probe)."""
    records = []
    for turn, (guess, _) in enumerate(game.history):
        possible = featurizer.candidates(game.history[:turn])
        if len(possible) == 1:
            continue
        records.append((len(possible), guess not in {featurizer.pool[i] for i in possible}))
    return records


def evaluate_chunk(spec, secrets, record_turns=False):
    """Play each secret once. Learners use their best guesses with fixed tie-breaks."""
    use_words(spec.get("word_set", _word_set))
    agent = build_strategy(spec)
    inner = agent.agent if isinstance(agent, ForcedOpener) else agent
    rows = []
    for secret in secrets:
        if spec["kind"] in ("planner", "pg-any", "pg-possible"):
            inner.training = False
            inner.rng = np.random.default_rng(TIE_SEED)
        game = play_game(agent, secret)
        row = {"secret": secret, "guesses": game.guesses_used, "history": [g for g, _ in game.history]}
        if record_turns:
            row["turns"] = _turn_records(_featurizer(spec["kind"]), game)
        rows.append(row)
    return rows


def lookahead_exam(state, width, secrets=None, word_set=None):
    """The planner with these learned numbers, thinking `width` guesses ahead each turn, on every secret."""
    use_words(word_set or _word_set)
    started = time.time()
    agent = LookaheadAgent(make("planner", state), width=width)
    rows = [{"secret": s, "guesses": play_game(agent, s).guesses_used} for s in (secrets or _words)]
    return {"rows": rows, "opener": agent.choose([]), "seconds": time.time() - started}


def train(kind, seed, games, check_words, marks, options=None, word_set=None):
    """Train one learner from scratch; skill checks (best guesses on check_words) at the given marks."""
    use_words(word_set or _word_set)
    started = time.time()
    agent = make(kind, seed=seed, options=options)
    agent.training = True
    secrets = np.random.default_rng(seed + 10_000)

    def skill():
        frozen = make(kind, learned_state(agent), options=options)
        return float(np.mean([play_game(frozen, w).guesses_used for w in check_words]))

    curve = [(0, skill())] if 0 in marks else []
    practice = []
    for game_number in range(1, games + 1):
        practice.append(play_game(agent, _words[secrets.integers(len(_words))]).guesses_used)
        if game_number in marks:
            curve.append((game_number, skill()))
    return {"kind": kind, "seed": seed, "options": options or {}, "state": learned_state(agent),
            "curve": curve, "practice": practice, "seconds": time.time() - started}


def top_openers(kind, state, options=None, k=10, word_set=None):
    """The k first guesses the learner likes most."""
    use_words(word_set or _word_set)
    return [word for word, _ in make(kind, state, options=options).top_choices([], k)]
