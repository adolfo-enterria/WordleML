from wordle.words import DEFAULT_WORD_SET

DEFAULT_MODEL = "models/agent.npz"
AGENT_NAMES = ["learning", "consistent"]
MODES = ("any", "possible")


def model_path(word_set=DEFAULT_WORD_SET):
    """Where the dashboard saves (and play/evaluate load) the trained AI for a word set."""
    return DEFAULT_MODEL if word_set == DEFAULT_WORD_SET else f"models/agent_{word_set}.npz"


def plan_path(word_set=DEFAULT_WORD_SET):
    """Where a searched plan (every answer's guesses, see agents/search.py) is saved for a word set."""
    return f"models/search_{word_set}.json"


def searched_plan(planner, width, word_set=DEFAULT_WORD_SET, log=print):
    """Search with this planner as the guide, save the plan, and return (agent, saved data).

    A saved plan with at least this width is reused (a search is deterministic, and it takes a while)."""
    import json
    import os
    import time
    from agents.search import SearchAgent, StrategyAgent
    path = plan_path(word_set)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("width", 0) >= width and info.get("coef") == [float(c) for c in planner.coef]:
            log(f"Using the saved plan in {path} (width {info['width']}).")
            return StrategyAgent(info.pop("paths"), info), info
    started = time.perf_counter()

    def progress(done, total, best):
        log(f"  [{time.perf_counter() - started:5.0f} s] opener {done}/{total}, best total so far "
            f"{'-' if best is None else f'{best:,}'}")

    search = SearchAgent(planner, width=width, progress=progress)
    log(f"Searching (width {width}): the planner's favourite guesses at every position, worked out exactly...")
    search.total()
    data = search.save(path, coef=[float(c) for c in planner.coef])
    log(f"Saved the plan to {path}.")
    return StrategyAgent(data.pop("paths"), data), data


def make_learner(mode, words=None, seed=None, featurizer=None, word_set=DEFAULT_WORD_SET):
    """A fresh, untrained learning agent for a mode and word set.

    "any":      may guess any valid word. The planning agent: learns how costly
                each situation is and plans one guess ahead (planning_agent.py).
    "possible": only guesses words that could be the answer. A policy trained
                by REINFORCE on 3 word facts (learning_agent.py).
    """
    if mode == "any":
        from agents.planning_agent import PlanningAgent
        return PlanningAgent(words, seed=seed, featurizer=featurizer, word_set=word_set)
    if mode == "possible":
        from agents.learning_agent import LearningAgent
        return LearningAgent(words, seed=seed, featurizer=featurizer, mode="possible", word_set=word_set)
    raise ValueError(f"mode must be one of {MODES}")


def load_learner(path, words=None, seed=None):
    """Load a saved learning agent of whichever kind it is (its word set is stored in the file)."""
    import numpy as np
    data = np.load(path)
    kind = str(data["kind"]) if "kind" in data else "policy"
    if kind == "planner":
        from agents.planning_agent import PlanningAgent
        return PlanningAgent.load(path, seed=seed)
    from agents.learning_agent import LearningAgent
    return LearningAgent.load(path, words, seed=seed)


def make_agent(name, words, model_path=DEFAULT_MODEL, seed=None):
    """Build an agent by name. "learning" loads the trained model from model_path."""
    if name == "consistent":
        from agents.consistent_agent import ConsistentAgent
        return ConsistentAgent(words, seed=seed)
    if name == "learning":
        agent = load_learner(model_path, words, seed=seed)
        agent.training = False
        return agent
    raise ValueError(f"Unknown agent '{name}', choose from {AGENT_NAMES}")
