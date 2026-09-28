DEFAULT_MODEL = "models/agent.npz"
AGENT_NAMES = ["learning", "consistent"]
MODES = ("any", "possible")


def make_learner(mode, words, seed=None, featurizer=None):
    """A fresh, untrained learning agent for a mode.

    "any":      may guess any valid word. The planning agent: learns how costly
                each situation is and plans one guess ahead (planning_agent.py).
    "possible": only guesses words that could be the answer. A policy trained
                by REINFORCE on 3 word facts (learning_agent.py).
    """
    if mode == "any":
        from agents.planning_agent import PlanningAgent
        return PlanningAgent(words, seed=seed, featurizer=featurizer)
    if mode == "possible":
        from agents.learning_agent import LearningAgent
        return LearningAgent(words, seed=seed, featurizer=featurizer, mode="possible")
    raise ValueError(f"mode must be one of {MODES}")


def load_learner(path, words, seed=None):
    """Load a saved learning agent of whichever kind it is."""
    import numpy as np
    data = np.load(path)
    kind = str(data["kind"]) if "kind" in data else "policy"
    if kind == "planner":
        from agents.planning_agent import PlanningAgent
        return PlanningAgent.load(path, words, seed=seed)
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
