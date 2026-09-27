DEFAULT_MODEL = "models/agent.npz"
AGENT_NAMES = ["learning", "consistent"]


def make_agent(name, words, model_path=DEFAULT_MODEL, seed=None):
    """Build an agent by name. The learning agent loads its trained weights."""
    if name == "consistent":
        from agents.consistent_agent import ConsistentAgent
        return ConsistentAgent(words, seed=seed)
    if name == "learning":
        from agents.learning_agent import LearningAgent
        return LearningAgent.load(model_path, words, seed=seed)
    raise ValueError(f"Unknown agent '{name}', choose from {AGENT_NAMES}")
