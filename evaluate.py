"""Measure how well an agent plays: one game per answer word, no guess limit.

    python evaluate.py                       # the trained learning agent (models/agent.npz)
    python evaluate.py --agent consistent    # no strategy: a random word that's still possible
"""
import argparse

import numpy as np
from colorama import Back, Style, just_fix_windows_console

from agents import AGENT_NAMES, DEFAULT_MODEL, make_agent
from agents.base import play_game
from wordle.words import load_words

HISTOGRAM_BUCKETS = ["1", "2", "3", "4", "5", "6", "7+"]


def evaluate(agent, secrets):
    """Play one game per secret word. Returns average/worst guesses and a histogram."""
    guesses = np.array([play_game(agent, secret).guesses_used for secret in secrets])
    histogram = {label: int(np.sum(guesses == n)) for n, label in enumerate(HISTOGRAM_BUCKETS[:-1], 1)}
    histogram["7+"] = int(np.sum(guesses >= 7))
    return {
        "games": len(guesses),
        "avg_guesses": guesses.mean(),
        "worst": int(guesses.max()),
        "histogram": histogram,
    }


def print_report(name, stats, width=40):
    print(f"Agent: {name}   games: {stats['games']:,} (every answer word once)")
    print(f"Average guesses: {stats['avg_guesses']:.3f}   worst game: {stats['worst']} guesses\n")
    biggest = max(stats["histogram"].values()) or 1
    for label, count in stats["histogram"].items():
        bar = " " * round(width * count / biggest)
        print(f"  {label:>2} | {Back.GREEN}{bar}{Style.RESET_ALL} {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--agent", choices=AGENT_NAMES, default="learning")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="weights file for the learning agent")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    just_fix_windows_console()

    words = load_words()
    agent = make_agent(args.agent, words, args.model, seed=args.seed)
    print_report(args.agent, evaluate(agent, words))


if __name__ == "__main__":
    main()
