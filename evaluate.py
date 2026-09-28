"""Measure how well an agent plays: one game per answer word, no guess limit.

    python evaluate.py                       # the trained AI for the official Wordle list
    python evaluate.py --lookahead           # ...thinking further ahead each turn (slower, better)
    python evaluate.py --length 6            # the trained AI for 6-letter common words
    python evaluate.py --agent consistent    # no strategy: a random word that's still possible
"""
import argparse
import time

import numpy as np
from colorama import Back, Style, just_fix_windows_console

from agents import AGENT_NAMES, make_agent, model_path
from agents.base import play_game
from agents.lookahead import LookaheadAgent
from wordle.words import load_word_set, word_set_name

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
    parser.add_argument("--length", type=int, default=5, choices=range(3, 9), help="word length (3-8)")
    parser.add_argument("--common", action="store_true", help="common-words list even at 5 letters")
    parser.add_argument("--model", help="weights file for the learning agent (default: the one for the word list)")
    parser.add_argument("--lookahead", type=int, nargs="?", const=10, default=0, metavar="WIDTH",
                        help="think ahead over the top WIDTH guesses each turn (default 10)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    just_fix_windows_console()

    name = word_set_name(args.length, not args.common)
    words = list(load_word_set(name).answers)
    agent = make_agent(args.agent, words, args.model or model_path(name), seed=args.seed)
    label = f"{args.agent} on {load_word_set(name).label}"
    if args.lookahead:
        if getattr(agent, "name", "") != "planner":
            raise SystemExit("--lookahead needs a planner model (trained in \"Any valid word\" mode).")
        agent = LookaheadAgent(agent, width=args.lookahead)
        label += f", look-ahead width {args.lookahead}"
    started = time.perf_counter()
    print_report(label, evaluate(agent, words))
    print(f"\n({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
