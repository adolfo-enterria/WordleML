"""Play Wordle in the terminal, or watch an AI play.

    python play.py                          # you play a random word
    python play.py --secret crane           # you play a specific word
    python play.py --watch                  # watch the trained AI play
    python play.py --watch --secret crane   # ...on a specific word
    python play.py --watch --agent consistent --delay 0
    python play.py --length 6               # 6-letter words (you or --watch)
    python play.py --watch --lookahead      # watch the AI thinking further ahead
    python play.py --watch --search         # watch the searched plan (searches first if there's none saved)
"""
import argparse
import random
import sys
import time
from pathlib import Path

from agents import AGENT_NAMES, make_agent, model_path, searched_plan
from agents.lookahead import LookaheadAgent
from wordle.display import render_board, render_keyboard, render_row
from wordle.game import MAX_GUESSES, WordleGame
from wordle.words import LENGTHS, load_valid_guesses, load_word_set, word_set_name


def human_game(secret, valid_guesses):
    game = WordleGame(secret, valid_guesses=valid_guesses)
    print(f"Guess the {game.length}-letter word in {MAX_GUESSES} tries. Type 'quit' to give up.\n")
    while not game.over:
        print(render_board(game.history, length=game.length) + "\n\n" + render_keyboard(game.history) + "\n")
        try:
            guess = input(f"Guess {game.guesses_used + 1}/{MAX_GUESSES}: ").strip()
        except (EOFError, KeyboardInterrupt):
            guess = "quit"
        if guess.lower() in ("quit", "exit"):
            break
        try:
            game.guess(guess)
        except ValueError as error:
            print(f"  {error}\n")
    print(render_board(game.history, length=game.length) + "\n")
    if game.won:
        print(f"You got it in {game.guesses_used}!")
    else:
        print(f"The word was {secret.upper()}.")


def watch_game(agent, secret, delay, top):
    game = WordleGame(secret, max_guesses=None)  # the AI keeps going until it gets it
    agent.new_game()
    print(f"Secret word: {secret.upper()}  (the AI can't see it)\n")
    while not game.over:
        if hasattr(agent, "top_choices"):
            likes = agent.top_choices(game.history, top)
            if agent.name in ("planner", "lookahead", "search"):  # expected total guesses from here, lower is better
                print("   thinking: " + ", ".join(f"{w.upper()} ~{cost:.2f}" for w, cost in likes))
            else:                        # how likely it is to pick each word
                print("   thinking: " + ", ".join(f"{w.upper()} {p:.0%}" if p >= 0.01 else f"{w.upper()} <1%"
                                              for w, p in likes))
        guess = agent.choose(game.history)
        feedback = game.guess(guess)
        print(f"   {render_row(guess, feedback)}\n")
        time.sleep(delay)
    print(f"The {agent.name} agent solved it in {game.guesses_used}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="watch an AI play instead of playing yourself")
    parser.add_argument("--agent", choices=AGENT_NAMES, default="learning", help="which AI to watch")
    parser.add_argument("--model", help="weights file for the learning agent (default: the one for the word list)")
    parser.add_argument("--length", type=int, default=5, choices=LENGTHS, help="word length (3-10)")
    parser.add_argument("--common", action="store_true", help="common-words list even at 5 letters")
    parser.add_argument("--lookahead", action="store_true", help="watch the AI think further ahead (planner only)")
    parser.add_argument("--search", action="store_true", help="watch the searched plan (planner only)")
    parser.add_argument("--secret", help="the word to guess (default: random)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between AI guesses")
    parser.add_argument("--top", type=int, default=5, help="how many favorite words to show")
    args = parser.parse_args()

    name = word_set_name(args.length, not args.common)
    words = list(load_word_set(name).answers)
    secret = (args.secret or random.choice(words)).lower()

    if not args.watch:
        valid = load_valid_guesses(name)
        if secret not in valid:
            sys.exit(f"'{secret}' isn't a valid {len(secret)}-letter word for this list.")
        human_game(secret, valid)
        return

    if secret not in words:
        sys.exit(f"The AI only knows the {len(words):,} answer words, and '{secret}' isn't one.")
    path = args.model or model_path(name)
    if args.agent == "learning" and not Path(path).exists():
        sys.exit(f"No trained model at {path}. Train one with the dashboard (python dashboard.py --length {args.length}).")
    agent = make_agent(args.agent, words, path)
    if (args.lookahead or args.search) and getattr(agent, "name", "") != "planner":
        sys.exit('--lookahead and --search need a planner model (trained in "Any valid word" mode).')
    if args.search:
        agent, _ = searched_plan(agent, 20, name)
    elif args.lookahead:
        agent = LookaheadAgent(agent)
    watch_game(agent, secret, args.delay, args.top)


if __name__ == "__main__":
    main()
