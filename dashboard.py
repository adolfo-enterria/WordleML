"""Watch the AI learn Wordle live in your browser, with Pause / Continue / Reset.

    python dashboard.py                        # 500 games at 10 games per second
    python dashboard.py --games 2000 --speed 50
    python dashboard.py --speed 0              # as fast as possible
    python dashboard.py --length 6             # 6-letter common words (the slider on the page does the same)

When the run finishes, the trained AI is saved (models/agent.npz for the official
list, models/agent_<list>.npz otherwise), so `python play.py --watch` and
`python evaluate.py` use it. Ctrl+C to quit.
"""
import argparse
import sys
import threading
import webbrowser

from live.server import make_server
from live.session import TrainingSession
from wordle.words import LENGTHS, word_set_name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--games", type=int, default=500, help="games to play before stopping")
    parser.add_argument("--speed", type=float, default=10, help="games per second (0 = max)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=None, help="fix the randomness to repeat a run")
    parser.add_argument("--mode", choices=["any", "possible"], default="any",
                        help="any: may guess any valid word (can probe); possible: only words that could win")
    parser.add_argument("--length", type=int, default=5, choices=LENGTHS, help="word length (3-10)")
    parser.add_argument("--common", action="store_true",
                        help="use the common-words list even at 5 letters (default: official Wordle list)")
    parser.add_argument("--model", default=None, help="where to save the trained AI (default: per word set)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    session = TrainingSession(word_set_name(args.length, not args.common), target_games=args.games,
                              games_per_second=args.speed, seed=args.seed, model_path=args.model, mode=args.mode)
    try:
        server = make_server(session, port=args.port)
    except OSError:
        sys.exit(f"Port {args.port} is busy (is another dashboard running?). Try --port 8766.")

    threading.Thread(target=session.run_forever, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
