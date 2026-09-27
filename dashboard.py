"""Watch the AI learn Wordle live in your browser, with Pause / Continue / Reset.

    python dashboard.py                        # 500 games at 10 games per second
    python dashboard.py --games 2000 --speed 50
    python dashboard.py --speed 0              # as fast as possible

When the run finishes, the trained AI is saved to models/agent.npz, so
`python play.py --watch` and `python evaluate.py` use it. Ctrl+C to quit.
"""
import argparse
import sys
import threading
import webbrowser

from agents import DEFAULT_MODEL
from live.server import make_server
from live.session import TrainingSession
from wordle.words import load_words


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--games", type=int, default=500, help="games to play before stopping")
    parser.add_argument("--speed", type=float, default=10, help="games per second (0 = max)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=None, help="fix the randomness to repeat a run")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="where to save the trained AI")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    session = TrainingSession(load_words(), target_games=args.games, games_per_second=args.speed,
                              seed=args.seed, model_path=args.model)
    try:
        server = make_server(session, port=args.port)
    except OSError:
        sys.exit(f"Port {args.port} is busy (is another dashboard running?). Try --port 8766.")

    threading.Thread(target=session.measure_baseline, daemon=True).start()
    threading.Thread(target=session.run_forever, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
