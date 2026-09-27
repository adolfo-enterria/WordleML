"""A training run you can watch and control: pause, continue, reset, change speed.

The learning agent plays in a background thread, one game at a time. Each game
the secret is a word picked at random from the full answer list, so the agent
never has a clue before its first guess. The web dashboard reads results with
snapshot() and sends the button presses to pause()/resume()/reset().
"""
import threading
import time

import numpy as np

from agents import DEFAULT_MODEL
from agents.base import play_game
from agents.consistent_agent import ConsistentAgent
from agents.learning_agent import LearningAgent

RECENT_WINDOW = 50  # games in the "recent average" line and stats


class TrainingSession:
    def __init__(self, words, target_games=500, games_per_second=10, seed=None,
                 model_path=DEFAULT_MODEL):
        self.words = list(words)
        self.target_games = target_games
        self.games_per_second = games_per_second  # 0 means as fast as possible
        self.model_path = model_path
        self.seeds = np.random.SeedSequence(seed)
        self.lock = threading.Lock()
        self.status = "running"
        self.baseline = None  # no-strategy average, measured in the background
        self.run_id = 0
        self._start_new_run()

    def _start_new_run(self):
        """Fresh, untrained agent and empty results. Caller holds the lock (or is __init__)."""
        agent_seed, secret_seed = self.seeds.spawn(2)
        self.agent = LearningAgent(self.words, seed=agent_seed)
        self.agent.training = True
        self.secret_rng = np.random.default_rng(secret_seed)
        self.results = []  # one (secret, guesses) per game
        self.run_id += 1

    # --- controls -----------------------------------------------------------

    def pause(self):
        with self.lock:
            if self.status == "running":
                self.status = "paused"

    def resume(self):
        with self.lock:
            if self.status == "paused":
                self.status = "running"

    def reset(self):
        """Throw away everything learned and start again from game 1."""
        with self.lock:
            self._start_new_run()
            if self.status == "finished":
                self.status = "running"

    def set_speed(self, games_per_second):
        with self.lock:
            self.games_per_second = max(0, float(games_per_second))

    # --- playing ------------------------------------------------------------

    def play_one_game(self):
        """Play one training game if the run is active. Returns True if a game was played."""
        with self.lock:
            if self.status != "running":
                return False
            agent, rng, run_id = self.agent, self.secret_rng, self.run_id
            secret = self.words[rng.integers(len(self.words))]

        game = play_game(agent, secret)  # the slow part, done without holding the lock

        with self.lock:
            if run_id != self.run_id:  # reset while this game was being played: discard it
                return False
            self.results.append((secret, game.guesses_used))
            if len(self.results) >= self.target_games:
                self.status = "finished"
                agent.save(self.model_path)
        return True

    def run_forever(self):
        """The background loop: play games at the chosen speed while running."""
        while True:
            started = time.perf_counter()
            if not self.play_one_game():
                time.sleep(0.05)
                continue
            with self.lock:
                speed = self.games_per_second
            if speed > 0:
                time.sleep(max(0.0, 1 / speed - (time.perf_counter() - started)))

    def measure_baseline(self):
        """Average guesses with no strategy (a random still-possible word), over every answer."""
        agent = ConsistentAgent(self.words, seed=0)
        average = float(np.mean([play_game(agent, w).guesses_used for w in self.words]))
        with self.lock:
            self.baseline = average

    # --- reading ------------------------------------------------------------

    def snapshot(self, since=0):
        """Everything the dashboard needs, plus only the games after `since`."""
        with self.lock:
            guesses = np.array([g for _, g in self.results])
            played = len(guesses)
            first = guesses[:RECENT_WINDOW]
            recent = guesses[-RECENT_WINDOW:]
            return {
                "run_id": self.run_id,
                "status": self.status,
                "target_games": self.target_games,
                "games_per_second": self.games_per_second,
                "baseline": self.baseline,
                "window": RECENT_WINDOW,
                "games_played": played,
                "stats": {
                    "last_game": ({"secret": self.results[-1][0], "guesses": int(guesses[-1])}
                                  if played else None),
                    "first_avg": float(first.mean()) if played else None,
                    "first_count": len(first),
                    "recent_avg": float(recent.mean()) if played else None,
                    "recent_count": len(recent),
                },
                "since": since,
                "results": [[i + 1, s, g] for i, (s, g) in enumerate(self.results[since:], since)],
            }
