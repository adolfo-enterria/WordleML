"""A training run you can watch and control: pause, continue, reset, speed, length.

The learning agent plays practice games in a background thread. Each game the
secret is a word picked at random from the full answer list, so the agent never
has a clue before its first guess.

Every so often training stops for a "skill check" on the same fixed set of
words, measured two ways (same words every time, so the luck of which word
comes up can't move either number):
  - best: always its highest-scoring word. Shows what it knows. Only the
    direction of the weights matters here, so this drops within a few games.
  - practice: choosing the way it does in practice, still exploring, with a
    fixed random seed. Shows how confidently it uses what it knows; this keeps
    improving as the weights grow and it explores less.
The first check is at game 0, before any training.

The web dashboard reads everything with snapshot() and sends button presses to
the control methods.
"""
import threading
import time

import numpy as np

from agents import DEFAULT_MODEL
from agents.base import play_game
from agents.features import FEATURE_NAMES, Featurizer
from agents.learning_agent import LearningAgent
from analysis.difficulty import analyze, trace_game, word_traits

RECENT_WINDOW = 50        # practice games in the rolling average
EARLY_GAMES = 50          # most of the learning happens here, so check more often
EARLY_INTERVAL = 5
SKILL_CHECK_WORDS = 200   # same words at every skill check
SKILL_CHECK_SEED = 12345  # picks those words and breaks ties the same way every time


class _NoLearning(LearningAgent):
    """Plays exactly like the training agent (exploring included) but never updates itself."""

    def end_game(self, game):
        pass


class TrainingSession:
    def __init__(self, words, target_games=500, games_per_second=10, seed=None,
                 model_path=DEFAULT_MODEL):
        self.words = list(words)
        self.featurizer = Featurizer(self.words)
        self.traits = word_traits(self.words)
        rng = np.random.default_rng(SKILL_CHECK_SEED)
        self.check_words = sorted(str(w) for w in rng.choice(self.words, SKILL_CHECK_WORDS, replace=False))
        self.target_games = target_games
        self.games_per_second = games_per_second  # 0 means as fast as possible
        self.model_path = model_path
        self.seeds = np.random.SeedSequence(seed)
        self.lock = threading.Lock()
        self.status = "running"
        self.run_id = 0
        self._start_new_run()

    def _start_new_run(self):
        """Fresh, untrained agent (all weights zero) and no results. Caller holds the lock."""
        agent_seed, secret_seed = self.seeds.spawn(2)
        self.agent = LearningAgent(self.words, seed=agent_seed, featurizer=self.featurizer)
        self.agent.training = True
        self.secret_rng = np.random.default_rng(secret_seed)
        self.results = []       # practice games: (secret, guesses)
        self.skill_checks = []  # (games trained, best-guess average, practice-style average)
        self.next_check = 0     # games trained at which the next skill check is due
        self.analysis = {"state": "idle", "progress": 0.0, "games_trained": None, "result": None}
        self.run_id += 1

    def check_interval(self, games_trained=None):
        """Games between skill checks: every 5 early on, then about 20 more checks per run."""
        if games_trained is not None and games_trained < EARLY_GAMES:
            return EARLY_INTERVAL
        return max(10, self.target_games // 20)

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
        """Throw away everything learned and start again from game 0 with a blank agent."""
        with self.lock:
            self._start_new_run()
            if self.status == "finished":
                self.status = "running"

    def set_speed(self, games_per_second):
        with self.lock:
            self.games_per_second = max(0, float(games_per_second))

    def set_target(self, games):
        """Change how many games this run trains for. Takes effect immediately."""
        with self.lock:
            self.target_games = max(1, int(games))
            played = len(self.results)
            if played >= self.target_games:
                self.next_check = played  # one last check, then it finishes
            else:
                last = self.skill_checks[-1][0] if self.skill_checks else None
                due = 0 if last is None else last + self.check_interval(last)
                self.next_check = max(played, min(due, self.target_games))
                if self.status == "finished":
                    self.status = "paused"  # more games to go: press Continue

    # --- training -----------------------------------------------------------

    def _copy_agent(self, weights, seed=SKILL_CHECK_SEED):
        """A frozen copy of the agent with these weights: best guesses, fixed tie-breaks."""
        agent = LearningAgent(self.words, seed=seed, featurizer=self.featurizer)
        agent.weights = weights.copy()
        return agent

    def skill_check(self, weights):
        """Average guesses on the check words: (best guesses, playing like in practice)."""
        best = self._copy_agent(weights)
        practice = _NoLearning(self.words, seed=SKILL_CHECK_SEED, featurizer=self.featurizer)
        practice.weights = weights.copy()
        practice.training = True  # sample like in practice; _NoLearning keeps it from learning
        return tuple(float(np.mean([play_game(agent, w).guesses_used for w in self.check_words]))
                     for agent in (best, practice))

    def step(self):
        """Do the next piece of work: a skill check if one is due, otherwise one practice game.

        Returns "check", "game", or None if there was nothing to do (paused/finished)."""
        with self.lock:
            if self.status != "running":
                return None
            agent, run_id, played = self.agent, self.run_id, len(self.results)
            check_due = played >= self.next_check
            weights = agent.weights.copy()
            secret = None if check_due else self.words[self.secret_rng.integers(len(self.words))]

        if check_due:
            best, practice = self.skill_check(weights)  # the slow parts run without holding the lock
            with self.lock:
                if run_id == self.run_id:
                    self.skill_checks.append((played, best, practice))
                    self.next_check = min(played + self.check_interval(played), self.target_games)
                    if played >= self.target_games:
                        self.status = "finished"
                        agent.save(self.model_path)
            return "check"

        game = play_game(agent, secret)
        with self.lock:
            if run_id == self.run_id:  # a reset while playing throws the game away
                self.results.append((secret, game.guesses_used))
                if len(self.results) >= self.target_games:
                    self.next_check = len(self.results)  # final skill check, then finished
        return "game"

    def run_forever(self):
        """The background loop: work at the chosen speed while running."""
        while True:
            started = time.perf_counter()
            done = self.step()
            if done is None:
                time.sleep(0.05)
            elif done == "game":
                with self.lock:
                    speed = self.games_per_second
                if speed > 0:
                    time.sleep(max(0.0, 1 / speed - (time.perf_counter() - started)))

    # --- word difficulty ----------------------------------------------------

    def start_analysis(self):
        """Analyze every answer word with the agent as it is right now (in the background)."""
        with self.lock:
            if self.analysis["state"] == "running":
                return
            weights, run_id, played = self.agent.weights.copy(), self.run_id, len(self.results)
            self.analysis = {"state": "running", "progress": 0.0, "games_trained": played, "result": None}

        def progress(fraction):
            with self.lock:
                if run_id == self.run_id:
                    self.analysis["progress"] = fraction

        def work():
            result = analyze(self._copy_agent(weights), self.words, self.traits, progress)
            with self.lock:
                if run_id == self.run_id:
                    self.analysis.update(state="done", progress=1.0, result=result)

        threading.Thread(target=work, daemon=True).start()

    def word_report(self, word):
        """How the agent, as it is right now, plays one particular word."""
        word = word.strip().lower()
        if word not in self.traits:
            return {"error": f"'{word.upper()}' isn't one of the {len(self.words):,} answer words."}
        with self.lock:
            weights, played = self.agent.weights.copy(), len(self.results)
        return {"word": word, "games_trained": played, "steps": trace_game(self._copy_agent(weights), word),
                **self.traits[word]}

    # --- reading ------------------------------------------------------------

    def snapshot(self, since=0):
        """Everything the dashboard needs, plus only the practice games after `since`."""
        with self.lock:
            guesses = np.array([g for _, g in self.results])
            played = len(guesses)
            first = self.skill_checks[0][1] if self.skill_checks else None
            latest = self.skill_checks[-1] if self.skill_checks else None
            return {
                "run_id": self.run_id,
                "status": self.status,
                "target_games": self.target_games,
                "games_per_second": self.games_per_second,
                "window": RECENT_WINDOW,
                "games_played": played,
                "skill_checks": [list(check) for check in self.skill_checks],
                "check_words": len(self.check_words),
                "check_interval": self.check_interval(),
                "weights": dict(zip(FEATURE_NAMES, self.agent.weights.round(3).tolist())),
                "stats": {
                    "skill_before": first,
                    "skill_now": latest[1] if latest else None,
                    "skill_now_games": latest[0] if latest else None,
                    "last_game": ({"secret": self.results[-1][0], "guesses": int(guesses[-1])}
                                  if played else None),
                    "recent_avg": float(guesses[-RECENT_WINDOW:].mean()) if played else None,
                },
                "analysis": {k: v for k, v in self.analysis.items() if k != "result"},
                "since": since,
                "results": [[i + 1, s, g] for i, (s, g) in enumerate(self.results[since:], since)],
            }

    def analysis_result(self):
        with self.lock:
            return dict(self.analysis)
