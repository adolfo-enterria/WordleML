"""A training run you can watch and control: pause, continue, reset, speed, length, mode.

The learning agent plays practice games in a background thread. Each game the
secret is a word picked at random from the full answer list, so the agent never
has a clue before its first guess.

Every so often training stops for a "skill check": the agent plays the same
fixed words using its best guesses (no exploring, fixed tie-breaks). Same words
every time, so the luck of which word comes up can't move the curve; only
learning can. The first check is at game 0, before any training.

Modes (see agents/__init__.py): "any" is the planning agent, which may guess
any valid word, including probe words that can't win; "possible" only guesses
words that could still be the answer. Changing the mode starts a fresh run.
When a run finishes it also takes a "final exam": every possible answer once.
For the official Wordle list that's the number to compare with the proven
optimum (3.4201). A "look-ahead exam" can also be run on demand: the same AI,
but thinking further ahead at each turn (agents/lookahead.py).

Word sets (word length): the official 5-letter Wordle list by default, or common
English words of 3-8 letters. Switching prepares the new set in the background
(its pattern table takes up to ~20 s to build the first time) and starts fresh.

From the practice games we also keep track of what the agent has learned to
do: how often it probes, a recent example of a probe, whether it follows a
weak opener with fresh letters, and how each opener has worked out.

The web dashboard reads everything with snapshot() and sends button presses to
the control methods.
"""
import threading
import time
from collections import deque

import numpy as np

from agents import MODES, make_learner
from agents import model_path as default_model_path
from agents.base import play_game
from agents.learning_agent import make_featurizer
from agents.lookahead import LookaheadAgent
from analysis.difficulty import analyze, trace_game, word_traits
from wordle.patterns import pattern_table
from wordle.words import DEFAULT_WORD_SET, LENGTHS, load_word_set, word_set_name

RECENT_WINDOW = 50        # practice games in the rolling average
EARLY_GAMES = 100         # most of the learning happens here, so check more often
EARLY_INTERVAL = 10
SKILL_CHECK_WORDS = 100   # same words at every skill check
SKILL_CHECK_SEED = 12345  # picks those words and breaks ties the same way every time
STATS_WINDOW = 200        # recent practice games behind the "what it learned to do" numbers
PROBE_BUCKETS = [(2, 2, "2 words possible"), (3, 5, "3-5"), (6, 20, "6-20"), (21, 10 ** 6, "21 or more")]
WEAK_OPENER = 100         # an opener that leaves more words than this wasn't much help
SMALL_PROBE = 12          # probe examples are shown when at most this many words were possible
LOOKAHEAD_WIDTH = 10      # candidate guesses the look-ahead exam weighs each turn
OPTIMUM = {"wordle5": 3.4201}  # proven best average (Bertsimas & Paskov; Selby); none for other lists


class TrainingSession:
    def __init__(self, word_set=DEFAULT_WORD_SET, target_games=500, games_per_second=10, seed=None,
                 model_path=None, mode="any", exam_size=None):
        self.model_path_override = model_path   # tests use a temporary file
        self.exam_size = exam_size              # tests use a few words for a quick exam
        self.target_games = target_games
        self.games_per_second = games_per_second  # 0 means as fast as possible
        self.seeds = np.random.SeedSequence(seed)
        self.lock = threading.Lock()
        self._featurizers = {}
        self.mode = mode
        self.preparing, self.words_error = None, None
        self._use_words(word_set)
        self.featurizer(mode)
        self.status = "running"
        self.run_id = 0
        self._start_new_run()

    def _use_words(self, name, featurizer=None):
        """Switch every word-dependent piece to word set `name` (caller holds the lock, or __init__)."""
        self.word_set = load_word_set(name)
        self.words = list(self.word_set.answers)
        self.exam_words = self.words[:self.exam_size] if self.exam_size else list(self.words)
        self.traits = word_traits(self.words)
        rng = np.random.default_rng(SKILL_CHECK_SEED)
        count = min(SKILL_CHECK_WORDS, len(self.words))
        self.check_words = sorted(str(w) for w in rng.choice(self.words, count, replace=False))
        self.model_path = self.model_path_override or default_model_path(name)
        self._featurizers = {key: f for key, f in self._featurizers.items() if key[0] == name}
        if featurizer is not None:
            self._featurizers[(name, self.mode)] = featurizer

    def featurizer(self, mode=None):
        """One shared (read-only) featurizer per word set and mode, built the first time it's needed."""
        key = (self.word_set.name, mode or self.mode)
        if key not in self._featurizers:
            self._featurizers[key] = make_featurizer(key[1], self.words, key[0])
        return self._featurizers[key]

    def _start_new_run(self):
        """Fresh, untrained agent (all weights zero) and no results. Caller holds the lock."""
        agent_seed, secret_seed = self.seeds.spawn(2)
        self.agent = make_learner(self.mode, self.words, seed=agent_seed, featurizer=self.featurizer(),
                                  word_set=self.word_set.name)
        self.agent.training = True
        self.secret_rng = np.random.default_rng(secret_seed)
        self.results = []       # practice games: (secret, guesses)
        self.skill_checks = []  # (games trained, average guesses on the check words)
        self.final_exam = None  # every answer once, when the run finishes
        self.next_check = 0     # games trained at which the next skill check is due
        self.analysis = {"state": "idle", "progress": 0.0, "games_trained": None, "result": None}
        self.lookahead = {"state": "idle"}
        self.decisions = deque(maxlen=STATS_WINDOW)     # per game: [(words possible, was a probe)]
        self.weak_openers = deque(maxlen=STATS_WINDOW)  # per weak opener: did guess 2 use fresh letters?
        self.latest_probe = None
        self.openers = {}       # opener -> [games, sum of guesses, sum of squares]
        self.run_id += 1

    def check_interval(self, games_trained=None):
        """Games between skill checks: every 10 early on, then about 20 more checks per run."""
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

    def set_mode(self, mode):
        """Switch between "any" and "possible". This starts a fresh run."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.featurizer(mode)  # may take a second the first time; build it before taking the lock
        with self.lock:
            self.mode = mode
            self._start_new_run()
            if self.status == "finished":
                self.status = "running"

    def set_words(self, length=5, official=True):
        """Switch word length / list. Prepares the new word set in the background, then starts fresh."""
        length = int(length)
        if length not in LENGTHS:
            raise ValueError(f"length must be {LENGTHS.start}-{LENGTHS.stop - 1}")
        name = word_set_name(length, bool(official))
        with self.lock:
            if name == self.word_set.name or self.preparing:
                return
            self.preparing, self.words_error, self.status = name, None, "preparing"
            mode = self.mode

        def work():
            try:
                pattern_table(name)  # the slow part the first time (built and cached on disk)
                featurizer = make_featurizer(mode, load_word_set(name).answers, name)
                with self.lock:
                    self._use_words(name, featurizer if mode == self.mode else None)
                    self._start_new_run()
                    self.status = "running"
            except Exception as error:  # e.g. the word lists haven't been built
                with self.lock:
                    self.words_error = f"{type(error).__name__}: {error}"
                    self.status = "paused"
            finally:
                with self.lock:
                    self.preparing = None

        threading.Thread(target=work, daemon=True).start()

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

    def skill_check(self, frozen):
        """Average guesses on the check words, using the agent's best guesses."""
        return float(np.mean([play_game(frozen, w).guesses_used for w in self.check_words]))

    def exam(self, frozen):
        """Every possible answer once: for the official list, the number to compare with the proven optimum."""
        guesses = np.array([play_game(frozen, w).guesses_used for w in self.exam_words])
        return {"avg": float(guesses.mean()), "worst": int(guesses.max()),
                "within_six": float((guesses <= 6).mean())}

    def _describe(self, agent, game):
        """What the agent did in one game: every real decision, plus notable moments."""
        featurizer, pool = agent.featurizer, agent.featurizer.pool
        decisions, probe, weak_fresh = [], None, None
        for turn, (guess, _) in enumerate(game.history):
            possible = [pool[i] for i in featurizer.candidates(game.history[:turn])]
            if len(possible) == 1:
                continue  # nothing to decide: the rules leave one word
            is_probe = guess not in set(possible)
            decisions.append((len(possible), is_probe))
            if is_probe and len(possible) <= SMALL_PROBE:
                shared = set.intersection(*(set(w) for w in possible))
                tested = sorted({c for c in guess if c not in shared and any(c in w for w in possible)})
                probe = {"guess": guess, "possible": possible, "tested": tested,
                         "left_after": len(featurizer.candidates(game.history[:turn + 1])),
                         "secret": game.secret}
        if game.guesses_used > 1 and len(featurizer.candidates(game.history[:1])) > WEAK_OPENER:
            opener, second = game.history[0][0], game.history[1][0]
            weak_fresh = not set(opener) & set(second)
        return decisions, probe, weak_fresh

    def step(self):
        """Do the next piece of work: a skill check if one is due, otherwise one practice game.

        Returns "check", "game", or None if there was nothing to do (paused/finished)."""
        with self.lock:
            if self.status != "running":
                return None
            agent, run_id, played = self.agent, self.run_id, len(self.results)
            check_due = played >= self.next_check
            frozen = agent.frozen_copy(SKILL_CHECK_SEED) if check_due else None
            secret = None if check_due else self.words[self.secret_rng.integers(len(self.words))]

        if check_due:
            score = self.skill_check(frozen)  # the slow parts run without holding the lock
            exam = self.exam(frozen) if played >= self.target_games else None
            with self.lock:
                if run_id == self.run_id:
                    self.skill_checks.append((played, score))
                    self.next_check = min(played + self.check_interval(played), self.target_games)
                    if exam is not None:
                        self.final_exam = exam | {"games_trained": played}
                        self.status = "finished"
                        agent.save(self.model_path)
            return "check"

        game = play_game(agent, secret)
        decisions, probe, weak_fresh = self._describe(agent, game)
        with self.lock:
            if run_id != self.run_id:  # a reset while playing throws the game away
                return "game"
            self.results.append((secret, game.guesses_used))
            self.decisions.append(decisions)
            if probe:
                self.latest_probe = probe | {"game": len(self.results)}
            if weak_fresh is not None:
                self.weak_openers.append(weak_fresh)
            stats = self.openers.setdefault(game.history[0][0], [0, 0, 0])
            stats[0] += 1
            stats[1] += game.guesses_used
            stats[2] += game.guesses_used ** 2
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

    # --- look-ahead exam ----------------------------------------------------

    def start_lookahead(self):
        """Exam with look-ahead (in the background): the planner as it is now, thinking further ahead."""
        with self.lock:
            if self.mode != "any" or self.lookahead.get("state") == "running":
                return
            frozen, run_id, played = self.agent.frozen_copy(), self.run_id, len(self.results)
            words = list(self.exam_words)
            self.lookahead = {"state": "running", "games_trained": played}

        def work():
            started = time.perf_counter()
            agent = LookaheadAgent(frozen, width=LOOKAHEAD_WIDTH)
            guesses = np.array([play_game(agent, w).guesses_used for w in words])
            with self.lock:
                if run_id == self.run_id:
                    self.lookahead = {"state": "done", "games_trained": played, "avg": float(guesses.mean()),
                                      "worst": int(guesses.max()), "within_six": float((guesses <= 6).mean()),
                                      "opener": agent.choose([]), "seconds": time.perf_counter() - started}

        threading.Thread(target=work, daemon=True).start()

    # --- word difficulty ----------------------------------------------------

    def start_analysis(self):
        """Analyze every answer word with the agent as it is right now (in the background)."""
        with self.lock:
            if self.analysis["state"] == "running":
                return
            frozen, run_id, played = self.agent.frozen_copy(), self.run_id, len(self.results)
            self.analysis = {"state": "running", "progress": 0.0, "games_trained": played, "result": None}

        def progress(fraction):
            with self.lock:
                if run_id == self.run_id:
                    self.analysis["progress"] = fraction

        def work():
            result = analyze(frozen, self.words, self.traits, progress)
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
            frozen, played = self.agent.frozen_copy(), len(self.results)
        return {"word": word, "games_trained": played, "steps": trace_game(frozen, word), **self.traits[word]}

    # --- reading ------------------------------------------------------------

    def _opener_table(self, top=8):
        rows = []
        for word, (games, total, squares) in self.openers.items():
            mean = total / games
            spread = np.sqrt(max(squares / games - mean ** 2, 0.0) * games / (games - 1)) if games > 1 else None
            rows.append({"word": word, "games": games, "avg": mean,
                         "ci": 1.96 * spread / np.sqrt(games) if spread is not None else None})
        return sorted(rows, key=lambda r: (-r["games"], r["avg"]))[:top]

    def _games_to_compare(self, guesses, difference=0.05):
        """Practice games per opener needed to tell apart two openers `difference` guesses apart
        (95% confidence, 80% power), given how much single games vary."""
        if len(guesses) < 30:
            return None
        variance = float(np.var(guesses[-STATS_WINDOW:], ddof=1))
        return int(np.ceil(2 * (1.96 + 0.84) ** 2 * variance / difference ** 2))

    def snapshot(self, since=0):
        """Everything the dashboard needs, plus only the practice games after `since`."""
        with self.lock:
            guesses = np.array([g for _, g in self.results])
            played = len(guesses)
            first = self.skill_checks[0][1] if self.skill_checks else None
            latest = self.skill_checks[-1] if self.skill_checks else None
            recent = [d for game in self.decisions for d in game]
            probing = []
            for lo, hi, label in PROBE_BUCKETS:
                bucket = [is_probe for n, is_probe in recent if lo <= n <= hi]
                probing.append({"label": label, "decisions": len(bucket), "probes": int(sum(bucket))})
            return {
                "run_id": self.run_id,
                "status": self.status,
                "mode": self.mode,
                "target_games": self.target_games,
                "games_per_second": self.games_per_second,
                "window": RECENT_WINDOW,
                "games_played": played,
                "skill_checks": [list(check) for check in self.skill_checks],
                "check_words": len(self.check_words),
                "check_interval": self.check_interval(),
                "knowledge": self.agent.knowledge(),
                "final_exam": self.final_exam,
                "lookahead": dict(self.lookahead),
                "words": {
                    "name": self.word_set.name, "label": self.word_set.label, "length": self.word_set.length,
                    "official": self.word_set.official, "answers": len(self.words),
                    "guesses": len(self.word_set.guesses), "exam_words": len(self.exam_words),
                    "optimum": OPTIMUM.get(self.word_set.name), "preparing": self.preparing,
                    "error": self.words_error,
                },
                "stats": {
                    "skill_before": first,
                    "skill_now": latest[1] if latest else None,
                    "skill_now_games": latest[0] if latest else None,
                    "last_game": ({"secret": self.results[-1][0], "guesses": int(guesses[-1])}
                                  if played else None),
                    "recent_avg": float(guesses[-RECENT_WINDOW:].mean()) if played else None,
                },
                "learned": {
                    "window": min(played, STATS_WINDOW),
                    "probing": probing,
                    "latest_probe": self.latest_probe,
                    "weak_openers": len(self.weak_openers),
                    "weak_fresh": int(sum(self.weak_openers)),
                    "weak_threshold": WEAK_OPENER,
                },
                "openers": {
                    "tried": len(self.openers),
                    "table": self._opener_table(),
                    "favorite": self.agent.top_choices([], 1)[0][0],
                    "games_to_compare": self._games_to_compare(guesses),
                },
                "analysis": {k: v for k, v in self.analysis.items() if k != "result"},
                "since": since,
                "results": [[i + 1, s, g] for i, (s, g) in enumerate(self.results[since:], since)],
            }

    def analysis_result(self):
        with self.lock:
            return dict(self.analysis)
