"""Floors, learning speed and transfer across word lengths.   python benchmark.py --limits-study

Three questions, answered by experiment on the "common words" lists (3-10 letters, built the
same way for every length; see wordle/wordlists.py), with the official Wordle list as a
reference because its optimum is published (7,920 = 3.4212, Selby):

1. The floor: the fewest guesses on average that ANY strategy can reach.
   Upper bound: the best total found by search (agents/search.py) guided by trained
   planners, at two widths and from three training seeds. Lower bound: proven over every
   valid guess (analysis/limits.py). Where they meet, that is the optimum.
2. Learning speed: the planner trained from scratch (several seeds per length) and scored
   exactly on every answer after 0, 1, 2, ... practice games. How many games until it
   stops improving, and how close to the floor it gets.
3. Transfer: the planner's knowledge is 3 numbers (its V curve) and no words, so a
   planner trained on one length can play any other. Every training length x every test
   length, and a closer look at 4 letters -> 7 letters.

Two more experiments support these:
  thinking  the search guided by planners after only 1, 3, 10, ... practice games: how much
            training does the search itself need to reach its best?
  fixed     the natural lists differ in size (669 to 3,152 answers), so length and list
            size are mixed up. The same number of answers at every length (random subsets
            as big as the 3-letter list; every word of the length may still be guessed)
            separates them.

Scoring: a frozen planner (fixed tie-breaks) plays the same way every time, so its total over
every answer is exact. It's computed by walking the planner's decision tree once
(agents/lookahead.py, width 1), which gives the same total as playing every answer one by one.

Phases cache their results as JSON in the output folder, so a run can be resumed or extended
(`phases`); the report is rebuilt from whatever is there. Within the long phases, every task's
result is also saved the moment it arrives (parts/), so an interrupted run loses only the tasks
that were running.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # parallel workers: one math thread each

import datetime  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from agents.base import play_game  # noqa: E402
from agents.lookahead import LookaheadAgent  # noqa: E402
from agents.planning_agent import PlanningAgent  # noqa: E402
from agents.search import SearchAgent  # noqa: E402
from agents.split_features import SplitFeaturizer  # noqa: E402
from analysis import benchmark_tasks as tasks  # noqa: E402
from analysis.difficulty import TIE_SEED  # noqa: E402
from analysis.limits import EvenSplit, ExactSearch, one_level  # noqa: E402
from wordle.patterns import pattern_table  # noqa: E402
from wordle.words import LENGTHS, load_word_set, word_set_name  # noqa: E402

COMMON = [word_set_name(length, official=False) for length in LENGTHS]
REFERENCE = "wordle5"
OFFICIAL_OPTIMUM = 7920        # Selby: original 2,315 answers, every one of 12,972 guesses allowed
# Most expensive first, so the long training runs don't end up last in the queue.
TRAIN_ORDER = ["common10", "common9", "common8", "common7", "common6", REFERENCE, "common5", "common4", "common3"]
MARKS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 65, 80, 100, 130, 160, 200, 250, 320, 400,
         500, 650, 800, 1000]
# The wider search only from 4 letters: at 3 letters (27 color patterns, deep trees) width 50 runs for hours.
FULL = {"seeds": 8, "games": 1000, "search_widths": [20, 50], "wide_from": 4, "search_seeds": 3, "proof_seconds": 1200,
        "think_marks": [1, 3, 10, 1000], "subset_size": 669, "subset_seeds": 3, "subset_proof_seconds": 180}
QUICK = {"seeds": 2, "games": 30, "search_widths": [5], "wide_from": 3, "search_seeds": 1, "proof_seconds": 20,
         "think_marks": [1, 30], "subset_size": 669, "subset_seeds": 1, "subset_proof_seconds": 10}
PHASES = ["train", "transfer", "floor", "thinking", "fixed", "timing", "structure"]
TIMING_GAMES = 100


# --- worker tasks ---------------------------------------------------------------------------------

def exam(state):
    """Exact total guesses over every answer for a frozen planner with these learned numbers."""
    return exam_with(tasks.make("planner", state))


def exam_with(planner):
    """The same for a frozen planner already built (walks its decision tree once)."""
    n = planner.n_answers
    return int(round(n * LookaheadAgent(planner, width=1).base_value(np.arange(n))))


def train_curve(word_set, seed, games, marks):
    """Train one planner from scratch; at each mark, its learned numbers and exact total."""
    tasks.use_words(word_set)
    started = time.time()
    words = list(load_word_set(word_set).answers)
    agent = tasks.make("planner", seed=seed)
    agent.training = True
    secrets = np.random.default_rng(seed + 10_000)   # the same games as benchmark_tasks.train
    curve, practice = [], []
    if 0 in marks:
        curve.append([0, exam([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]])
    for game in range(1, games + 1):
        practice.append(play_game(agent, words[secrets.integers(len(words))]).guesses_used)
        if game in marks:
            state = [float(c) for c in agent.coef]
            curve.append([game, exam(state), state])
    return {"word_set": word_set, "seed": seed, "curve": curve, "practice": practice,
            "seconds": round(time.time() - started, 1)}


def exams(word_set, states):
    """Exact totals on this word set for many learned states (e.g. planners trained on other lengths)."""
    tasks.use_words(word_set)
    return [exam(state) for state in states]


def play_every_answer(word_set, state):
    """Each answer's guesses, for a frozen planner with these learned numbers."""
    tasks.use_words(word_set)
    agent = tasks.make("planner", state)
    return {w: [g for g, _ in play_game(agent, w).history] for w in load_word_set(word_set).answers}


def search_floor(word_set, state, width):
    """The search's best total (an upper bound on the optimum), its opener and how many guesses each answer needs."""
    tasks.use_words(word_set)
    started = time.time()
    agent = SearchAgent(tasks.make("planner", state), width=width)
    total = int(agent.total())
    counts = np.bincount([play_game(agent, w).guesses_used for w in load_word_set(word_set).answers]).tolist()
    return {"word_set": word_set, "width": width, "total": total, "opener": agent.choose([]),
            "guess_counts": counts, "nodes": agent.nodes, "seconds": round(time.time() - started, 1)}


def prove_floor(word_set, best_known, seconds):
    """Lower bound on the optimum over every valid guess (the optimum itself if it finishes)."""
    tasks.use_words(word_set)
    result = ExactSearch(tasks._featurizer("any")).prove(best_known, seconds)
    return {"word_set": word_set, "best_known": int(best_known), **result}


def smaller_game(word_set, size, seed, state, width, proof_seconds):
    """`size` random answers of a word set (every word may still be guessed): search, then proven bounds."""
    tasks.use_words(word_set)
    answers = list(load_word_set(word_set).answers)
    keep = answers if size >= len(answers) else sorted(
        str(w) for w in np.random.default_rng(seed).choice(answers, size, replace=False))
    featurizer = SplitFeaturizer(word_set, answers=keep)
    planner = PlanningAgent(seed=TIE_SEED, featurizer=featurizer)
    planner.coef = np.array(state, float)
    started = time.time()
    search = SearchAgent(planner, width=width)
    total = int(search.total())
    searched = round(time.time() - started, 1)
    proof = ExactSearch(featurizer).prove(total, proof_seconds)
    return {"word_set": word_set, "size": len(keep), "seed": seed, "planner": exam_with(planner),
            "search": total, "search_seconds": searched, "opener": search.choose([]), **proof}


class Saved:
    """Submits worker tasks, saving each result in `folder` the moment it is collected. A task whose result
    is already there isn't run again, so an interrupted study resumes where it stopped."""

    def __init__(self, pool, folder):
        self.pool, self.folder = pool, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def submit(self, key, func, args):
        """Returns a function that waits for (or loads) the result."""
        path = self.folder / f"{key}.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return lambda: value
        job = self.pool.apply_async(func, args)

        def collect():
            value = job.get()
            path.write_text(json.dumps(value), encoding="utf-8")
            return value
        return collect


# --- the phases -------------------------------------------------------------------------------------

def _final_states(train):
    """{word set: [(seed, final learned numbers, final exact total)]}."""
    out = {}
    for r in train.values():
        out.setdefault(r["word_set"], []).append((r["seed"], r["curve"][-1][2], r["curve"][-1][1]))
    return {ws: sorted(v) for ws, v in out.items()}


def _ranked_seeds(train, word_set):
    """[(seed, final learned numbers, final total)]: the median run first, then the best, the worst, the rest."""
    runs = sorted(_final_states(train)[word_set], key=lambda r: (r[2], r[0]))
    order = [(len(runs) - 1) // 2, 0, len(runs) - 1] + list(range(len(runs)))
    return [runs[i] for i in dict.fromkeys(order)]


def phase_train(pool, settings, log):
    marks = [m for m in MARKS if m <= settings["games"]] + ([settings["games"]] if settings["games"] not in MARKS
                                                             else [])
    jobs = {(ws, seed): pool.apply_async(train_curve, (ws, seed, settings["games"], marks))
            for ws in TRAIN_ORDER for seed in range(settings["seeds"])}
    results = {}
    for (ws, seed), job in jobs.items():
        r = job.get()
        results[f"{ws}/{seed}"] = r
        log(f"trained {ws} seed {seed}: {r['curve'][-1][1]:,} total after {settings['games']} games "
            f"({r['seconds']:.0f} s)")
    return results


def phase_transfer(pool, settings, log, train):
    finals = _final_states(train)
    states = [(ws, seed, state) for ws in COMMON for seed, state, _ in finals[ws]]
    jobs = {test: pool.apply_async(exams, (test, [s for _, _, s in states] + [[0.0, 0.0, 0.0]])) for test in COMMON}
    # A closer look at 4 -> 7 letters: the median 4-letter and 7-letter planners, every answer played.
    four, seven = _ranked_seeds(train, "common4")[0], _ranked_seeds(train, "common7")[0]
    detail = {name: pool.apply_async(play_every_answer, ("common7", state))
              for name, state in [("4->7", four[1]), ("7->7", seven[1])]}
    searched = {name: pool.apply_async(search_floor, ("common7", state, settings["search_widths"][0]))
                for name, state in [("4->7", four[1]), ("7->7", seven[1])]}
    matrix = []
    for test, job in jobs.items():
        totals = job.get()
        for (train_ws, seed, _), total in zip(states, totals):
            matrix.append({"train": train_ws, "seed": seed, "test": test, "total": total})
        matrix.append({"train": "blank", "seed": 0, "test": test, "total": totals[-1]})
        log(f"transfer: every planner tested on {test}")
    return {"matrix": matrix, "seeds": {"common4": four[0], "common7": seven[0]},
            "games": {name: job.get() for name, job in detail.items()},
            "search": {name: job.get() for name, job in searched.items()}}


def phase_floor(saved, settings, log, train):
    """Searches for the best plan (upper bounds), then proven lower bounds. Each list's proof starts as soon
    as its own searches are done, while the other lists' searches go on."""
    sets = COMMON + [REFERENCE]
    jobs = {ws: [] for ws in sets}
    for ws in sets:
        ranked = _ranked_seeds(train, ws)
        for width in settings["search_widths"]:
            if width == settings["search_widths"][0]:
                seeds = ranked[:settings["search_seeds"]]
            else:
                seeds = ranked[:1] if load_word_set(ws).length >= settings["wide_from"] else []
            for seed, state, _ in seeds:
                jobs[ws].append((seed, width, saved.submit(f"floor-search-{ws}-w{width}-s{seed}", search_floor,
                                                           (ws, state, width))))
    searches, proofs = [], {}
    for ws in sets:
        for seed, width, result in jobs[ws]:
            r = result()
            r["seed"] = seed
            searches.append(r)
            log(f"search {ws} width {width} seed {seed}: {r['total']:,} ({r['opener']}, {r['seconds']:.0f} s)")
            if ws == REFERENCE and r["total"] < OFFICIAL_OPTIMUM:   # below a proven minimum: a bug, never a result
                raise RuntimeError(f"search claims {r['total']} < proven optimum {OFFICIAL_OPTIMUM}")
        best = min(r["total"] for r in searches if r["word_set"] == ws)
        proofs[ws] = (best, saved.submit(f"floor-proof-{ws}", prove_floor, (ws, best, settings["proof_seconds"])))
    results = {}
    for ws, (best, result) in proofs.items():
        r = result()
        results[ws] = r
        log(f"bounds {ws}: {r['lower']:,} <= optimum <= {r['upper']:,}"
            f"{' (proven)' if r['proven'] else ''} ({r['seconds']:.0f} s)")
        if r["lower"] > best or (ws == REFERENCE and r["lower"] > OFFICIAL_OPTIMUM):   # a bound above a real total
            raise RuntimeError(f"{ws}: lower bound {r['lower']} above a total that is actually reached")
    return {"searches": searches, "proofs": results}


def phase_thinking(saved, settings, log, train):
    """The search guided by the median run's planner after only a few practice games."""
    jobs = []
    for ws in TRAIN_ORDER:
        seed = _ranked_seeds(train, ws)[0][0]
        for games, total, state in train[f"{ws}/{seed}"]["curve"]:
            if games in settings["think_marks"]:
                result = saved.submit(f"thinking-{ws}-g{games}", search_floor,
                                      (ws, state, settings["search_widths"][0]))
                jobs.append((ws, seed, games, total, result))
    rows = []
    for ws, seed, games, total, result in jobs:
        r = result()
        rows.append({"word_set": ws, "seed": seed, "games": games, "planner": total, "search": r["total"],
                     "opener": r["opener"], "seconds": r["seconds"]})
        log(f"thinking {ws} after {games} games: planner {total:,}, search {r['total']:,} ({r['seconds']:.0f} s)")
    return rows


def phase_fixed(saved, settings, log, train):
    """The same number of answers at every length."""
    jobs = []
    for ws in COMMON:
        state = _ranked_seeds(train, ws)[0][1]
        seeds = 1 if len(load_word_set(ws).answers) <= settings["subset_size"] else settings["subset_seeds"]
        for seed in range(seeds):
            jobs.append(saved.submit(f"fixed-{ws}-s{seed}", smaller_game,
                                     (ws, settings["subset_size"], seed, state, settings["search_widths"][0],
                                      settings["subset_proof_seconds"])))
    rows = []
    for result in jobs:
        r = result()
        rows.append(r)
        log(f"fixed size {r['word_set']} subset {r['seed']}: planner {r['planner']:,}, search {r['search']:,}, "
            f"{r['lower']:,} <= optimum <= {r['upper']:,}{' (proven)' if r['proven'] else ''}")
    return rows


def split_summary(featurizer, guess):
    """How one opener splits every answer: groups, words left on average, the largest group."""
    sizes = np.bincount(featurizer.by_answer[:, featurizer.index[guess]])
    sizes = sizes[sizes > 0]
    return {"guess": guess, "groups": int(len(sizes)), "mean_left": float((sizes ** 2).sum() / sizes.sum()),
            "largest": int(sizes.max())}


def phase_structure(log, train, floor):
    """Why length matters: how many color groups one guess can make, and how much the openers narrow
    things down. The largest group after the planner's own opener is also about the largest number of
    words left it ever sees while practicing, i.e. the range its curve V(m) is fitted on."""
    rows = {}
    for ws in COMMON + [REFERENCE]:
        tasks.use_words(ws)
        featurizer = tasks._featurizer("any")
        _, k = one_level(featurizer, np.arange(featurizer.n_answers), EvenSplit(featurizer))
        planner = tasks.make("planner", _ranked_seeds(train, ws)[0][1])
        rows[ws] = {"length": featurizer.length, "answers": featurizer.n_answers, "guesses": len(featurizer.pool),
                    "patterns": featurizer.n_patterns, "most_groups": int(k),
                    "planner_opener": split_summary(featurizer, planner.choose([]))}
        if floor:
            best = min((r for r in floor["searches"] if r["word_set"] == ws), key=lambda r: r["total"])
            rows[ws]["best_opener"] = split_summary(featurizer, best["opener"])
        log(f"structure {ws}: at most {k} groups per guess; planner opener {rows[ws]['planner_opener']}")
    return rows


def phase_timing(log, train):
    """Milliseconds per practice game, trained planner, one process: run it with nothing else going on
    (the training runs' own timings include whatever else the machine was doing, sleep included)."""
    rows = {}
    for ws in TRAIN_ORDER:
        tasks.use_words(ws)
        words = list(load_word_set(ws).answers)
        agent = tasks.make("planner", _ranked_seeds(train, ws)[0][1], seed=0)
        agent.training = True
        rng = np.random.default_rng(0)
        for _ in range(3):                      # warm-up: first-turn caches
            play_game(agent, words[rng.integers(len(words))])
        started = time.perf_counter()
        for _ in range(TIMING_GAMES):
            play_game(agent, words[rng.integers(len(words))])
        rows[ws] = (time.perf_counter() - started) / TIMING_GAMES
        log(f"timing {ws}: {1000 * rows[ws]:.0f} ms per practice game")
    return rows


def run(workers, quick=False, out=None, phases=None):
    settings = QUICK if quick else FULL
    started = time.time()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(out or f"reports/{stamp}_limits{'_quick' if quick else ''}")
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(f"[{time.time() - started:6.0f}s] {msg}", flush=True)  # noqa: E731
    phases = phases or PHASES
    for ws in COMMON + [REFERENCE]:   # build the pattern tables once, here, rather than in every worker
        pattern_table(ws)

    def load(name):
        path = out / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save(name, data):
        (out / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

    with Pool(workers, initializer=tasks.init_worker, initargs=(COMMON[0],)) as pool:
        saved = Saved(pool, out / "parts")
        train = load("train")
        if train is None and "train" in phases:
            log(f"training {settings['seeds']} planners per word set, {settings['games']} games each...")
            train = phase_train(pool, settings, log)
            save("train", train)
        if train is not None and "transfer" in phases and load("transfer") is None:
            log("transfer: every trained planner on every length...")
            save("transfer", phase_transfer(pool, settings, log, train))
        if train is not None and "floor" in phases and load("floor") is None:
            log("floor: searches, then proven bounds...")
            save("floor", phase_floor(saved, settings, log, train))
        if train is not None and "thinking" in phases and load("thinking") is None:
            log("thinking: search guided by barely trained planners...")
            save("thinking", phase_thinking(saved, settings, log, train))
        if train is not None and "fixed" in phases and load("fixed") is None:
            log(f"fixed: {settings['subset_size']} answers at every length...")
            save("fixed", phase_fixed(saved, settings, log, train))
    if train is not None and "timing" in phases and load("timing") is None:
        save("timing", phase_timing(log, train))
    if train is not None and "structure" in phases and load("structure") is None:
        save("structure", phase_structure(log, train, load("floor")))
    (out / "settings.json").write_text(json.dumps({"settings": settings, "marks": MARKS, "workers": workers,
                                                    "quick": quick}, indent=2), encoding="utf-8")
    log(f"done: {out}")
    return out
