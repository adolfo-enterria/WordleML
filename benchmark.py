"""Benchmark the Wordle AIs: reproducible numbers you can put in a write-up.

    python benchmark.py              # full run (about 20-30 minutes)
    python benchmark.py --quick      # smoke test (a few minutes)
    python benchmark.py --length-study   # how word length (3-8 letters) changes the game

Protocol: every one of the 2,315 possible answers is played exactly once, with
each strategy's best guesses and fixed tie-breaks, so a score has no luck in
it. Learning agents are trained from scratch with several seeds, and results
are reported as the mean and a 95% confidence interval across seeds.

Output in reports/<date-time>/: report.md, CSV files, charts, and config.json
(every setting, seed and file hash needed to reproduce the numbers).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # parallel workers: one math thread each

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from agents.planning_agent import PlanningAgent  # noqa: E402
from analysis import benchmark_tasks as tasks  # noqa: E402
from analysis import figures  # noqa: E402
from analysis.difficulty import analyze, trace_game, word_traits  # noqa: E402
from analysis.stats import games_to_tell_apart, mean_ci, ols, paired_bootstrap  # noqa: E402
from wordle.patterns import CACHE_PATH, pattern_table  # noqa: E402
from wordle.words import ALLOWED_GUESSES_PATH, ANSWERS_PATH, load_words  # noqa: E402

OPTIMUM, HARD_MODE_OPTIMUM = 3.4201, 3.5076
LITERATURE_OPENERS = ["salet", "reast", "crate", "trace", "slate", "soare", "roate", "raise", "crane", "stare"]
FAMILY_EXAMPLES = ["night", "catch", "mound", "shave", "foyer", "tight"]
BUCKETS = [(2, 2, "2 words possible"), (3, 5, "3-5"), (6, 20, "6-20"), (21, 10 ** 6, "21 or more")]
LABELS = {
    "random": "No strategy: random word that could be the answer",
    "blank": "Blank planner (game 0): random valid words",
    "pg-possible": "Policy gradient, only possible words (3 word facts)",
    "pg-any": "Policy gradient, any word (4 split facts)",
    "greedy-avg_left": "Hand-made: fewest words left on average",
    "greedy-entropy": "Hand-made: most information (max entropy)",
    "planner": "Learned planner: V(m), plans one guess ahead",
}

FULL = {"seeds": 10, "pg_seeds": 5, "games": {"planner": 500, "pg-any": 1000, "pg-possible": 1000},
        "ablation_seeds": 5, "random_seeds": 5, "check_words": 200, "top_openers": 10, "lookahead_widths": [5, 10]}
QUICK = {"seeds": 2, "pg_seeds": 2, "games": {"planner": 40, "pg-any": 40, "pg-possible": 60},
         "ablation_seeds": 1, "random_seeds": 1, "check_words": 40, "top_openers": 3, "lookahead_widths": [5]}
MARKS = [0, 1, 3, 10, 30, 100, 200, 300, 500, 700, 1000]


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def submit_eval(pool, spec, words, record_turns=False, chunks=6):
    parts = [list(map(str, part)) for part in np.array_split(words, chunks) if len(part)]
    return [pool.apply_async(tasks.evaluate_chunk, (spec, part, record_turns)) for part in parts]


def collect(jobs):
    rows = []
    for job in jobs:
        rows += job.get()
    return rows


def summarize_runs(runs):
    """runs: list of per-run row lists (every answer once). Mean/CI across runs, worst, % within 6."""
    means = [np.mean([r["guesses"] for r in rows]) for rows in runs]
    mean, ci = mean_ci(means)
    worst = max(max(r["guesses"] for r in rows) for rows in runs)
    within6 = np.mean([np.mean([r["guesses"] <= 6 for r in rows]) for rows in runs])
    return {"mean": mean, "ci": ci, "runs": len(runs), "worst": worst, "within6": float(within6),
            "per_run": [float(m) for m in means]}


def per_word(runs, words):
    """Average guesses per secret word across runs, in `words` order."""
    table = {w: [] for w in words}
    for rows in runs:
        for r in rows:
            table[r["secret"]].append(r["guesses"])
    return np.array([np.mean(table[w]) for w in words])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small, fast smoke test")
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--out", help="output folder (default reports/<date-time>)")
    parser.add_argument("--length-study", action="store_true",
                        help="compare word lengths 3-8 instead (see analysis/length_study.py)")
    args = parser.parse_args()
    if args.length_study:
        from analysis.length_study import run
        run(args.workers, quick=args.quick, out=args.out)
        return

    settings = QUICK if args.quick else FULL
    started = time.time()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out or f"reports/{stamp}{'_quick' if args.quick else ''}")
    out.mkdir(parents=True, exist_ok=True)
    words = load_words()
    pattern_table()  # build the cache once here rather than in every worker
    check_words = sorted(str(w) for w in np.random.default_rng(12345).choice(words, settings["check_words"],
                                                                             replace=False))
    exam_words = (sorted(str(w) for w in np.random.default_rng(7).choice(words, 300, replace=False))
                  if args.quick else words)
    marks = {kind: [m for m in MARKS if m <= games] + ([games] if games not in MARKS else [])
             for kind, games in settings["games"].items()}
    log = lambda msg: print(f"[{time.time() - started:6.0f}s] {msg}", flush=True)  # noqa: E731

    with Pool(args.workers, initializer=tasks.init_worker) as pool:
        # 1. Train every learner from scratch, several seeds each (plus the ablation variants).
        log(f"training with {args.workers} workers...")
        train_jobs = {}
        for kind, n_seeds in [("planner", settings["seeds"]), ("pg-any", settings["pg_seeds"]),
                              ("pg-possible", settings["seeds"])]:
            for seed in range(n_seeds):
                train_jobs[(kind, "main", seed)] = pool.apply_async(
                    tasks.train, (kind, seed, settings["games"][kind], check_words, marks[kind]))
        ablations = {"no forgetting": {"forget": 1.0}, "straight-line V (2 numbers)": {"curve": "line"}}
        for name, options in ablations.items():
            for seed in range(settings["ablation_seeds"]):
                train_jobs[("planner", name, seed)] = pool.apply_async(
                    tasks.train, ("planner", seed, settings["games"]["planner"], check_words,
                                  marks["planner"], options))
        trained = {key: job.get() for key, job in train_jobs.items()}
        log("training done; playing every answer with every strategy...")

        # 2. Play every answer once with every strategy.
        eval_jobs = {}
        for (kind, variant, seed), result in trained.items():
            spec = {"kind": kind, "state": result["state"], "options": result["options"]}
            eval_jobs[(kind, variant, seed)] = submit_eval(pool, spec, exam_words)
        for criterion in ("entropy", "avg_left"):
            eval_jobs[(f"greedy-{criterion}", "main", 0)] = submit_eval(
                pool, {"kind": "greedy", "criterion": criterion}, exam_words)
        for seed in range(settings["random_seeds"]):
            eval_jobs[("random", "main", seed)] = submit_eval(pool, {"kind": "random", "seed": seed}, exam_words)
        eval_jobs[("blank", "main", 0)] = submit_eval(pool, {"kind": "planner", "state": [0, 0, 0]}, exam_words,
                                                      chunks=12)
        evaluated = {key: collect(jobs) for key, jobs in eval_jobs.items()}
        log("evaluation done; opener study...")

        # 3. Openers: the median planner, with its first guess forced to each candidate opener.
        planner_runs = sorted((np.mean([r["guesses"] for r in evaluated[("planner", "main", s)]]), s)
                              for s in range(settings["seeds"]))
        median_seed = planner_runs[len(planner_runs) // 2][1]
        median_state = trained[("planner", "main", median_seed)]["state"]
        own_opener = evaluated[("planner", "main", median_seed)][0]["history"][0]
        top = pool.apply(tasks.top_openers, ("planner", median_state, None, settings["top_openers"]))
        literature = LITERATURE_OPENERS if not args.quick else LITERATURE_OPENERS[:2]
        openers = list(dict.fromkeys(top + literature))
        opener_jobs = {o: submit_eval(pool, {"kind": "planner", "state": median_state, "opener": o}, exam_words)
                       for o in openers}
        emergent_job = submit_eval(pool, {"kind": "planner", "state": median_state}, exam_words, record_turns=True)
        lookahead_jobs = {width: pool.apply_async(tasks.lookahead_exam, (median_state, width, exam_words))
                          for width in settings["lookahead_widths"]}
        opener_rows = {o: collect(jobs) for o, jobs in opener_jobs.items()}
        emergent_rows = collect(emergent_job)
        lookahead = {width: job.get() for width, job in lookahead_jobs.items()}
        for width, result in lookahead.items():
            evaluated[(f"lookahead-w{width}", "main", 0)] = result["rows"]
        log("openers and look-ahead done; word difficulty...")

    # 4. Word difficulty with the median planner (single process), and a regression over traits.
    agent = PlanningAgent(seed=0)
    agent.coef = np.array(median_state)
    difficulty = analyze(agent, words, keep_rows=True)
    traces = {w: trace_game(agent.frozen_copy(), w) for w in FAMILY_EXAMPLES}
    traits = word_traits(words)

    report = write_outputs(out, settings, args, words, exam_words, check_words, marks, trained, evaluated,
                           ablations, median_seed, own_opener, top, opener_rows, emergent_rows,
                           difficulty, traces, traits, lookahead, time.time() - started)
    log(f"done: {report}")


def write_outputs(out, settings, args, words, exam_words, check_words, marks, trained, evaluated, ablations,
                  median_seed, own_opener, top, opener_rows, emergent_rows, difficulty, traces, traits, lookahead,
                  seconds):
    def runs(kind, variant="main"):
        return [rows for (k, v, _), rows in sorted(evaluated.items()) if k == kind and v == variant]

    # --- ladder ---------------------------------------------------------------
    ladder = []
    kinds = ["random", "blank", "pg-possible", "pg-any", "greedy-avg_left", "greedy-entropy", "planner"]
    kinds += [f"lookahead-w{width}" for width in sorted(lookahead)]
    for kind in kinds:
        s = summarize_runs(runs(kind))
        label = LABELS.get(kind) or f"Learned planner + look-ahead ({kind.split('-w')[1]} candidates per turn)"
        ladder.append({"strategy": label, "kind": kind, **s})
    with open(out / "ladder.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "runs", "mean_guesses", "ci95", "worst", "share_within_6"])
        for row in ladder:
            w.writerow([row["strategy"], row["runs"], f"{row['mean']:.4f}", f"{row['ci']:.4f}", row["worst"],
                        f"{row['within6']:.4f}"])
    best = {row["kind"]: row for row in ladder}

    # --- learning curves ------------------------------------------------------
    curves, curve_rows = {}, []
    for kind, label in [("planner", "planner"), ("pg-any", "policy gradient, any word"),
                        ("pg-possible", "policy gradient, possible words")]:
        results = [r for (k, v, _), r in sorted(trained.items()) if k == kind and v == "main"]
        games = np.array([g for g, _ in results[0]["curve"]], float)
        values = np.array([[score for _, score in r["curve"]] for r in results])
        mean = values.mean(axis=0)
        ci = np.array([mean_ci(col)[1] if len(col) > 1 else 0.0 for col in values.T])
        curves[label] = (games, mean, np.nan_to_num(ci))
        for r in results:
            for g, score in r["curve"]:
                curve_rows.append([kind, r["seed"], g, f"{score:.4f}"])
    with open(out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "seed", "games_trained", "skill_on_check_words"])
        w.writerows(curve_rows)
    figures.learning_curves(curves, out / "learning_curves.png", len(check_words))
    g_planner, m_planner, _ = curves["planner"]
    within = next(int(g) for g, m in zip(g_planner, m_planner) if m <= m_planner[-1] + 0.05)

    # --- per word + paired comparisons -----------------------------------------
    pw = {kind: per_word(runs(kind), exam_words) for kind in best}
    with open(out / "per_word.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["secret"] + list(pw))
        for i, word in enumerate(exam_words):
            w.writerow([word] + [f"{pw[k][i]:.3f}" for k in pw])
    comparisons = [(other, *paired_bootstrap(pw["planner"], pw[other]))
                   for other in ["greedy-entropy", "greedy-avg_left", "pg-any", "pg-possible", "random"]]
    widest = f"lookahead-w{max(lookahead)}"
    lookahead_gain = paired_bootstrap(pw[widest], pw["planner"])

    # --- openers -------------------------------------------------------------------
    opener_table = sorted(((o, float(np.mean([r["guesses"] for r in rows])), int(max(r["guesses"] for r in rows)))
                           for o, rows in opener_rows.items()), key=lambda t: t[1])
    with open(out / "openers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opener", "mean_guesses", "worst"])
        w.writerows([[o, f"{m:.4f}", worst] for o, m, worst in opener_table])
    figures.dot_plot([(o.upper(), m, None) for o, m, _ in opener_table], out / "openers.png",
                     "Openers, each followed by the learned planner",
                     "Exact average over all answers; orange = the planner's own choice; dashed = proven "
                     "optimum 3.4201 (SALET with perfect play after it)",
                     highlight=own_opener.upper())
    spread = float(np.std([r["guesses"] for r in evaluated[("planner", "main", median_seed)]], ddof=1))

    # --- emergent strategies ---------------------------------------------------------
    decisions = [t for r in emergent_rows for t in r["turns"]]
    probing = []
    for lo, hi, label in BUCKETS:
        bucket = [p for n, p in decisions if lo <= n <= hi]
        probing.append((label, np.mean(bucket) if bucket else 0.0, f"{sum(bucket)} of {len(bucket)} guesses"))
    figures.bars(probing, out / "probing.png", "How often the planner plays a word that can't win",
                 "By how many words were still possible, over every answer", "share of guesses that were probes")

    # --- word difficulty regression -------------------------------------------------------
    rows_by_word = {r["word"]: r for r in difficulty["rows"]}
    x = np.array([[traits[w]["repeated_letters"], len(traits[w]["look_alikes"]),
                   np.log2(max(rows_by_word[w]["left_after_opener"], 1)), traits[w]["letter_commonness"]]
                  for w in words])
    regression, r2 = ols(x, per_word(runs("planner"), words) if exam_words == words else
                         np.array([rows_by_word[w]["guesses"] for w in words], float),
                         ["repeated letter (0/1)", "one-letter look-alikes", "log2(words left after opener)",
                          "letter commonness"])

    # --- ablation ---------------------------------------------------------------------------
    ablation_rows = [("full planner (curved V, forgetting)", best["planner"]["mean"], best["planner"]["ci"])]
    for name in ablations:
        s = summarize_runs(runs("planner", name))
        ablation_rows.append((name, s["mean"], s["ci"]))
    with open(out / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mean_guesses", "ci95"])
        w.writerows([[n, f"{m:.4f}", f"{c:.4f}"] for n, m, c in ablation_rows])

    figures.dot_plot([(row["strategy"], row["mean"], row["ci"] if row["runs"] > 1 else None) for row in ladder],
                     out / "ladder.png", "How well each strategy plays",
                     "Average guesses over all 2,315 answers; bars = 95% CI across seeds; dashed = proven optimum")

    config = {
        "command": " ".join(sys.argv), "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": round(seconds), "settings": settings, "workers": args.workers,
        "check_words": check_words, "exam_words": len(exam_words), "marks": marks,
        "median_planner_seed": median_seed,
        "files": {"answers.txt": sha1(ANSWERS_PATH), "allowed_guesses.txt": sha1(ALLOWED_GUESSES_PATH),
                  "patterns.npz": sha1(CACHE_PATH)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "learned": {f"{k}/{v}/{s}": r["state"] for (k, v, s), r in trained.items()},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    report = render_report(settings, args, exam_words, check_words, ladder, best, within, comparisons,
                           opener_table, own_opener, top, spread, probing, traces, difficulty, regression, r2,
                           ablation_rows, curves, lookahead, lookahead_gain, seconds)
    (out / "report.md").write_text(report, encoding="utf-8")
    return out / "report.md"


def render_report(settings, args, exam_words, check_words, ladder, best, within, comparisons, opener_table,
                  own_opener, top, spread, probing, traces, difficulty, regression, r2, ablation_rows, curves,
                  lookahead, lookahead_gain, seconds):
    planner = best["planner"]
    gap = planner["mean"] - OPTIMUM
    width = max(lookahead)
    ahead = best[f"lookahead-w{width}"]
    ahead_gap = ahead["mean"] - OPTIMUM
    full_list = len(exam_words) == 2315
    lines = [
        "# WordleML benchmark report",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} in {seconds / 60:.0f} minutes "
        f"({'QUICK smoke test, not for citing' if args.quick else 'full run'}). "
        "Every number here can be regenerated with `python benchmark.py`; settings, seeds and file hashes are in "
        "`config.json`.",
        "",
        "## Headline",
        "",
        f"- A planner that learns one 3-number curve from its own games (how many more guesses it needs with m "
        f"words still possible) and plans one guess ahead averages **{planner['mean']:.4f} ± {planner['ci']:.4f}** "
        f"guesses over all {len(exam_words):,} answers ({planner['runs']} independent training runs of "
        f"{settings['games']['planner']} games each).",
        (f"- The proven optimum is **{OPTIMUM}** (any valid guess allowed; Bertsimas & Paskov 2022, Selby 2022), so "
         f"the learned planner is **{gap:+.4f} guesses ({100 * gap / OPTIMUM:+.2f}%)** from perfect play. "
         if len(exam_words) == 2315 else
         f"- This run only played a random subset of {len(exam_words)} answers, so it can't be compared with the "
         f"proven optimum ({OPTIMUM}, over all 2,315 answers). "
         ) +
        f"Worst game over all runs: {planner['worst']} guesses; {100 * planner['within6']:.2f}% solved within 6.",
        f"- It gets within 0.05 guesses of its final skill after about **{within} game{'' if within == 1 else 's'}** "
        "of practice.",
        f"- Thinking further ahead at test time (look-ahead: for its top {width} guesses each turn, the exact "
        f"expected number of guesses if it plays on; opener {lookahead[width]['opener'].upper()}) brings the median "
        f"planner to **{ahead['mean']:.4f}**"
        + (f", **{ahead_gap:+.4f} ({100 * ahead_gap / OPTIMUM:+.2f}%)** from the proven optimum" if full_list else "")
        + f" (worst game {ahead['worst']}; {lookahead[width]['seconds'] / 60:.1f} minutes for every answer). "
        f"Against the same planner without look-ahead: {lookahead_gain[0]:+.4f} guesses per game"
        + (", an exact difference: over all the answers it plans for, look-ahead can't do worse than its planner. "
           if full_list else " on this subset (look-ahead is only guaranteed to be no worse over ALL answers). ")
        + f"Treating the answers as a sample of possible words, the 95% bootstrap CI is "
        f"[{lookahead_gain[1]:+.4f}, {lookahead_gain[2]:+.4f}]: one-guess differences on single words are large "
        "next to a 0.01 average, so a new word list could shift it.",
        "",
        "## Setup",
        "",
        "- Word lists: the original 2,315 possible answers, and 12,972 valid guesses (answers + 10,657 others).",
        "- No guess limit: every game continues until solved; the score is the number of guesses.",
        "- Protocol: each strategy plays every answer exactly once, using its best guess every turn, with fixed "
        "tie-breaks, so there is no luck in a score. Learners are trained from scratch with different seeds "
        "(each practice game uses a uniformly random secret); ± values are 95% confidence intervals across seeds.",
        f"- Learning curves use a fixed set of {len(check_words)} check words.",
        "",
        "## How well each strategy plays",
        "",
        "| Strategy | Runs | Average guesses | Worst | Solved within 6 |",
        "|---|---|---|---|---|",
    ]
    for row in ladder:
        ci = f" ± {row['ci']:.4f}" if row["runs"] > 1 else ""
        lines.append(f"| {row['strategy']} | {row['runs']} | {row['mean']:.4f}{ci} | {row['worst']} | "
                     f"{100 * row['within6']:.2f}% |")
    lines += [
        f"| *Proven optimum, any valid guess (literature)* | | *{OPTIMUM}* | *5* | *100%* |",
        f"| *Proven optimum, hard mode (literature)* | | *{HARD_MODE_OPTIMUM}* | | |",
        "",
        "![ladder](ladder.png)",
        "",
        "## Learning curves",
        "",
        "![learning curves](learning_curves.png)",
        "",
        f"These are averages over the {len(check_words)} check words only, so they can't be compared with the "
        f"{OPTIMUM} optimum over all 2,315 answers; the table above can.",
        "",
        "| Games trained | " + " | ".join(curves) + " |",
        "|---|" + "---|" * len(curves),
    ]
    all_games = sorted({int(g) for games, _, _ in curves.values() for g in games})
    for g in all_games:
        cells = []
        for games, mean, ci in curves.values():
            idx = np.flatnonzero(games == g)
            cells.append(f"{mean[idx[0]]:.3f} ± {ci[idx[0]]:.3f}" if len(idx) else "")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired comparisons (same answers, planner minus other)",
        "",
        "Negative means the planner needs fewer guesses. 95% bootstrap CI over the answers.",
        "",
        "| Compared with | Difference in average guesses | 95% CI |",
        "|---|---|---|",
    ]
    for other, diff, lo, hi in comparisons:
        lines.append(f"| {LABELS[other]} | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += [
        "",
        "## Openers",
        "",
        f"The learned planner's own opener is **{own_opener.upper()}**. Below, its first guess is forced to each of "
        f"its top {len(top)} choices and to openers from the literature; every other guess is the planner's.",
        "",
        "| Opener | Average guesses (all answers) | Worst |",
        "|---|---|---|",
    ]
    lines += [f"| {o.upper()}{' (its choice)' if o == own_opener else ''} | {m:.4f} | {w} |"
              for o, m, w in opener_table]
    lines += [
        "",
        "![openers](openers.png)",
        "",
        f"Why not find the best opener by trial and error? Single games vary with a standard deviation of "
        f"{spread:.2f} guesses, so telling apart two openers that differ by 0.05 guesses would take about "
        f"**{games_to_tell_apart(spread, 0.05):,} games with each** (95% confidence, 80% power), and "
        f"{games_to_tell_apart(spread, 0.02):,} each for a 0.02 difference. Scoring every opener exactly on all "
        "answers, as above, removes luck entirely.",
        "",
        "## Emergent strategies",
        "",
        "Nobody tells the planner when to play a word that can't be the answer (a probe). How often it did, over "
        "every answer:",
        "",
        "| Words possible before the guess | Probes |",
        "|---|---|",
    ]
    lines += [f"| {label} | {100 * share:.0f}% ({note}) |" for label, share, note in probing]
    lines += ["", "![probing](probing.png)", "", "Example games (letter families and near-misses):", ""]
    for word, steps in traces.items():
        parts = []
        for s in steps:
            tag = " (probe)" if s["probe"] else ""
            parts.append(f"{s['guess'].upper()}{tag} -> {s['words_left']}")
        lines.append(f"- **{word.upper()}**: " + ", ".join(parts))
    lines += [
        "",
        "## What makes a word hard",
        "",
        f"Played by the median planner; average {difficulty['avg_guesses']:.4f}, opener "
        f"{difficulty['opener'].upper()}.",
        "",
        "| Trait | Harder group | Easier group | Gap |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c['title']} | {c['hard_label']}: {c['hard']:.2f} | {c['easy_label']}: {c['easy']:.2f} | "
              f"{c['gap']:.2f} |" for c in difficulty["conclusions"]]
    lines += [
        "",
        f"Linear regression of guesses per word on the traits (R² = {r2:.3f}; the rest is which colors "
        "come up along the way):",
        "",
        "| Term | Coefficient | 95% CI |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {coef:+.4f} | ± {ci:.4f} |" for name, coef, ci in regression]
    lines += [
        "",
        "## Ablation (planner variants)",
        "",
        "| Variant | Average guesses |",
        "|---|---|",
    ]
    lines += [f"| {name} | {m:.4f}" + (f" ± {c:.4f}" if c == c else "") + " |" for name, m, c in ablation_rows]
    lines += [
        "",
        "## Related work and how to read these numbers",
        "",
        "- Exact optimum: 3.4201 average with SALET, proven by exhaustive search (Bertsimas & Paskov, *An Exact "
        "and Interpretable Solution to Wordle*, 2022; A. Selby, 2022). Hard mode: about 3.5076. Summary: "
        "https://www.poirrier.ca/notes/wordle-optimal/",
        "- Reinforcement learning: rollout methods get near-optimal (Bhambri, Bhattacharjee & Bertsekas, "
        "arXiv:2211.10298); a deep RL agent reached about 3.9 guesses after hundreds of thousands of games "
        "(A. Ho, https://andrewkho.github.io/wordle-solver/).",
        "- Word difficulty: see arXiv:2305.03502 for a study of which word attributes make Wordle words hard.",
        "- Limits: the planner is told the rules and can compute how any guess splits the remaining answers "
        "(one-step lookahead); what it learns is the value curve V(m). The 'split' idea is hand-designed, so the "
        "honest claim is sample efficiency and interpretability with a tiny learned model, not learning from raw "
        "pixels or letters.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
