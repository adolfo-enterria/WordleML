"""How does word length change the game?   python benchmark.py --length-study

For every length 3-8 (the "common words" lists, built the same way for every
length) it trains the planner a few times from scratch and plays every answer
once with: no strategy (a random word that could be the answer), the hand-made
max-entropy rule, the learned planner, and the planner with look-ahead.

List sizes differ by length (natural lists), so besides the average number of
guesses it reports "bits per guess": log2(possible answers) / average guesses,
i.e. how much each guess narrows things down. That separates "longer words
give more information per guess" from "some lengths simply have more words".

Output in reports/<date-time>_lengths/: report.md, length_study.csv, length_study.png.
"""
import csv
import datetime
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analysis import benchmark_tasks as tasks
from analysis import figures
from analysis.stats import mean_ci
from wordle.patterns import pattern_table
from wordle.words import LENGTHS, load_word_set, word_set_name

FULL = {"seeds": 5, "games": 300, "random_seeds": 5, "width": 10, "subset": None}
QUICK = {"seeds": 1, "games": 30, "random_seeds": 1, "width": 5, "subset": 150}
STRATEGIES = [("random", "No strategy (random possible word)"),
              ("greedy", "Hand-made max-entropy"),
              ("planner", "Learned planner"),
              ("lookahead", "Planner + look-ahead")]


def run(workers, quick=False, out=None):
    settings = QUICK if quick else FULL
    started = time.time()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(out or f"reports/{stamp}_lengths{'_quick' if quick else ''}")
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(f"[{time.time() - started:6.0f}s] {msg}", flush=True)  # noqa: E731

    names = [word_set_name(length, official=False) for length in LENGTHS]
    exams = {}
    for name in names:  # build the pattern tables once, here, rather than in every worker
        pattern_table(name)
        answers = list(load_word_set(name).answers)
        if settings["subset"]:
            answers = sorted(str(w) for w in np.random.default_rng(7).choice(
                answers, min(settings["subset"], len(answers)), replace=False))
        exams[name] = answers

    rows = {}

    def evaluate(pool, spec, words, chunks):
        parts = [list(map(str, p)) for p in np.array_split(words, chunks) if len(p)]
        return [pool.apply_async(tasks.evaluate_chunk, (spec, part)) for part in parts]

    # One word length at a time: workers switch word lists (and their big tables) only once per length.
    with Pool(workers, initializer=tasks.init_worker, initargs=(names[0],)) as pool:
        for name in names:
            words = exams[name]
            log(f"{name}: training the planner ({settings['seeds']} runs x {settings['games']} games)...")
            trained = [pool.apply_async(tasks.train, ("planner", seed, settings["games"], [], [], None, name))
                       for seed in range(settings["seeds"])]
            trained = [job.get()["state"] for job in trained]

            log(f"{name}: playing every answer with every strategy...")
            jobs = {}
            for seed, state in enumerate(trained):
                jobs[("planner", seed)] = evaluate(pool, {"kind": "planner", "state": state, "word_set": name},
                                                   words, 2)
            jobs[("greedy", 0)] = evaluate(pool, {"kind": "greedy", "criterion": "entropy", "word_set": name},
                                           words, workers)
            for seed in range(settings["random_seeds"]):
                jobs[("random", seed)] = evaluate(pool, {"kind": "random", "seed": seed, "word_set": name}, words, 1)
            for (kind, seed), parts in jobs.items():
                rows[(name, kind, seed)] = [row for part in parts for row in part.get()]

            scores = sorted((np.mean([r["guesses"] for r in rows[(name, "planner", s)]]), s)
                            for s in range(settings["seeds"]))
            median_state = trained[scores[len(scores) // 2][1]]
            log(f"{name}: look-ahead...")
            result = pool.apply(tasks.lookahead_exam, (median_state, settings["width"], words, name))
            rows[(name, "lookahead", 0)] = result["rows"]
            log(f"{name}: done (look-ahead {np.mean([r['guesses'] for r in result['rows']]):.3f}, "
                f"{result['seconds']:.0f} s)")

    table = []
    for name in names:
        ws = load_word_set(name)
        entry = {"length": ws.length, "answers": len(ws.answers), "guesses": len(ws.guesses),
                 "exam_words": len(exams[name])}
        for kind, _ in STRATEGIES:
            runs = [np.mean([r["guesses"] for r in rs]) for (n, k, _), rs in rows.items() if n == name and k == kind]
            mean, ci = mean_ci(runs)
            worst = max(max(r["guesses"] for r in rs) for (n, k, _), rs in rows.items() if n == name and k == kind)
            entry[kind] = (mean, ci if len(runs) > 1 else None, worst)
        entry["bits_per_guess"] = np.log2(entry["answers"]) / entry["lookahead"][0]
        table.append(entry)

    with open(out / "length_study.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "answers", "guesses"] + [k for k, _ in STRATEGIES] + ["bits_per_guess"])
        for e in table:
            writer.writerow([e["length"], e["answers"], e["guesses"]] + [f"{e[k][0]:.4f}" for k, _ in STRATEGIES]
                            + [f"{e['bits_per_guess']:.3f}"])
    figures.length_study(table, STRATEGIES, out / "length_study.png")
    (out / "report.md").write_text(render(table, settings, quick, time.time() - started), encoding="utf-8")
    log(f"done: {out / 'report.md'}")
    return out / "report.md"


def render(table, settings, quick, seconds):
    lines = [
        "# How word length changes the game",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} in {seconds / 60:.0f} minutes"
        + (" (QUICK smoke test on a subset of answers, not for citing)." if quick else "."),
        "Regenerate with `python benchmark.py --length-study`.",
        "",
        "## Setup",
        "",
        "- Word lists: common English words of each length, built by `python -m wordle.wordlists` with one "
        "recipe (dictionary words among the most frequent English words, simple plurals dropped; the cutoff "
        "is set so that 5 letters gives 2,315 answers, as many as the official list, though not the same words). "
        "Allowed guesses: every dictionary word "
        "of that length, up to 15,000.",
        "- Every answer played once, no guess limit, best guesses with fixed tie-breaks.",
        f"- Learned planner: {settings['seeds']} independent training runs of {settings['games']} games per length "
        "(± = 95% CI across runs; ± 0.000: the runs' averages agree to 3 decimals). Look-ahead: the "
        f"median run, weighing its top {settings['width']} guesses exactly each turn. No strategy: "
        f"{settings['random_seeds']} random seeds. Max-entropy has no randomness, so it's run once.",
        "- These lists are not the official Wordle lists, so the 3.4201 optimum does not apply to any row here.",
        "",
        "## Results",
        "",
        "| Letters | Answers | Allowed guesses | " + " | ".join(label for _, label in STRATEGIES)
        + " | Bits per guess |",
        "|---|---|---|" + "---|" * len(STRATEGIES) + "---|",
    ]
    for e in table:
        cells = []
        for kind, _ in STRATEGIES:
            mean, ci, worst = e[kind]
            cells.append(f"{mean:.3f}" + (f" ± {ci:.3f}" if ci is not None else "") + f" (worst {worst})")
        lines.append(f"| {e['length']} | {e['answers']:,} | {e['guesses']:,} | " + " | ".join(cells)
                     + f" | {e['bits_per_guess']:.2f} |")
    lines += [
        "",
        "![length study](length_study.png)",
        "",
        "Bits per guess = log2(number of possible answers) / average guesses with look-ahead: how many "
        "halvings of the list each guess is worth on average. It doesn't depend on how many words a length "
        "happens to have, so it's the fairer way to compare lengths.",
        "",
    ]
    return "\n".join(lines)
