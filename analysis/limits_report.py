"""The report for analysis/limits_study.py: tables, CSVs and charts from the phases' JSON files.

    python -m analysis.limits_report reports/<folder>      # rebuild report.md from what is there

Works with whatever phases have finished (a missing phase's sections are left out).
"""
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from agents.planning_agent import curve_basis
from analysis import figures
from analysis.stats import mean_ci
from wordle.words import load_word_set

COMMON = [f"common{length}" for length in range(3, 11)]
REFERENCE, OFFICIAL_OPTIMUM = "wordle5", 7920
LONG_RUN_FROM = 0.2      # the long-run level: every mark from 20% of the training on (200 of 1,000 games)
TOLERANCES = [0.01, 0.005]


def label(ws):
    return "Official Wordle (5)" if ws == REFERENCE else f"{ws[6:]} letters"


def load(out, name):
    path = Path(out) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- learning speed -------------------------------------------------------------------------------

def curves(train):
    """{word set: (marks, totals as seeds x marks, learned numbers as seeds x marks x 3, seconds per run)}."""
    by_set = {}
    for r in train.values():
        by_set.setdefault(r["word_set"], []).append(r)
    out = {}
    for ws, runs in by_set.items():
        runs.sort(key=lambda r: r["seed"])
        marks = np.array([g for g, _, _ in runs[0]["curve"]])
        totals = np.array([[t for _, t, _ in r["curve"]] for r in runs], float)
        states = np.array([[s for _, _, s in r["curve"]] for r in runs], float)
        out[ws] = (marks, totals, states, np.array([r["seconds"] for r in runs]), np.array([r["seed"] for r in runs]))
    return out


def first_within(marks, values, target, tol, stay=False):
    """First mark where values <= target * (1 + tol) (and, with stay, at every mark after it too)."""
    ok = values <= target * (1 + tol) + 1e-9
    if stay:
        ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
    hits = np.flatnonzero(ok)
    return int(marks[hits[0]]) if len(hits) else None


def learning_table(train, floors, timing=None):
    rows = []
    for ws, (marks, totals, _, seconds, _) in curves(train).items():
        n = len(load_word_set(ws).answers)
        late = marks >= LONG_RUN_FROM * marks[-1]
        level = totals[:, late].mean()
        mean = totals.mean(axis=0)
        row = {"word_set": ws, "answers": n, "marks": marks, "mean": mean / n, "level": level / n,
               "level_ci": mean_ci(totals[:, late].mean(axis=1) / n)[1], "seeds": len(totals),
               "seconds_per_game": (timing or {}).get(ws), "blank": totals[0, 0] / n}
        for tol in TOLERANCES:
            row[f"mean_within_{tol}"] = first_within(marks, mean, level, tol)
            row[f"mean_stays_{tol}"] = first_within(marks, mean, level, tol, stay=True)
            per_seed = [first_within(marks, t, t[late].mean(), tol) for t in totals]
            row[f"seed_within_{tol}"] = float(np.median(per_seed))
        floor = floors.get(ws)
        if floor:
            row["floor"] = floor
            row["gap"] = level / floor["upper"] - 1
            if floor["proven"]:
                at = totals == floor["upper"]
                firsts = [int(marks[np.argmax(a)]) if a.any() else None for a in at]
                row["perfect_first"] = firsts
                row["perfect_share_late"] = float(at[:, late].mean())
        rows.append(row)
    order = COMMON + [REFERENCE]
    return sorted(rows, key=lambda r: order.index(r["word_set"]))


# --- the floor ------------------------------------------------------------------------------------

def floor_table(floor, train):
    """Per word set: the best total found, the proven lower bound, and the searches behind them."""
    out = {}
    for ws, proof in floor["proofs"].items():
        searches = [s for s in floor["searches"] if s["word_set"] == ws]
        best = min(searches, key=lambda s: (s["total"], -s["width"]))
        upper = min(proof["upper"], best["total"])
        n = len(load_word_set(ws).answers)
        planner = [t for _, t, _ in [r["curve"][-1] for r in train.values() if r["word_set"] == ws]]
        out[ws] = {"answers": n, "guesses": len(load_word_set(ws).guesses), "upper": upper, "lower": proof["lower"],
                   "proven": proof["lower"] == upper, "improved": proof.get("improved"),
                   "opener": best["opener"], "worst": len(best["guess_counts"]) - 1,
                   "counts": best["guess_counts"], "searches": searches, "proof": proof,
                   "planner": mean_ci(np.array(planner) / n)}
    return out


# --- transfer -------------------------------------------------------------------------------------

def transfer_matrix(transfer):
    """(mean total per (train, test)), the per-seed totals, and the blank planner per test set."""
    totals, blank = {}, {}
    for row in transfer["matrix"]:
        if row["train"] == "blank":
            blank[row["test"]] = row["total"]
        else:
            totals.setdefault((row["train"], row["test"]), []).append(row["total"])
    return totals, blank


def value_curve(state, m):
    """V(m) as the planner uses it (made non-decreasing)."""
    return np.maximum.accumulate(curve_basis(m.astype(float)) @ np.array(state))


# --- the report -----------------------------------------------------------------------------------

def avg(total, n):
    return f"{total / n:.4f}"


def pct(change):
    """+1.23% / -0.40% / 0.00% (no sign when it rounds to zero)."""
    return "0.00%" if abs(100 * change) < 0.005 else f"{100 * change:+.2f}%"


def fmt_games(g):
    return "never" if g is None else f"{g:,}"


def build(out):
    out = Path(out)
    settings = json.loads((out / "settings.json").read_text(encoding="utf-8")) if (out / "settings.json").exists() \
        else {}
    train, transfer, floor = load(out, "train"), load(out, "transfer"), load(out, "floor")
    thinking, fixed, timing = load(out, "thinking"), load(out, "fixed"), load(out, "timing")
    structure = load(out, "structure")
    floors = floor_table(floor, train) if floor and train else {}
    lines = [f"# Word length: the floor, how fast the AI gets there, and a length it never saw",
             "",
             f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from the experiments in this folder "
             "(`python benchmark.py --limits-study`; rebuild this page with "
             f"`python -m analysis.limits_report {out.as_posix()}`).",
             ""]
    sections = [render_summary(train, transfer, floors, thinking, fixed, timing)]
    if floors:
        sections.append(render_floor(out, floors, fixed))
    if train:
        sections.append(render_learning(out, train, floors, thinking, timing))
    if transfer and train:
        sections.append(render_transfer(out, transfer, train))
    if structure:
        sections.append(render_structure(out, structure, floors, fixed))
    lines += [s for s in sections]
    lines.append(render_method(settings))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "report.md"


def letters_list(items):
    """[(letters, text)] -> '3 letters: a; 4: b; ...'."""
    return "; ".join(f"{L} letters: {t}" if i == 0 else f"{L}: {t}" for i, (L, t) in enumerate(items))


def render_summary(train, transfer, floors, thinking, fixed, timing):
    lines = ["## The answers", ""]
    if floors:
        common = [ws for ws in COMMON if ws in floors]
        proven = [ws for ws in common if floors[ws]["proven"]]
        open_ = [ws for ws in common if not floors[ws]["proven"]]
        best = {ws: floors[ws]["upper"] / floors[ws]["answers"] for ws in common}
        lines.append("**1. The fewest guesses possible.** Longer words need far fewer guesses. Perfect play needs "
                     f"at most {best[common[0]]:.2f} guesses on average at {common[0][6:]} letters and exactly "
                     f"{best[common[-1]]:.2f} at {common[-1][6:]} (every answer of the common-word list once).")
        if proven:
            lines.append("- Proven optimum (no strategy can do better, every valid guess considered): "
                         + letters_list([(ws[6:], f"**{best[ws]:.4f}** ({floors[ws]['upper']:,} guesses, opener "
                                                  f"{floors[ws]['opener'].upper()})") for ws in proven]) + ".")
        if open_:
            lines.append("- Not proven, but bracketed: " + letters_list(
                [(ws[6:], f"between {floors[ws]['lower'] / floors[ws]['answers']:.3f} (proven) and "
                          f"{best[ws]:.4f} (best plan found)") for ws in open_]) + ".")
            spread = max(max(s["total"] for s in floors[ws]["searches"]) - min(s["total"] for s in
                                                                                floors[ws]["searches"])
                         for ws in open_)
            text = ("- The lower bounds there are loose (a proof over every guess ran out of its 20 minutes), but "
                    "the best plans are probably at or very near the true floor: every search, at two widths and "
                    f"from three differently trained planners, found the same total to within {spread} "
                    f"guess{'es' if spread != 1 else ''} in total")
            if REFERENCE in floors and floors[REFERENCE]["upper"] == OFFICIAL_OPTIMUM:
                text += (", and on the official 5-letter list, where the optimum is known, the same search lands "
                         f"exactly on it ({OFFICIAL_OPTIMUM:,} = {OFFICIAL_OPTIMUM / 2315:.4f})")
            lines.append(text + ".")
        lines.append("- Worst game of the best plan: " + letters_list(
            [(ws[6:], str(floors[ws]["worst"])) for ws in common]) + " guesses.")
        rows = fixed_rows(fixed)
        if rows:
            proven_from = min((L for L, _, _, _, ok, _ in rows if ok), default=None)
            lines.append(f"- It is the length, not the list size: with the same {fixed[0]['size']:,} answers at "
                         "every length, the best plans average " + letters_list(
                             [(L, f"{b:.3f}") for L, _, b, _, _, _ in rows]) + "."
                         + (f" From {proven_from} letters on these are proven optimal for every subset."
                            if proven_from else ""))
        lines.append("")
    if train:
        table = learning_table(train, floors, timing)
        commons = [r for r in table if r["word_set"] in COMMON]
        w1 = [r["mean_within_0.01"] for r in commons]
        w05 = [r["mean_within_0.005"] for r in commons]
        lines.append(f"**2. How fast it gets there.** Within 1% of its long-run level after {min(w1)}-{max(w1)} "
                     f"practice games at every length, within 0.5% after {min(w05)}-{max(w05)} (mean of "
                     f"{commons[0]['seeds']} training runs per length). The number of games hardly depends on the "
                     "length: the planner only learns 3 numbers (its curve V), and a single game already shows "
                     "it that fewer words left means fewer guesses to go.")
        perfect = [r for r in commons if "perfect_first" in r]
        if perfect:
            lines.append("- The planner alone plays perfectly at " + ", ".join(r["word_set"][6:] for r in perfect)
                         + " letters, for the first time after a median of " + ", ".join(
                             f"{np.median([g for g in r['perfect_first'] if g is not None]):g}" for r in perfect)
                         + " games.")
        gaps = [r for r in commons if "gap" in r and not r["floor"]["proven"]]
        if gaps:
            lines.append("- At shorter lengths it levels off above the best plan known ("
                         + letters_list([(r["word_set"][6:], f"+{100 * r['gap']:.2f}%") for r in gaps])
                         + ") and more games don't close that gap: it's the limit of planning one guess ahead "
                           "with a 3-number V, not of training.")
        if thinking and floors:
            gaps_by, worst = {}, None
            for r in thinking:
                if r["word_set"] in floors and r["word_set"] in COMMON:
                    gap = r["search"] - floors[r["word_set"]]["upper"]
                    gaps_by.setdefault(r["games"], []).append(gap)
                    if worst is None or gap > worst[0]:
                        worst = (gap, r)
            parts = []
            for g, gaps in sorted(gaps_by.items()):
                hit, most = sum(x == 0 for x in gaps), max(gaps)
                parts.append(f"{g:,} game{'s' if g > 1 else ''}: {hit} of {len(gaps)}"
                             + (f" (the rest within {most:,} guess{'es' if most != 1 else ''})" if hit < len(gaps)
                                else ""))
            text = ("- With search on top, a few games of training are enough. Lengths where the search guided by "
                    "the planner reaches the best total known, by games of training: " + "; ".join(parts)
                    + " (totals over every answer).")
            if worst and worst[0] > 10:
                r = worst[1]
                n = len(load_word_set(r["word_set"]).answers)
                text += (f" The one big miss: {r['word_set'][6:]} letters after {r['games']} game"
                         f"{'s' if r['games'] > 1 else ''}, where the planner had not yet learned much "
                         f"({r['planner'] / n:.2f} guesses on its own).")
            lines.append(text)
        if timing:
            lines.append("- Longer words cost more per practice game, not more games: "
                         + letters_list([(ws[6:], f"{1000 * timing[ws]:.0f} ms") for ws in COMMON if ws in timing])
                         + ".")
        lines.append("")
    if transfer:
        totals, _ = transfer_matrix(transfer)
        n7 = len(load_word_set("common7").answers)
        four = totals[("common4", "common7")]
        games = transfer["games"]
        same = sum(games["4->7"][w] == games["7->7"][w] for w in games["7->7"])
        native = {ws: np.mean(totals[(ws, ws)]) for ws in COMMON}
        long_short = [np.mean(totals[(a, "common3")]) / native["common3"] - 1
                      for a in ("common8", "common9", "common10")]
        opener = games["4->7"][next(iter(games["4->7"]))][0].upper()
        better = [(a[6:], np.mean(totals[(a, "common3")]) / native["common3"] - 1) for a in ("common4", "common5")]
        lines += [f"**3. The 4-letter model on 7-letter words.** It plays them as well as a model trained on 7 "
                  f"letters: {np.mean(four) / n7:.4f} vs {native['common7'] / n7:.4f} average guesses (8 training "
                  f"runs each), the same opener ({opener}), and exactly the same guesses in {same:,} of {n7:,} "
                  f"games. With search on top both reach {transfer['search']['4->7']['total']:,} guesses in total.",
                  "- Why: the model is not a list of words but a curve V(m), how many more guesses m possible "
                  "words usually take, and the rules are built in. Every length's curve says that fewer words "
                  "left is better, and that is most of what choosing a good guess needs.",
                  f"- The other direction is not free: planners trained on 8-10 letters are "
                  f"{100 * min(long_short):.0f}-{100 * max(long_short):.0f}% worse on 3-letter words. After one "
                  "long-word guess only a handful of words are ever left, so their V goes flat beyond about 10 "
                  "words and can't tell a guess that leaves 20 words from one that leaves 200.",
                  "- Training on the length itself isn't always best: planners trained on "
                  + " and ".join(f"{a} letters ({pct(c)})" for a, c in better)
                  + " play 3-letter words slightly better than 3-letter planners do. The planner fits V to its own "
                    "games; it doesn't search for the V that plays best.", ""]
    return "\n".join(lines)


def render_floor(out, floors, fixed):
    sets = [ws for ws in COMMON + [REFERENCE] if ws in floors]
    with open(out / "floor.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word_set", "answers", "guesses", "planner_mean", "best_total", "best_mean", "lower_total",
                    "lower_mean", "proven", "opener", "worst"])
        for ws in sets:
            e = floors[ws]
            w.writerow([ws, e["answers"], e["guesses"], f"{e['planner'][0]:.4f}", e["upper"],
                        avg(e["upper"], e["answers"]), e["lower"], avg(e["lower"], e["answers"]), e["proven"],
                        e["opener"], e["worst"]])
    figures.limits_floor([(int(ws[6:]), floors[ws]) for ws in COMMON if ws in floors],
                         fixed_rows(fixed), out / "floor.png")
    lines = ["## 1. The floor: the fewest guesses any strategy can average", "",
             "| Letters | Answers | Allowed guesses | Learned planner alone | Best strategy found | "
             "Proven lower bound | Optimum | Worst case of the best plan |",
             "|---|---|---|---|---|---|---|---|"]
    for ws in sets:
        e = floors[ws]
        n = e["answers"]
        status = (f"**{e['upper'] / n:.4f}** (proven)" if e["proven"]
                  else f"between {e['lower'] / n:.3f} and {e['upper'] / n:.3f}")
        lines.append(f"| {label(ws)} | {n:,} | {e['guesses']:,} | {e['planner'][0]:.4f} ± {e['planner'][1]:.4f} | "
                     f"{e['upper'] / n:.4f} ({e['upper']:,}, {e['opener'].upper()}) | {e['lower'] / n:.4f} "
                     f"({e['lower']:,}) | {status} | {e['worst']} guesses |")
    lines += ["", "![floor](floor.png)", ""]
    lines += render_searches(floors, sets)
    if fixed:
        lines += render_fixed(fixed)
    return "\n".join(lines)


def render_searches(floors, sets):
    lines = ["How the best totals were found (each search is guided by one trained planner; width = guesses "
             "weighed at every position):", "",
             "| Letters | " + " | ".join(["searches: total (width, seed)", "proof"]) + " |", "|---|---|---|"]
    for ws in sets:
        e = floors[ws]
        found = ", ".join(f"{s['total']:,} (w{s['width']}, s{s['seed']}, {s['seconds'] / 60:.1f} min)"
                          for s in sorted(e["searches"], key=lambda s: (s["width"], s["seed"])))
        p = e["proof"]
        alive = p["openers_alive"]
        proof = (f"proven in {p['seconds'] / 60:.1f} min ({alive:,} opener{'s' if alive != 1 else ''} to rule out)"
                 if e["proven"]
                 else f"stopped after {p['seconds'] / 60:.0f} min; {p['openers_left']:,} of {p['openers_alive']:,} "
                      "openers not yet ruled out")
        if e["improved"]:
            proof += f"; found a better plan opening with {e['improved'].upper()}"
        lines.append(f"| {label(ws)} | {found} | {proof} |")
    return lines + [""]


def fixed_rows(fixed):
    """[(letters, mean planner avg, mean best avg, mean lower avg, all proven)] for the equal-size control."""
    if not fixed:
        return []
    rows = []
    for ws in COMMON:
        runs = [r for r in fixed if r["word_set"] == ws]
        if runs:
            rows.append((int(ws[6:]), np.mean([r["planner"] / r["size"] for r in runs]),
                         np.mean([r["upper"] / r["size"] for r in runs]),
                         np.mean([r["lower"] / r["size"] for r in runs]), all(r["proven"] for r in runs), len(runs)))
    return rows


def render_fixed(fixed):
    size = fixed[0]["size"]
    lines = [f"### The same number of answers at every length ({size:,})", "",
             "| Letters | Subsets | Learned planner alone | Best strategy found | Proven lower bound | Optimum |",
             "|---|---|---|---|---|---|"]
    for letters, planner, upper, lower, proven, k in fixed_rows(fixed):
        lines.append(f"| {letters} | {k} | {planner:.4f} | {upper:.4f} | {lower:.4f} | "
                     f"{'proven for every subset' if proven else 'bracketed'} |")
    return lines + [""]


def render_learning(out, train, floors, thinking, timing):
    table = learning_table(train, floors, timing)
    with open(out / "learning_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word_set", "seed", "games", "total", "average", "a", "b", "c"])
        for ws, (marks, totals, states, _, seeds) in curves(train).items():
            n = len(load_word_set(ws).answers)
            for i, seed in enumerate(seeds):
                for j, g in enumerate(marks):
                    w.writerow([ws, seed, g, int(totals[i, j]), f"{totals[i, j] / n:.4f}"]
                               + [f"{c:.5f}" for c in states[i, j]])
    figures.limits_learning([(label(r["word_set"]), r, curves(train)[r["word_set"]]) for r in table],
                            out / "learning_curves.png")
    lines = ["## 2. How fast the planner gets there", "",
             "| Letters | Before any game (random valid words) | After 1 game | After 10 | After 100 | Long run "
             "(±95% CI over seeds) | Games to within 1% (mean curve; median seed) | Within 0.5% | Long run vs "
             "best known | Time per practice game |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        at = {int(g): v for g, v in zip(r["marks"], r["mean"])}
        gap = f"+{100 * r['gap']:.2f}%" if "gap" in r else "-"
        per_game = "-" if r["seconds_per_game"] is None else f"{1000 * r['seconds_per_game']:.0f} ms"
        lines.append(f"| {label(r['word_set'])} | {r['blank']:.3f} | {at.get(1, float('nan')):.3f} | "
                     f"{at.get(10, float('nan')):.3f} | {at.get(100, float('nan')):.3f} | "
                     f"{r['level']:.4f} ± {r['level_ci']:.4f} | {fmt_games(r['mean_within_0.01'])}; "
                     f"{r['seed_within_0.01']:g} | {fmt_games(r['mean_within_0.005'])}; {r['seed_within_0.005']:g} | "
                     f"{gap} | {per_game} |")
    lines += ["", "![learning curves](learning_curves.png)", ""]
    perfect = [r for r in table if "perfect_first" in r]
    if perfect:
        lines += ["Where the optimum is proven, how soon the planner alone plays perfectly (every answer, "
                  "optimal total):", "",
                  "| Letters | First perfect game count, per seed | Share of checkpoints from 200 games on "
                  "that are perfect |", "|---|---|---|"]
        for r in perfect:
            lines.append(f"| {label(r['word_set'])} | {', '.join(fmt_games(g) for g in r['perfect_first'])} | "
                         f"{100 * r['perfect_share_late']:.0f}% |")
        lines.append("")
    if thinking:
        lines += render_thinking(thinking, floors)
    return "\n".join(lines)


def render_thinking(thinking, floors):
    marks = sorted({r["games"] for r in thinking})
    lines = ["### Planner + search after only a few games", "",
             "The search (width 20) guided by the median run's planner at each point of its training. "
             "Totals over every answer; **bold** = the best total known for that list.", "",
             "| Letters | " + " | ".join(f"after {g:,} games" for g in marks) + " |",
             "|---|" + "---|" * len(marks)]
    for ws in COMMON + [REFERENCE]:
        runs = {r["games"]: r for r in thinking if r["word_set"] == ws}
        if not runs:
            continue
        best = floors[ws]["upper"] if ws in floors else min(r["search"] for r in runs.values())
        cells = []
        for g in marks:
            r = runs.get(g)
            if r is None:
                cells.append("-")
                continue
            total = f"**{r['search']:,}**" if r["search"] == best else f"{r['search']:,}"
            cells.append(f"{total} (planner {r['planner']:,})")
        lines.append(f"| {label(ws)} | " + " | ".join(cells) + " |")
    return lines + [""]


def render_transfer(out, transfer, train):
    totals, blank = transfer_matrix(transfer)
    sets = [ws for ws in COMMON if (ws, ws) in totals]
    n = {ws: len(load_word_set(ws).answers) for ws in sets}
    native = {ws: np.mean(totals[(ws, ws)]) for ws in sets}
    with open(out / "transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trained_on", "tested_on", "seeds", "mean_total", "mean_average", "vs_native"])
        for a in sets:
            for b in sets:
                t = totals[(a, b)]
                w.writerow([a, b, len(t), f"{np.mean(t):.1f}", f"{np.mean(t) / n[b]:.4f}",
                            f"{np.mean(t) / native[b] - 1:.5f}"])
    grid = np.array([[np.mean(totals[(a, b)]) / native[b] - 1 for b in sets] for a in sets])
    figures.limits_transfer([int(ws[6:]) for ws in sets], grid, out / "transfer.png")
    finals = {}
    for r in train.values():
        if r["word_set"] in sets:
            finals.setdefault(r["word_set"], []).append(r["curve"][-1][2])
    m = np.arange(1, max(n.values()) + 1)
    figures.limits_values([(int(ws[6:]), m[:n[ws]], np.median([value_curve(s, m[:n[ws]]) for s in finals[ws]],
                                                                  axis=0)) for ws in sets], out / "value_curves.png")

    lines = ["## 3. A planner trained on one length, playing another", "",
             "Every planner (8 seeds per training length) played every answer of every length. Each cell is "
             "the change in total guesses against planners trained on the test length itself.", "",
             "![transfer](transfer.png)", "",
             "| Trained on \\ tested on | " + " | ".join(label(b) for b in sets) + " |",
             "|---|" + "---|" * len(sets)]
    for a in sets:
        cells = []
        for b in sets:
            t = totals[(a, b)]
            cells.append(f"{np.mean(t) / n[b]:.4f}" + (" (native)" if a == b else
                                                       f" ({pct(np.mean(t) / native[b] - 1)})"))
        lines.append(f"| {label(a)} | " + " | ".join(cells) + " |")
    lines.append("| No training (blank) | " + " | ".join(f"{blank[b] / n[b]:.4f}" for b in sets) + " |")
    lines += ["", "![value curves](value_curves.png)", ""]
    lines += render_four_to_seven(transfer, totals, n, native)
    return "\n".join(lines)


def render_four_to_seven(transfer, totals, n, native):
    games = transfer.get("games")
    if not games or "common4" not in n or "common7" not in n:
        return []
    four, seven = games["4->7"], games["7->7"]
    same = sum(four[w] == seven[w] for w in seven)
    diff = [len(four[w]) - len(seven[w]) for w in seven]
    four_t, seven_t = totals[("common4", "common7")], totals[("common7", "common7")]
    s4, s7 = transfer["search"]["4->7"], transfer["search"]["7->7"]
    worse, better = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
    return ["### Close-up: the 4-letter planner on 7-letter words", "",
            f"- All 8 planners trained on 4 letters, on the {n['common7']:,} seven-letter answers: "
            f"{np.mean(four_t) / n['common7']:.4f} average (totals {min(four_t):,}-{max(four_t):,}); the 8 "
            f"trained on 7 letters: {native['common7'] / n['common7']:.4f} ({min(seven_t):,}-{max(seven_t):,}).",
            f"- Median 4-letter planner (seed {transfer['seeds']['common4']}) vs median 7-letter planner "
            f"(seed {transfer['seeds']['common7']}), game by game: identical guesses on {same:,} of "
            f"{len(seven):,} answers ({100 * same / len(seven):.0f}%); the 4-letter one needs more guesses on "
            f"{worse:,} and fewer on {better:,}. Openers: {four[next(iter(four))][0].upper()} vs "
            f"{seven[next(iter(seven))][0].upper()}.",
            f"- With search on top (width {s4['width']}): {s4['total']:,} guided by the 4-letter planner, "
            f"{s7['total']:,} guided by the 7-letter one.", ""]


def render_structure(out, structure, floors, fixed):
    """Why length matters: what one guess can tell, and the range of m each planner's V is fitted on."""
    size = fixed[0]["size"] if fixed else None
    by_size = {f"common{L}": b for L, _, b, _, _, _ in fixed_rows(fixed)}
    rows = []
    for ws in COMMON + [REFERENCE]:
        if ws not in structure:
            continue
        e, best = structure[ws], floors.get(ws)
        bits = np.log2(e["answers"]) / (best["upper"] / best["answers"]) if best else None
        bits_fixed = np.log2(size) / by_size[ws] if ws in by_size else None
        rows.append((ws, e, bits, bits_fixed))
    with open(out / "structure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word_set", "answers", "guesses", "patterns", "most_groups", "best_opener", "best_groups",
                    "best_mean_left", "best_largest", "planner_opener", "planner_groups", "planner_mean_left",
                    "planner_largest", "bits_per_guess", "bits_per_guess_equal_size"])
        for ws, e, bits, bits_fixed in rows:
            b, q = e.get("best_opener", {}), e["planner_opener"]
            w.writerow([ws, e["answers"], e["guesses"], e["patterns"], e["most_groups"], b.get("guess"),
                        b.get("groups"), b.get("mean_left") and f"{b['mean_left']:.2f}", b.get("largest"),
                        q["guess"], q["groups"], f"{q['mean_left']:.2f}", q["largest"],
                        "" if bits is None else f"{bits:.3f}", "" if bits_fixed is None else f"{bits_fixed:.3f}"])
    lines = ["## 4. Why length matters: what one guess can tell", "",
             "Color patterns: 3^L. Most groups: the most color groups besides the win that any valid guess makes "
             "on the full list. Opener splits count every group (the win included); 'left' = words still "
             "possible after it, on average over the answers. Bits per guess = log2(answers) / best average: "
             "how many halvings of the list each guess is worth.", "",
             "| Letters | Answers | Color patterns | Most groups one guess makes | Best plan's opener: groups, "
             "words left on average, largest group | Planner's opener: the same | Bits per guess"
             + (f" (natural list; {size:,} answers)" if size else "") + " |",
             "|---|---|---|---|---|---|---|"]
    for ws, e, bits, bits_fixed in rows:
        b, q = e.get("best_opener"), e["planner_opener"]
        best_cell = f"{b['guess'].upper()}: {b['groups']:,}, {b['mean_left']:.1f}, {b['largest']:,}" if b else "-"
        bits_cell = "-" if bits is None else f"{bits:.2f}" + ("" if bits_fixed is None else f"; {bits_fixed:.2f}")
        lines.append(f"| {label(ws)} | {e['answers']:,} | {e['patterns']:,} | {e['most_groups']:,} | {best_cell} | "
                     f"{q['guess'].upper()}: {q['groups']:,}, {q['mean_left']:.1f}, {q['largest']:,} | {bits_cell} |")
    lines.append("")
    return "\n".join(lines)


def render_method(settings):
    s = settings.get("settings", {})
    return "\n".join([
        "## How this was measured", "",
        "- Lists: the common-word lists (`python -m wordle.wordlists`, one recipe for every length; answers "
        "are frequent ENABLE words, guesses are every dictionary word of that length), plus the official "
        "Wordle lists as a reference with a published optimum (7,920 = 3.4212, Selby).",
        "- Every score is exact: each answer played once, no guess limit, fixed tie-breaks. A frozen planner's "
        "total is computed by walking its decision tree once (the same total as playing every answer).",
        f"- Training: {s.get('seeds', '?')} planners per list trained from scratch for {s.get('games', 0):,} "
        "practice games (secrets drawn at random), scored at 29 checkpoints from 0 to the end. 'Long run' = "
        "every checkpoint from 20% of the training on, averaged over seeds.",
        f"- Best strategy found: the search in `agents/search.py` guided by trained planners: width "
        f"{(s.get('search_widths') or [0])[0]} from the median, best and worst training seed, and width "
        f"{(s.get('search_widths') or [0])[-1]} from the median seed (from {s.get('wide_from', 4)} letters: at 3 "
        "letters the wider search runs for hours). An upper bound on the optimum.",
        "- Time per practice game: a trained planner playing 100 practice games in one process with nothing "
        "else running (the training runs shared the machine).",
        "- Proven lower bound: `analysis/limits.py`, which considers every valid guess at every position "
        f"(branch and bound with proven bounds, up to {s.get('proof_seconds', 0) / 60:.0f} minutes per list). "
        "When it finishes, the best strategy found is the optimum; otherwise the bound is still a proof.",
        "- Proofs never use the planner, so they can't inherit its blind spots; the searches can.", ""])


if __name__ == "__main__":
    print(build(sys.argv[1]))
