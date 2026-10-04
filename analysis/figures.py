"""Static charts for the benchmark report (matplotlib, light theme, one idea per chart)."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated palette order)
OPTIMUM = 3.4212   # 7,920 / 2,315: proven optimum for the original answer list (Selby)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width=8, height=4.5):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _title(ax, text, subtitle=None):
    subtitle = textwrap.fill(subtitle, 70) if subtitle else ""
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(text, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=10 + 13 * lines)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def learning_curves(curves, path, check_words):
    """curves: {label: (games array, mean array, ci array)}.

    No optimum line here on purpose: these are averages over a subset of check
    words, which aren't comparable with the 3.4212 optimum over all answers."""
    fig, ax = _figure()
    ends = []
    for color, (label, (games, mean, ci)) in zip(SERIES, curves.items()):
        ax.fill_between(games, mean - ci, mean + ci, color=color, alpha=0.12, linewidth=0)
        ax.plot(games, mean, color=color, linewidth=2)
        ends.append([float(mean[-1]), float(games[-1]), f"{label}: {mean[-1]:.2f}"])
    top = max(float(np.max(mean)) for _, mean, _ in curves.values())
    y_low, y_high = 3.3, min(top + 0.2, 6.2)
    ends.sort()
    gap = 0.05 * (y_high - y_low)  # end labels at least ~5% of the axis apart, so they never collide
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    right = max(float(g[-1]) for g, _, _ in curves.values())
    for y, _, text in ends:
        ax.text(right * 1.15, y, text, color=INK_2, fontsize=9, va="center")
    ax.set_xscale("symlog", linthresh=10)
    ticks = [t for t in (0, 1, 3, 10, 30, 100, 300, 1000, 3000) if t <= right]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("games trained (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("average guesses on the check words", color=INK_2, fontsize=9)
    ax.set_ylim(y_low, y_high)
    ax.set_xlim(0, right * 4)
    _title(ax, "Learning curves", f"Skill check on the same {check_words} words (best guesses); "
                                  "band = 95% CI across seeds")
    _save(fig, path)


def dot_plot(rows, path, title, subtitle, highlight=None, reference=OPTIMUM):
    """rows: list of (label, value, ci or None), drawn top to bottom in the given order."""
    fig, ax = _figure(8, 0.45 * len(rows) + 1.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, ci) in zip(y, rows):
        color = SERIES[1] if highlight and label == highlight else SERIES[0]
        if ci:
            ax.plot([value - ci, value + ci], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(value, yi, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        right_end = value + (ci or 0)
        ax.text(right_end, yi, f"   {value:.3f}" + (f" ± {ci:.3f}" if ci else ""), color=INK_2,
                fontsize=8.5, va="center")
    ax.set_yticks(y, [label for label, _, _ in rows])
    if reference:
        ax.axvline(reference, color=INK_2, linewidth=1, linestyle=(0, (4, 4)))
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.18)  # room for the value labels
    ax.set_xlabel("average guesses (all 2,315 answers)", color=INK_2, fontsize=9)
    _title(ax, title, subtitle)
    _save(fig, path)


def length_study(table, strategies, path):
    """Left: average guesses by word length, one line per strategy. Right: bits per guess."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    lengths = [e["length"] for e in table]
    ends = []
    for color, (kind, label) in zip(SERIES, strategies):
        values = [e[kind][0] for e in table]
        left.plot(lengths, values, color=color, linewidth=2, marker="o", markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends.append([values[-1], label, color])
    ends.sort()
    low, high = left.get_ylim()
    for i in range(1, len(ends)):  # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.055 * (high - low))
    for y, label, color in ends:  # a colored key mark, so each label is tied to its line
        left.plot([lengths[-1] + 0.2, lengths[-1] + 0.45], [y, y], color=color, linewidth=2.5,
                  solid_capstyle="round", clip_on=False)
        left.text(lengths[-1] + 0.55, y, label, color=INK_2, fontsize=8.5, va="center")
    left.set_xlim(lengths[0] - 0.2, lengths[-1] + 2.9)
    left.set_xticks(lengths)
    left.set_xlabel("letters per word", color=INK_2, fontsize=9)
    left.set_ylabel("average guesses (every answer once)", color=INK_2, fontsize=9)
    _style(left)
    _title(left, "Guesses needed, by word length", "Common-word lists; list sizes differ by length")

    bits = [e["bits_per_guess"] for e in table]
    right.bar(lengths, bits, width=0.6, color=SERIES[0])
    for x, b in zip(lengths, bits):
        right.text(x, b, f"{b:.2f}", ha="center", va="bottom", color=INK_2, fontsize=8.5)
    right.set_xticks(lengths)
    right.set_xlabel("letters per word", color=INK_2, fontsize=9)
    right.set_ylabel("bits per guess", color=INK_2, fontsize=9)
    _style(right)
    _title(right, "Information per guess", "log2(answers) / average guesses (with search)")
    _save(fig, path)


def bars(rows, path, title, subtitle, xlabel):
    """rows: list of (label, value in 0..1, note)."""
    fig, ax = _figure(8, 0.5 * len(rows) + 1.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [v for _, v, _ in rows], height=0.55, color=SERIES[0])
    for yi, (_, value, note) in zip(y, rows):
        ax.text(value, yi, f"  {note}", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    _title(ax, title, subtitle)
    _save(fig, path)


# --- limits study (analysis/limits_study.py) ------------------------------------------------------------
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]  # ordinal
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]   # blue (better) - gray - red (worse)


def _end_labels(ax, ends, x, span):
    """ends: [y, text, color]; keeps labels at least ~5% of the axis apart, each with a colored key mark."""
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.055 * span)
    for y, text, color in ends:
        ax.plot([x + 0.15, x + 0.4], [y, y], color=color, linewidth=2.5, solid_capstyle="round", clip_on=False)
        ax.text(x + 0.5, y, text, color=INK_2, fontsize=8.5, va="center")


def limits_floor(rows, fixed, path):
    """rows: [(letters, floor entry)]; fixed: [(letters, planner, best, lower, proven, subsets)]."""
    fig, ax = _figure(8.5, 4.8)
    letters = [L for L, _ in rows]
    best = [e["upper"] / e["answers"] for _, e in rows]
    lower = [e["lower"] / e["answers"] for _, e in rows]
    planner = [e["planner"][0] for _, e in rows]
    ends = []
    if fixed:
        fl = [r[0] for r in fixed]
        ax.plot(fl, [r[2] for r in fixed], color=SERIES[2], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends.append([fixed[-1][2], f"best found, {fixed[0][5] and 669} answers at every length", SERIES[2]])
    ax.plot(letters, planner, color=SERIES[1], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ends.append([planner[-1], "learned planner alone", SERIES[1]])
    for L, b, lo, (_, e) in zip(letters, best, lower, rows):
        if not e["proven"]:   # the proven bracket: the optimum lies somewhere on this bar
            ax.plot([L, L], [lo, b], color=SERIES[0], linewidth=6, alpha=0.25, solid_capstyle="round")
    ax.plot(letters, best, color=SERIES[0], linewidth=2)
    for L, b, (_, e) in zip(letters, best, rows):
        ax.plot(L, b, "o", markersize=8, color=SERIES[0] if e["proven"] else SURFACE, markeredgecolor=SERIES[0],
                markeredgewidth=2)
    ends.append([best[-1], "best found (filled: proven optimum)", SERIES[0]])
    low, high = ax.get_ylim()
    _end_labels(ax, ends, letters[-1], high - low)
    ax.set_xlim(letters[0] - 0.3, letters[-1] + 3.6)
    ax.set_xticks(letters)
    ax.set_xlabel("letters per word", color=INK_2, fontsize=9)
    ax.set_ylabel("average guesses (every answer once)", color=INK_2, fontsize=9)
    _title(ax, "The fewest guesses possible, by word length",
           "Natural common-word lists. Shaded bar: where the optimum must lie when it isn't proven "
           "(from the proven lower bound up to the best plan found)")
    _save(fig, path)


def limits_learning(items, path):
    """items: [(title, learning row, (marks, totals, states, seconds, seeds))]: one panel per word list."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows + 0.6), facecolor=SURFACE, sharex=True)
    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)
    for ax, (title, row, (marks, totals, _, _, _)) in zip(axes.ravel(), items):
        _style(ax)
        best = row["floor"]["upper"] if "floor" in row else totals.min()
        pct = 100 * (totals / best - 1)
        later = pct[:, marks >= 1]
        top = max(2.0, float(np.percentile(later, 98)) * 1.15)
        for seed_curve in pct:
            ax.plot(marks, seed_curve, color=SERIES[0], linewidth=0.8, alpha=0.25)
        ax.plot(marks, pct.mean(axis=0), color=SERIES[0], linewidth=2)
        ax.axhline(0, color=INK_2, linewidth=1, linestyle=(0, (4, 4)))
        ax.set_ylim(-0.04 * top, top)
        ax.text(0.98, 0.95, f"game 0: +{pct[:, 0].mean():.0f}% (off the chart)", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=8)
        ax.set_xscale("symlog", linthresh=1)
        ticks = [t for t in (0, 1, 10, 100, 1000) if t <= marks[-1]]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(0, marks[-1] * 1.05)
        digits = 0 if top >= 4 else 1
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, d=digits: f"+{v:.{d}f}%" if v > 1e-9 else "0"))
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold")
    for ax in axes[-1]:
        ax.set_xlabel("practice games (log scale)", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("above best known", color=INK_2, fontsize=9)
    fig.suptitle("How fast the planner learns: total guesses over every answer, vs the best plan known",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.text(0.01, 0.955, "Thin lines: each training seed. Thick: their mean. Dashed: the best total known "
             "(proven optimum at 8-10 letters and for official Wordle).", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def limits_transfer(lengths, grid, path):
    """grid[i, j]: change in total guesses, planner trained on lengths[i] tested on lengths[j], vs native."""
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = _figure(7.2, 6.2)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    span = max(float(np.abs(grid).max()), 1e-4)
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    ax.imshow(100 * grid, cmap=cmap, vmin=-100 * span, vmax=100 * span)
    for i in range(len(lengths)):
        for j in range(len(lengths)):
            value = 100 * grid[i, j]
            text = "native" if i == j else ("0.00%" if abs(value) < 0.005 else f"{value:+.2f}%")
            ax.text(j, i, text, ha="center", va="center", color=INK, fontsize=8)
    ax.set_xticks(range(len(lengths)), [str(L) for L in lengths])
    ax.set_yticks(range(len(lengths)), [str(L) for L in lengths])
    ax.set_xlabel("tested on (letters)", color=INK_2, fontsize=9)
    ax.set_ylabel("trained on (letters)", color=INK_2, fontsize=9)
    _title(ax, "Playing a length it never trained on",
           "Total guesses vs planners trained on the test length itself (mean of 8 seeds each). "
           "Red: more guesses, blue: fewer.")
    _save(fig, path)


def limits_values(items, path):
    """items: [(letters, m, V(m))]: what each length's planner learned."""
    fig, ax = _figure(8.5, 4.8)
    ends = []
    for color, (letters, m, v) in zip(BLUE_RAMP, items):
        ax.plot(m, v, color=color, linewidth=2)
        ends.append([float(v[-1]), f"{letters} letters", color])
    ax.set_xscale("log")
    right = max(float(m[-1]) for _, m, _ in items)
    ends.sort()
    low, high = ax.get_ylim()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05 * (high - low))
    for y, text, color in ends:   # a colored key mark ties each label to its line
        ax.plot([right * 1.2, right * 1.5], [y, y], color=color, linewidth=2.5, solid_capstyle="round",
                clip_on=False)
        ax.text(right * 1.6, y, text, color=INK_2, fontsize=8.5, va="center")
    for (letters, m, v), color in zip(items, BLUE_RAMP):
        ax.plot(m[-1], v[-1], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xlim(1, right * 5)
    ax.set_xlabel("words still possible, m (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("V(m): expected guesses still needed", color=INK_2, fontsize=9)
    _title(ax, "What each planner learned",
           "Median of 8 trained planners per length, drawn up to that list's number of answers")
    _save(fig, path)
