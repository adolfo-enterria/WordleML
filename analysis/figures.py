"""Static charts for the benchmark report (matplotlib, light theme, one idea per chart)."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
OPTIMUM = 3.4201


def _figure(width=8, height=4.5):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
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
    words, which aren't comparable with the 3.4201 optimum over all answers."""
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
