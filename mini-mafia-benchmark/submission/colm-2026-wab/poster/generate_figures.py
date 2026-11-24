import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/colm-poster-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

TEXT = "#273238"
TEAL = "#0C7283"
GREEN = "#47A57A"
RED = "#D95966"
BLUE = "#5B9DB8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.edgecolor": TEXT,
        "text.color": TEXT,
        "pdf.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)


def name_bias_plot() -> None:
    names = ["Bob", "Alice", "Charlie", "Diana"]
    rates = np.array([55.9669, 55.5566, 54.1631, 53.7612])
    errors = np.array([0.4843, 0.4847, 0.4844, 0.4887])
    colors = ["#087184", "#2591A1", "#62B6C1", "#9CD3DA"]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.barh(
        y,
        rates - 50,
        left=50,
        color=colors,
        edgecolor=TEAL,
        linewidth=1.4,
        height=0.44,
        xerr=errors,
        error_kw={"ecolor": TEXT, "elinewidth": 1.3, "capsize": 3.5, "capthick": 1.3},
    )
    ax.set_yticks(y, names, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(50, 57.2)
    ax.set_xlabel("Win rate (%)", fontweight="bold")
    ax.set_xticks(np.arange(50, 58, 1))
    ax.grid(axis="x", color="#BFD6DA", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for yi, value, err in zip(y, rates, errors):
        ax.text(value + err + 0.13, yi, f"{value:.2f} ± {err:.2f}", va="center", fontsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "name-bias-bars.pdf")


def recency_plot() -> None:
    roles = ["Villager", "Mafioso", "Detective"]
    overall = np.array([64.5908, 35.4092, 64.5908])
    overall_err = np.array([0.4041, 0.4041, 0.4041])
    last = np.array([63.6031, 41.4474, 71.6867])
    last_err = np.array([0.7058, 0.7176, 0.6607])
    colors = [GREEN, RED, BLUE]
    x = np.arange(len(roles))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    for idx, color in enumerate(colors):
        ax.bar(
            x[idx] - width / 2,
            overall[idx],
            width,
            color=color,
            edgecolor=color,
            linewidth=1.8,
            yerr=overall_err[idx],
            error_kw={"ecolor": TEXT, "elinewidth": 1.2, "capsize": 3, "capthick": 1.2},
        )
        ax.bar(
            x[idx] + width / 2,
            last[idx],
            width * 0.82,
            facecolor="none",
            edgecolor=color,
            linewidth=2.2,
            hatch="////",
            yerr=last_err[idx],
            error_kw={"ecolor": TEXT, "elinewidth": 1.2, "capsize": 3, "capthick": 1.2},
        )
        ax.text(x[idx] - width / 2, overall[idx] + 1.5, f"{overall[idx]:.1f}", ha="center", fontsize=10.5, fontweight="bold")
        ax.text(x[idx] + width / 2, last[idx] + 1.8, f"{last[idx]:.1f}", ha="center", fontsize=10.5, fontweight="bold")

    ax.set_ylim(30, 76)
    ax.set_ylabel("Win rate (%)", fontweight="bold")
    ax.set_xticks(x, roles, fontweight="bold")
    ax.set_yticks(np.arange(30, 81, 10))
    ax.grid(axis="y", color="#BFD6DA", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Patch(facecolor="#718087", edgecolor="#718087", linewidth=1.8, label="Overall"),
        Patch(facecolor="none", edgecolor="#718087", linewidth=2.0, hatch="////", label="Last to speak"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2, fontsize=11.5)
    save(fig, "recency-bars.pdf")


if __name__ == "__main__":
    name_bias_plot()
    recency_plot()
