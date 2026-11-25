# Author: Can Zhang
# Readout figures, from results/*.json only.
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#0072B2", "#D55E00"  # Okabe-Ito
INK, MUTED = "#222222", "#666666"
META = {"Software": None}  # no tool stamp in the PNG
DESIGNS = [
    ("pooled", "Pooled A/B"),
    ("diverted", "Data-diverted A/B"),
    ("switchback", "Switchback"),
    ("share", "Share replicas (the fix)"),
]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def error_by_design():
    d = load("designs")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharex=True, sharey=True)
    for k, (ax, c) in enumerate(zip(axes, d["cells"])):
        delta = c["delta"]
        ax.axvspan(-300, -100, color="#EEEEEE", lw=0)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(-100, color=MUTED, lw=0.8, ls=":")
        for i, (key, _) in enumerate(DESIGNS):
            rel = 100 * (np.array([t[key] for t in c["tests"]]) - delta) / abs(delta)
            q1, med, q3 = np.percentile(rel, [25, 50, 75])
            col = ORANGE if key == "share" else BLUE
            ax.plot([q1, q3], [i, i], color=col, lw=2.5, solid_capstyle="round")
            ax.plot(med, i, "o", color=col, ms=7, mec="white", mew=1)
        ax.set_title(
            f"{c['W']}-day window\ntreatment explores {c['eps1']:.0%}",
            fontsize=9,
            color=INK,
        )
        ax.set_xlim(-300, 600)
        ax.set_xticks([-200, 0, 200, 400, 600])
        ax.tick_params(axis="y", length=0)
        if k == 0:
            ax.text(-290, -0.75, "wrong\nsign", color=MUTED, fontsize=8, va="center")
            ax.text(15, -0.75, "right\nanswer", color=MUTED, fontsize=8, va="center")
    axes[0].set_yticks(range(len(DESIGNS)), [label for _, label in DESIGNS])
    axes[0].set_ylim(3.5, -1.2)
    fig.supxlabel(
        "Error relative to the true launch effect (%); dotted line: -100% = sees no effect",
        fontsize=9,
        color=INK,
    )
    fig.suptitle(
        "Standard tests miss the gain from exploration; the fix is too noisy to use\n"
        "Median over 100 simulated 14-day tests (dot) and the middle half of the tests (bar)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "error_by_design.png", dpi=150, metadata=META)
    plt.close(fig)


def noise_vs_traffic():
    c = load("fix_variants")["cells"][0]
    sets = [
        100 * np.array([t["S2_linear"] for t in c["tests"]]),
        100 * np.array([t["S2x10_linear"] for t in c["extra"]]),
    ]
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(sets[0]))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(100 * c["delta"], color=INK, lw=1, ls="--")
    ax.text(
        1.3,
        100 * c["delta"] + 0.5,
        "true launch\neffect",
        color=INK,
        fontsize=8,
        va="bottom",
    )
    for i, x in enumerate(sets):
        ax.scatter(i + jitter, x, s=14, color=BLUE, alpha=0.7, lw=0)
        ax.text(i, 31.5, f"SD {x.std():.1f} points", ha="center", color=INK, fontsize=9)
    ax.text(
        1,
        29,
        f"({sets[0].std() / np.sqrt(10):.1f} if the noise were user sampling)",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    ax.set_xticks(
        [0, 1],
        [
            "1x traffic\n10,000 impressions a day",
            "10x traffic\n100,000 impressions a day",
        ],
    )
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(-22, 34)
    ax.set_yticks(range(-20, 31, 10))
    ax.set_ylabel("Estimated launch effect (percentage points of click rate)")
    ax.tick_params(axis="x", length=0)
    ax.set_title(
        "The fix's noise does not shrink with 10x traffic\n"
        "Two replicas, 28-day window, treatment explores 5%; 100 tests each",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fix_noise_vs_traffic.png", dpi=150, metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    error_by_design()
    noise_vs_traffic()
