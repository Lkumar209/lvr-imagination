"""Figure: accuracy with a degraded oracle as position noise grows, one panel per task.

Solid lines: model accuracy with the noisy imagined image. Dotted lines, same colour:
that model with the original image only. Gray dashed: how often the noisy imagined
state still implies the right answer (the information left in the image).

    python scripts/fig_noise.py data/v1 paper/figs/noise.pdf \\
        --models Qwen3-VL-8B-Instruct-q4 Qwen2.5-VL-7B-Instruct-q4
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3 (PDF checkers reject Type 3)
import numpy as np  # noqa: E402

TASKS = [("viewpoint", "Viewpoint"), ("swap", "Rearrangement"), ("stability", "Physical outcome")]
# dataviz reference palette, categorical slots 1-2 (validated: CVD and normal-vision pass)
COLORS = ["#2a78d6", "#eb6834"]
MARKERS = ["o", "s"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SHORT = {"Qwen3-VL-8B-Instruct-q4": "Qwen3-VL-8B", "Qwen2.5-VL-7B-Instruct-q4": "Qwen2.5-VL-7B"}


def acc(path, task):
    rs = [json.loads(line) for line in open(path)]
    rs = [r for r in rs if r["task"] == task]
    return 100 * np.mean([r["correct"] for r in rs]) if rs else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0])
    a = p.parse_args()
    ds = os.path.basename(os.path.normpath(a.root))

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), sharey=True, gridspec_kw={"wspace": 0.08})
    xs = np.arange(len(a.sigmas))                  # even spacing; sigmas are not linear
    for ax, (task, title) in zip(axes, TASKS):
        info = []
        for s in a.sigmas:
            rows = [json.loads(line) for line in open(os.path.join(a.root, "recon", f"noise_{s:g}", "index.jsonl"))]
            info.append(100 * np.mean([r["implied_correct"] for r in rows if r["task"] == task]))
        ax.plot(xs, info, color=MUTED, lw=1.2, ls="--", zorder=1)
        for m, (model, c, mk) in enumerate(zip(a.models, COLORS, MARKERS)):
            ys = [acc(os.path.join(f"results_noise/s{s:g}", ds, model, "recon.jsonl"), task) for s in a.sigmas]
            ax.plot(xs, ys, color=c, lw=1.5, marker=mk, ms=3.5, mec="white", mew=0.6, zorder=3,
                    label=SHORT.get(model, model))
            base = acc(os.path.join("results", ds, model, "direct.jsonl"), task)
            ax.axhline(base, color=c, lw=1, ls=":", zorder=2)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.set_xticks(xs, [f"{s:g}" for s in a.sigmas])
        ax.set_ylim(0, 102)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("position noise $\\sigma$ (m)")
    axes[0].set_ylabel("accuracy (%)")
    for ax in axes:
        ax.set_xlim(-0.2, len(xs) - 0.8)
    from matplotlib.lines import Line2D
    handles = axes[0].get_legend_handles_labels()[0] + [
        Line2D([], [], color=MUTED, lw=1, ls=":", label="original image only"),
        Line2D([], [], color=MUTED, lw=1.2, ls="--", label="noisy state implies answer")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=5.6, frameon=False,
                   handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(a.out)


if __name__ == "__main__":
    main()
