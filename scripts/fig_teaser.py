"""Figure 1: one example per task, original -> imagined state, with the question.

    python scripts/fig_teaser.py data/v1_figs/sel paper/figs/teaser.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PICK = [("viewpoint_00018", "Viewpoint change",
         "Walk a quarter turn to your right.\nIs the orange cylinder left or right\nof the red sphere?"),
        ("swap_00003", "Rearrangement",
         "The blue cube and the gray cube trade\nplaces. What is then closest to\nthe gray cube?"),
        ("stability_00001", "Physical outcome",
         "This stack is released.\nWill it stay standing\nor fall over?")]
CROP = (0, 105, 448, 400)          # drop most of the empty sky and foreground


def main(root, out):
    items = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(root, "questions.jsonl"))}
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.0), gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for g, (iid, title, q) in enumerate(PICK):
        it = items[iid]
        for c, st in enumerate(("original", "imagined")):
            ax = axes[2 * g + c]
            ax.imshow(Image.open(os.path.join(root, it[st]["image"])).crop(CROP))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            ax.set_title("seen" if c == 0 else "imagined (oracle)", fontsize=6.5, pad=2)
        box = axes[2 * g].get_position()
        mid = (box.x0 + axes[2 * g + 1].get_position().x1) / 2
        fig.text(mid, box.y1 + 0.22, title, ha="center", va="bottom", fontsize=7.5, weight="bold")
        fig.text(mid, box.y0 - 0.04, q, ha="center", va="top", fontsize=6.3, linespacing=1.15)
        fig.text(mid, box.y0 - 0.42, f"answer: {it['answer']}", ha="center", va="top",
                 fontsize=6.3, style="italic")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
