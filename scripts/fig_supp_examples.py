"""Supplementary figure: two questions per task, each with the four second images and
Qwen3-VL-8B's actual answer under each (read from the results files).

    python scripts/fig_supp_examples.py paper/figs/supp_examples.pdf
"""
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3
from PIL import Image  # noqa: E402

MODEL = "results/v1/Qwen3-VL-8B-Instruct-q4"
IDS = ["viewpoint_00003", "viewpoint_00015", "swap_00000", "swap_00002", "stability_00003", "stability_00005"]
COLS = [("seen", "original", "direct"), ("own imagined state", None, "recon"),
        ("oracle", "imagined", "oracle"), ("misleading", "wrong", "wrong")]
CROP = (0, 105, 448, 400)


def main(out):
    q = {r["id"]: r for r in map(json.loads, open("data/v1_figs/questions.jsonl"))}
    res = {c: {r["id"]: r for r in map(json.loads, open(f"{MODEL}/{c}.jsonl"))} for _, _, c in COLS}
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, axes = plt.subplots(len(IDS), 4, figsize=(6.8, 1.55 * len(IDS)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.62})
    for r, iid in enumerate(IDS):
        it = q[iid]
        for c, (title, state, cond) in enumerate(COLS):
            ax = axes[r, c]
            path = f"data/v1_figs/recon/{iid}.png" if state is None else f"data/v1_figs/images/{iid}/{state}.png"
            ax.imshow(Image.open(path).crop(CROP))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            p = res[cond][iid]
            ax.text(0.5, -0.05, f"“{p['pred']}” ({'correct' if p['correct'] else 'wrong'})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.3, style="italic")
        q_text = "\n".join(textwrap.wrap(f"{it['question']}  (answer: {it['answer']})", 150))
        box = axes[r, 0].get_position()
        fig.text(box.x0, box.y1 + 0.006, q_text, fontsize=6.3, va="bottom", ha="left")
    for c, (title, _, _) in enumerate(COLS):                 # column headers above the first question
        box = axes[0, c].get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + 0.032, title, ha="center", va="bottom",
                 fontsize=7.5, weight="bold")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.03)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
