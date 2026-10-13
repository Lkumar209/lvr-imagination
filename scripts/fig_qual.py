"""Qualitative figure: one question, the four images the model can be given, and
its actual answer with each (read from the results files).

    python scripts/fig_qual.py viewpoint_00014 paper/figs/qual.pdf
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3 (PDF checkers reject Type 3)
from PIL import Image  # noqa: E402

MODEL = "results/v1/Qwen3-VL-8B-Instruct-q4"
CROP = (0, 105, 448, 400)


def main(iid, out):
    q = next(json.loads(line) for line in open("data/v1_figs/questions.jsonl") if f'"{iid}"' in line)
    pred = {c: next(json.loads(line) for line in open(f"{MODEL}/{c}.jsonl") if f'"{iid}"' in line)
            for c in ("direct", "recon", "oracle", "wrong")}
    panels = [("seen only", f"data/v1_figs/images/{iid}/original.png", "direct"),
              ("+ own imagined state", f"data/v1_figs/recon/{iid}.png", "recon"),
              ("+ oracle imagined state", f"data/v1_figs/images/{iid}/imagined.png", "oracle"),
              ("+ misleading state", f"data/v1_figs/images/{iid}/wrong.png", "wrong")]
    plt.rcParams.update({"font.size": 6.5, "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.3), gridspec_kw={"wspace": 0.04, "hspace": 0.48})
    for ax, (title, path, cond) in zip(axes.flat, panels):
        ax.imshow(Image.open(path).crop(CROP))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        p = pred[cond]
        mark = "correct" if p["correct"] else "wrong"
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.text(0.5, -0.04, f"answers “{p['pred']}” ({mark})", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.3, style="italic")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(q["question"], "| answer:", q["answer"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
