"""Numbers for the stability and Text analyses that are not plain accuracies; written to
results/extra_checks.json, which audit_numbers.py reads.

1. Pixel change on physical outcome (Qwen3-VL-8B). Re-renders the original, the oracle
   aftermath of falling stacks, and the model's own reconstructions, and counts the
   share of pixels whose largest channel difference from the original exceeds 30/255.
   A model that only checked whether the picture changed would answer "falls over"
   for reconstructions that show a standing stack.
2. Rearrangement Text traces (Qwen3-VL-8B): how many place the target at the swap
   partner's original position (keyword match on the reply).

    python scripts/extra_checks.py
"""
import json
import os
import re
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from analyze_imagined import implied_from_recon  # noqa: E402
from imagine.render import Camera, render  # noqa: E402

MODEL = "Qwen3-VL-8B-Instruct-q4"
Q = {r["id"]: r for r in map(json.loads, open("data/v1_figs/questions.jsonl"))}
REC = {r["id"]: r for r in map(json.loads, open(f"data/v1/recon/{MODEL}/index.jsonl"))}
load = lambda c: {r["id"]: r for r in map(json.loads, open(f"results/v1/{MODEL}/{c}.jsonl"))}  # noqa: E731


def img(scene, cam):
    return np.asarray(render(scene, Camera.from_dict(cam), ss=1)[0], float)


def changed(a, b):
    return float((np.abs(a - b).max(-1) > 30).mean() * 100)


def job(i):
    q = Q[i]
    cam = q["original"]["camera"]
    o = img(q["original"]["scene"], cam)
    out = {"id": i, "implied": implied_from_recon(q, REC[i]), "recon": changed(o, img(REC[i]["scene"], cam))}
    if q["answer"] == "falls over":
        out["oracle"] = changed(o, img(q["imagined"]["scene"], cam))
    return out


def pixel_change():
    recon, oracle = load("recon"), load("oracle")
    with Pool(8) as p:
        rows = p.map(job, [i for i, q in Q.items() if q["task"] == "stability"])
    st = [r for r in rows if r["implied"] == "stays standing"]
    fa = [r for r in rows if "oracle" in r]
    a, f = np.array([r["recon"] for r in st]), np.array([r["oracle"] for r in fa])
    lo, hi = max(a.min(), f.min()), min(a.max(), f.max())
    st_in = [r for r in st if lo <= r["recon"] <= hi]
    fa_in = [r for r in fa if lo <= r["oracle"] <= hi]
    stands = lambda rs: sum(recon[r["id"]]["pred"] == "stays standing" for r in rs)  # noqa: E731
    top = [r for r in st if r["recon"] >= np.percentile(a, 75)]
    return {
        "recon_standing_n": len(st), "recon_standing_says_stands": stands(st),
        "recon_standing_says_stands_pct": 100 * stands(st) / len(st),
        "recon_top_quartile_says_stands_pct": 100 * stands(top) / len(top),
        "recon_change_mean": a.mean(), "recon_change_min": a.min(), "recon_change_max": a.max(),
        "fallen_change_mean": f.mean(), "fallen_change_min": f.min(), "fallen_change_max": f.max(),
        "shared_range": [lo, hi],
        "shared_recon_n": len(st_in), "shared_recon_says_stands": stands(st_in),
        "shared_recon_says_stands_pct": 100 * stands(st_in) / len(st_in),
        "shared_fallen_n": len(fa_in),
        "shared_fallen_says_falls": sum(oracle[r["id"]]["pred"] == "falls over" for r in fa_in),
        "shared_fallen_says_falls_pct": 100 * sum(oracle[r["id"]]["pred"] == "falls over" for r in fa_in) / len(fa_in),
    }


def cot_rewrite():
    rs = [r for r in load("cot").values() if r["task"] == "swap"]
    hits = 0
    for r in rs:
        p, x = Q[r["id"]]["meta"]["partner"].lower(), r["reply"].lower()
        pats = [f"where the {p} was", f"occupied by the {p}", f"original {p}'s position",
                f"{p}'s original position", f"position of the {p}", f"where the {p} is",
                f"{p}'s (?:old|previous|current|former) position",
                f"(?:old|previous|current|former|original) position of the {p}"]
        hits += any(re.search(s, x) for s in pats)
    return {"cot_swap_n": len(rs), "cot_swap_partner_position": hits}


if __name__ == "__main__":
    out = {**pixel_change(), **cot_rewrite()}
    out = {k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}
    json.dump(out, open("results/extra_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
