"""Split errors on an imagined second image into "the imagined state was wrong"
and "the model misread a correct imagined state".

For each question: the answer the imagined state implies (from its geometry),
whether that is the true answer, and whether the model's prediction matches it.

    python scripts/analyze_imagined.py data/v1 results/v1/Qwen3-VL-8B-Instruct-q4/recon.jsonl \\
        --recon Qwen3-VL-8B-Instruct-q4
    python scripts/analyze_imagined.py data/v1 results_noise/s0.5/v1/Qwen3-VL-8B-Instruct-q4/recon.jsonl \\
        --recon noise_0.5
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imagine.recon import stack_from_reply  # noqa: E402
from imagine.render import Camera  # noqa: E402
from imagine.tasks import Reject, left_right, nearest, stack_failure  # noqa: E402

LR = re.compile(r"is the (.+?) to the left or to the right of the (.+?)\?")


def implied_from_recon(item, rec):
    """Answer implied by a model reconstruction (recon/<model>/index.jsonl row)."""
    if rec.get("image") is None:
        return None
    task = item["task"]
    if task == "stability":
        try:
            return "stays standing" if stack_failure(stack_from_reply(rec["reply"]), 0.0) is None else "falls over"
        except Reject:
            return None
    by = {o["name"]: o for o in rec["scene"]["objects"]}
    if task == "viewpoint":
        a, b = LR.search(item["question"]).groups()
        if a not in by or b not in by:
            return None
        return left_right(Camera.from_dict(item["imagined"]["camera"]), by[a], by[b])[0]
    t = by.get(item["meta"]["target"])
    if t is None:
        return None
    d = nearest(rec["scene"]["objects"], t)
    return d[0][1] if d else None


def pct(xs):
    return f"{100 * np.mean(xs):5.1f}% (n={len(xs)})" if xs else "   -"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("results", help="recon.jsonl written by evaluate.py")
    p.add_argument("--recon", required=True, help="recon/<tag> folder the results used")
    a = p.parse_args()

    items = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))}
    idx = {r["id"]: r for r in map(json.loads, open(os.path.join(a.root, "recon", a.recon, "index.jsonl")))}
    res = [json.loads(line) for line in open(a.results)]

    print(f"{'task':10s} {'imagined right':>18s} {'follows imagined':>18s} "
          f"{'acc | imagined right':>22s} {'acc | imagined wrong':>22s}")
    for task in ("viewpoint", "swap", "stability"):
        right, follow, acc_r, acc_w = [], [], [], []
        for r in res:
            if r["task"] != task or r["id"] not in idx:
                continue
            it, rec = items[r["id"]], idx[r["id"]]
            imp = rec["implied"] if "implied" in rec else implied_from_recon(it, rec)
            if imp is None:
                continue
            ok = imp == it["answer"]
            right.append(ok)
            follow.append(r["pred"] == imp)
            (acc_r if ok else acc_w).append(r["correct"])
        print(f"{task:10s} {pct(right):>18s} {pct(follow):>18s} {pct(acc_r):>22s} {pct(acc_w):>22s}")


if __name__ == "__main__":
    main()
