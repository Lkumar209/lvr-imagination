"""Accuracy per task x condition with 95% bootstrap intervals, plus how often the
model follows a misleading image.

    python scripts/score.py results/v1/Qwen3-VL-8B-Instruct
    python scripts/score.py results/v1/*            # several models
"""
import argparse
import glob
import json
import os
import random

ORDER = ["direct", "cot", "oracle", "oracle_only", "wrong", "shuffled", "generated"]


def ci(xs, n_boot=2000, seed=0):
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def fmt(xs):
    if not xs:
        return "-"
    lo, hi = ci(xs)
    return f"{100 * sum(xs) / len(xs):5.1f} [{100 * lo:4.1f},{100 * hi:4.1f}]"


def load(folder):
    rows = {}
    for path in glob.glob(os.path.join(folder, "*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            rows.setdefault(r["condition"], []).append(r)
    return rows


def report(folder):
    rows = load(folder)
    conds = [c for c in ORDER if c in rows] + sorted(set(rows) - set(ORDER))
    tasks = sorted({r["task"] for rs in rows.values() for r in rs})
    print(f"\n== {folder}")
    print(f"{'condition':12s} " + " ".join(f"{t:>24s}" for t in tasks + ['all']) + "   n  parse-fail")
    for c in conds:
        rs = rows[c]
        cells = [fmt([r["correct"] for r in rs if r["task"] == t]) for t in tasks]
        fails = sum(r["pred"] is None for r in rs)
        print(f"{c:12s} " + " ".join(f"{x:>24s}" for x in cells + [fmt([r['correct'] for r in rs])])
              + f" {len(rs):4d} {fails:5d}")

    # does the model follow the image? (rate of choosing the answer the second image implies)
    for c, key in (("wrong", "wrong_answer"), ("shuffled", "partner_answer")):
        if c in rows:
            rs = [r for r in rows[c] if r.get(key) is not None]
            follow = [r["pred"] == r[key] for r in rs]
            per = {t: fmt([r["pred"] == r[key] for r in rs if r["task"] == t]) for t in tasks}
            print(f"follows {c:9s} image: {fmt(follow)}  " + "  ".join(f"{t}={v}" for t, v in per.items()))

    # splits that matter for interpretation
    for c in conds:
        vp = [r for r in rows[c] if r["task"] == "viewpoint"]
        if vp:
            flip = [r["correct"] for r in vp if r["meta"].get("answer_flips_from_current")]
            same = [r["correct"] for r in vp if not r["meta"].get("answer_flips_from_current")]
            st = [r for r in rows[c] if r["task"] == "stability"]
            stable = [r["correct"] for r in st if r["answer"] == "stays standing"]
            falls = [r["correct"] for r in st if r["answer"] == "falls over"]
            print(f"  {c:12s} viewpoint flip={fmt(flip)} same={fmt(same)} | "
                  f"stability stable={fmt(stable)} falls={fmt(falls)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    for f in p.parse_args().folders:
        report(f)


if __name__ == "__main__":
    main()
