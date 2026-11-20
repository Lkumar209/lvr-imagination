"""Paired significance tests for the paper's headline comparisons.

Each test compares two conditions on the same questions with an exact McNemar test
(two-sided binomial test on the discordant pairs), with a Holm correction over the
whole family. Writes a LaTeX-ready summary and prints a table.

    python scripts/significance.py --out paper/tables/significance.txt
"""
import argparse
import json
import os
from math import comb

MODELS = {"Qwen3-VL-8B": "results/v1/Qwen3-VL-8B-Instruct-q4",
          "Qwen2.5-VL-7B": "results/v1/Qwen2.5-VL-7B-Instruct-q4",
          "InternVL3-8B": "results/v1/InternVL3-8B-hf-q4",
          "Qwen3-VL-32B": "results/v1/Qwen3-VL-32B-Instruct-q4"}
TASKS = ["viewpoint", "swap", "stability"]


def load(path):
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, open(path))}


def mcnemar_exact(a, b):
    """Two-sided exact p for paired booleans a, b (lists of equal length)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def compare(name, model, task, ra, rb, ids=None):
    common = sorted(set(ra) & set(rb) if ids is None else set(ra) & set(rb) & set(ids))
    common = [i for i in common if i.startswith(task)]
    if not common:
        return {"claim": name, "model": model, "task": task, "n": 0}
    a = [ra[i]["correct"] for i in common]
    b = [rb[i]["correct"] for i in common]
    p, n01, n10 = mcnemar_exact(a, b)
    return {"claim": name, "model": model, "task": task, "n": len(common),
            "acc_a": 100 * sum(a) / len(a), "acc_b": 100 * sum(b) / len(b), "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/tables/significance.txt")
    a = ap.parse_args()

    rows = []
    for model, d in MODELS.items():
        R = {c: load(f"{d}/{c}.jsonl") for c in ("direct", "cot", "oracle", "oracle_only", "wrong", "shuffled", "recon")}
        for t in TASKS:
            rows.append(compare("oracle > direct", model, t, R["direct"], R["oracle"]))
            rows.append(compare("wrong < direct", model, t, R["direct"], R["wrong"]))
            rows.append(compare("oracle > shuffled", model, t, R["shuffled"], R["oracle"]))
            rows.append(compare("cot > direct", model, t, R["direct"], R["cot"]))
            rows.append(compare("cot vs oracle", model, t, R["cot"], R["oracle"]))
            rows.append(compare("recon vs direct", model, t, R["direct"], R["recon"]))
        flips = [i for i, r in R["direct"].items()
                 if r["task"] == "viewpoint" and r["meta"].get("answer_flips_from_current")]
        rows.append(compare("imagined alone > both (flip)", model, "viewpoint", R["oracle"], R["oracle_only"], flips))
        noise = load(f"results_noise/s1/v1/{os.path.basename(d)}/recon.jsonl")
        rows.append(compare("oracle@1m > direct", model, "viewpoint", R["direct"], noise))
        rows.append(compare("oracle@1m > recon", model, "viewpoint", R["recon"], noise))

    rows = [r for r in rows if r["n"]]
    for r, adj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = adj
    lines = [f"{'claim':30s} {'model':14s} {'task':10s} {'n':>4s} {'A':>6s} {'B':>6s} {'p':>9s} {'p_holm':>9s}"]
    for r in rows:
        lines.append(f"{r['claim']:30s} {r['model']:14s} {r['task']:10s} {r['n']:4d} "
                     f"{r['acc_a']:6.1f} {r['acc_b']:6.1f} {r['p']:9.2g} {r['p_holm']:9.2g}"
                     + ("  *" if r["p_holm"] < 0.05 else ""))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text + "\n\nA = first condition named, B = second; exact McNemar, Holm-corrected over all rows.\n")
    print(text)


if __name__ == "__main__":
    main()
