"""Main results table (LaTeX), generated from results/ so no number is typed by hand.

    python scripts/make_table.py results/v1 paper/tables/main.tex \\
        --models Qwen3-VL-8B-Instruct-q4 Qwen2.5-VL-7B-Instruct-q4
"""
import argparse
import json
import os

ROWS = [("direct", "Original image only"),
        ("cot", "+ describe the change in text"),
        ("recon", "+ own imagined state"),
        ("oracle", "+ oracle imagined state"),
        ("shuffled", "+ another question's state"),
        ("wrong", "+ misleading imagined state")]
TASKS = [("viewpoint", "View"), ("swap", "Swap"), ("stability", "Stab.")]
SHORT = {"Qwen3-VL-8B-Instruct-q4": "Qwen3-VL-8B", "Qwen2.5-VL-7B-Instruct-q4": "Qwen2.5-VL-7B"}


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("out")
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()

    cells, partial = {}, set()
    for m in a.models:
        for cond, _ in ROWS:
            rs = load(os.path.join(a.results, m, f"{cond}.jsonl"))
            for t, _ in TASKS:
                sub = [r["correct"] for r in rs if r["task"] == t]
                cells[m, cond, t] = (100 * sum(sub) / len(sub)) if sub else None
                if sub and len(sub) < 200:
                    partial.add((m, cond))
        # follow rate on the misleading image
        rs = load(os.path.join(a.results, m, "wrong.jsonl"))
        for t, _ in TASKS:
            sub = [r["pred"] == r["wrong_answer"] for r in rs if r["task"] == t]
            cells[m, "follow", t] = (100 * sum(sub) / len(sub)) if sub else None

    def best(m, t):
        vals = [cells[m, c, t] for c, _ in ROWS if c not in ("wrong",) and cells[m, c, t] is not None]
        return max(vals) if vals else None

    ncol = len(TASKS) * len(a.models)
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(TASKS)}}}{{c}}{{{SHORT.get(m, m)}}}" for m in a.models) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + k * len(TASKS)}-{1 + (k + 1) * len(TASKS)}}}" for k in range(len(a.models))),
             "Second input & " + " & ".join(n for _ in a.models for _, n in TASKS) + r" \\", r"\midrule"]
    for cond, label in ROWS:
        vals = []
        for m in a.models:
            for t, _ in TASKS:
                v = cells[m, cond, t]
                if v is None:
                    vals.append("--")
                    continue
                s = f"{v:.1f}"
                if cond != "wrong" and best(m, t) is not None and abs(v - best(m, t)) < 1e-9:
                    s = rf"\textbf{{{s}}}"
                vals.append(s)
        mark = r"$^\dagger$" if any((m, cond) in partial for m in a.models) else ""
        if cond == "wrong":
            lines.append(r"\midrule")
        lines.append(f"{label}{mark} & " + " & ".join(vals) + r" \\")
    lines.append(r"\quad \emph{answers as the misleading state implies} & "
                 + " & ".join(f"{cells[m, 'follow', t]:.1f}" if cells[m, 'follow', t] is not None else "--"
                              for m in a.models for t, _ in TASKS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if partial:
        print("partial (fewer than 200 per task):", sorted(partial))


if __name__ == "__main__":
    main()
