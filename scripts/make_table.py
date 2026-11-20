"""Main results table (LaTeX), generated from results/ so no number is typed by hand.
Rows: model x task. Columns: second input. Best non-misleading cell per row in bold.

    python scripts/make_table.py results/v1 paper/tables/main.tex \\
        --models Qwen3-VL-8B-Instruct-q4 Qwen2.5-VL-7B-Instruct-q4 InternVL3-8B-hf-q4 \\
                 Qwen3-VL-32B-Instruct-q4
"""
import argparse
import json
import os

COLS = [("direct", "Img."), ("cot", "Text"), ("recon", "Own"), ("oracle", "Oracle"),
        ("shuffled", "Unrel."), ("wrong", "Misl.")]
TASKS = [("viewpoint", "View"), ("swap", "Swap"), ("stability", "Stab.")]
SHORT = {"Qwen3-VL-8B-Instruct-q4": "Qwen3-VL-8B", "Qwen2.5-VL-7B-Instruct-q4": "Qwen2.5-VL-7B",
         "InternVL3-8B-hf-q4": "InternVL3-8B", "Qwen3-VL-32B-Instruct-q4": "Qwen3-VL-32B"}


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("out")
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()

    lines = [r"\begin{tabular}{l" + "c" * len(COLS) + "c}", r"\toprule",
             " & " + " & ".join(n for _, n in COLS) + r" & Follows \\"]
    notes = []
    for mi, m in enumerate(a.models):
        lines += [r"\midrule", rf"\multicolumn{{{len(COLS) + 2}}}{{l}}{{\emph{{{SHORT.get(m, m)}}}}} \\"]
        R = {c: load(os.path.join(a.results, m, f"{c}.jsonl")) for c, _ in COLS}
        for ti, (t, tname) in enumerate(TASKS):
            vals = {}
            for c, _ in COLS:
                sub = [r["correct"] for r in R[c] if r["task"] == t]
                vals[c] = (100 * sum(sub) / len(sub), len(sub)) if sub else None
            best = max(v[0] for c, v in vals.items() if v and c != "wrong")
            cells = []
            for c, _ in COLS:
                v = vals[c]
                if v is None:
                    cells.append("--")
                    continue
                s = f"{v[0]:.1f}"
                if c != "wrong" and abs(v[0] - best) < 1e-9:
                    s = rf"\textbf{{{s}}}"
                if v[1] < 200:
                    s += r"$^\dagger$"
                    notes.append((SHORT.get(m, m), t, c, v[1]))
                cells.append(s)
            fol = [r["pred"] == r["wrong_answer"] for r in R["wrong"] if r["task"] == t]
            cells.append(f"{100 * sum(fol) / len(fol):.1f}" if fol else "--")
            lines.append(f"\\quad {tname} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for n in notes:
        print("partial:", n)


if __name__ == "__main__":
    main()
