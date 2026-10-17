"""Check every number in the paper's prose against numbers computed from the results.

Collects all accuracies we can compute (per model / condition / task, the viewpoint
same/flip splits, stability stable/falls splits, follow rates, noise sweep,
reconstruction analyses, subsets used in the text), then lists each number in the
.tex prose with where it matched. Numbers that match nothing are printed for a
manual check (most are design constants like 200 questions or 1.4 ratios).

    python scripts/audit_numbers.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze_imagined import implied_from_recon  # noqa: E402

TASKS = ("viewpoint", "swap", "stability")
known = {}


def add(v, where):
    for s in {f"{v:.1f}", f"{round(v)}", f"{v:.0f}"}:
        known.setdefault(s, set()).add(where)
    if abs(v - round(v)) < 1e-9:
        known.setdefault(f"{v:.1f}", set()).add(where)


def acc(rows):
    return 100 * sum(r["correct"] for r in rows) / len(rows) if rows else None


def main():
    Q = {r["id"]: r for r in map(json.loads, open("data/v1_figs/questions.jsonl"))}
    for d in glob.glob("results*/**/v1/*", recursive=True) + glob.glob("results/v1/*") + glob.glob("results_camturn/v1_camturn/*"):
        if not os.path.isdir(d):
            continue
        m = d.replace("results", "R")
        for f in glob.glob(f"{d}/*.jsonl"):
            c = os.path.basename(f)[:-6]
            rows = [json.loads(line) for line in open(f)]
            byid = {r["id"]: r for r in rows}
            for t in TASKS:
                rs = [r for r in rows if r["task"] == t]
                if not rs:
                    continue
                add(acc(rs), f"{m}/{c}/{t}")
                for k in (50, 100):
                    sub = [r for r in rs if int(r["id"].split("_")[1]) < k]
                    if sub and len(sub) < len(rs):
                        add(acc(sub), f"{m}/{c}/{t}/first{k}")
                if t == "viewpoint":
                    for flag in (True, False):
                        sub = [r for r in rs if r["meta"].get("answer_flips_from_current") == flag]
                        add(acc(sub), f"{m}/{c}/{t}/flip={flag}")
                if t == "stability":
                    for ans in ("stays standing", "falls over"):
                        add(acc([r for r in rs if r["answer"] == ans]), f"{m}/{c}/{t}/{ans}")
                if c == "wrong":
                    add(100 * sum(r["pred"] == r["wrong_answer"] for r in rs) / len(rs), f"{m}/follow/{t}")
            # accuracy of other conditions on the ids a partial condition (cot) covers
            if c == "cot":
                for c2 in ("direct", "oracle", "recon"):
                    p2 = f"{d}/{c2}.jsonl"
                    if os.path.exists(p2):
                        other = {r["id"]: r for r in map(json.loads, open(p2))}
                        for t in TASKS:
                            ids = [i for i in byid if i.startswith(t) and i in other]
                            if ids:
                                add(acc([other[i] for i in ids]), f"{m}/{c2}/{t}/on-cot-ids")
            for r in rows:
                pass
        cot = f"{d}/cot.jsonl"
        if os.path.exists(cot):
            toks = [json.loads(line)["gen_tokens"] for line in open(cot)]
            add(sum(toks) / len(toks), f"{m}/cot/mean_tokens")
    # reconstruction analyses
    for idx in glob.glob("data/v1/recon/*/index.jsonl"):
        tag = idx.split("/")[-2]
        rec = {r["id"]: r for r in map(json.loads, open(idx))}
        for t in TASKS:
            if tag.startswith("noise_"):
                rs = [r for r in rec.values() if r["task"] == t]
                add(100 * sum(r["implied_correct"] for r in rs) / len(rs), f"{tag}/implied/{t}")
    for tag, res in (("Qwen3-VL-8B-Instruct-q4", "results/v1/Qwen3-VL-8B-Instruct-q4/recon.jsonl"),
                     ("Qwen3-VL-8B-Instruct-q4-rescaled", "results_rescaled/v1/Qwen3-VL-8B-Instruct-q4/recon.jsonl"),
                     ("Qwen2.5-VL-7B-Instruct-q4", "results/v1/Qwen2.5-VL-7B-Instruct-q4/recon.jsonl"),
                     ("InternVL3-8B-hf-q4", "results/v1/InternVL3-8B-hf-q4/recon.jsonl")):
        idx = f"data/v1/recon/{tag}/index.jsonl"
        if not (os.path.exists(idx) and os.path.exists(res)):
            continue
        rec = {r["id"]: r for r in map(json.loads, open(idx))}
        for t in TASKS:
            right, follow, ar, aw = [], [], [], []
            for r in map(json.loads, open(res)):
                if r["task"] != t or r["id"] not in rec:
                    continue
                imp = implied_from_recon(Q[r["id"]], rec[r["id"]])
                if imp is None:
                    continue
                ok = imp == Q[r["id"]]["answer"]
                right.append(ok)
                follow.append(r["pred"] == imp)
                (ar if ok else aw).append(r["correct"])
            for name, xs in (("implied_right", right), ("follows", follow), ("acc|right", ar), ("acc|wrong", aw)):
                if xs:
                    add(100 * sum(xs) / len(xs), f"recon:{tag}/{name}/{t}")
    if os.path.exists("results/extra_checks.json"):            # scripts/extra_checks.py
        for k, v in json.load(open("results/extra_checks.json")).items():
            for x in (v if isinstance(v, list) else [v]):
                add(float(x), f"extra/{k}")
    tex = ""
    for f in ["0_abstract", "1_intro", "2_related", "3_benchmark", "5_results", "6_discussion", "7_supp"]:
        tex += open(f"paper/sec/{f}.tex").read() + "\n"
    tex += open("paper/main.tex").read()
    nums = re.findall(r"(?<![\w.\\])(\d+(?:\.\d+)?)(?![\w])", re.sub(r"\\(cite|ref|cref|Cref|label)\{[^}]*\}", "", tex))
    unmatched = []
    for n in dict.fromkeys(nums):
        if n in known:
            continue
        unmatched.append(n)
    print("numbers in text:", len(set(nums)), " matched:", len(set(nums)) - len(unmatched))
    print("unmatched (check by hand):", " ".join(unmatched))


if __name__ == "__main__":
    main()
