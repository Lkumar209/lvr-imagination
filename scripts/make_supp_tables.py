"""Supplementary result tables (LaTeX), generated from results/ so no number is typed by hand.

    python scripts/make_supp_tables.py paper/tables/supp_results.tex
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import significance as sig  # noqa: E402

MODELS = [("Qwen3-VL-8B", "Qwen3-VL-8B-Instruct-q4"), ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B-Instruct-q4"),
          ("InternVL3-8B", "InternVL3-8B-hf-q4"), ("Qwen2.5-VL-3B", "Qwen2.5-VL-3B-Instruct")]
TASKS = [("viewpoint", "View"), ("swap", "Swap"), ("stability", "Stab.")]


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def acc(rows, ids=None):
    rs = [r for i, r in rows.items() if ids is None or i in ids]
    return 100 * sum(r["correct"] for r in rs) / len(rs) if rs else None


def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def significance_table():
    rows = []
    for model, d in sig.MODELS.items():
        R = {c: sig.load(f"{d}/{c}.jsonl") for c in ("direct", "cot", "oracle", "oracle_only", "wrong", "shuffled", "recon")}
        for t in sig.TASKS:
            for name, a, b in (("Oracle $>$ Img.", "direct", "oracle"), ("Misl. $<$ Img.", "direct", "wrong"),
                               ("Oracle $>$ Unrel.", "shuffled", "oracle"), ("Text vs.\\ Img.", "direct", "cot"),
                               ("Text vs.\\ Oracle", "cot", "oracle"), ("Own vs.\\ Img.", "direct", "recon")):
                rows.append(sig.compare(name, model, t, R[a], R[b]))
        flips = [i for i, r in R["direct"].items() if r["task"] == "viewpoint" and r["meta"].get("answer_flips_from_current")]
        rows.append(sig.compare("Imagined only $>$ both (flip)", model, "viewpoint", R["oracle"], R["oracle_only"], flips))
        noise = sig.load(f"results_noise/s1/v1/{os.path.basename(d)}/recon.jsonl")
        rows.append(sig.compare("Oracle@1\\,m $>$ Img.", model, "viewpoint", R["direct"], noise))
        rows.append(sig.compare("Oracle@1\\,m $>$ Own", model, "viewpoint", R["recon"], noise))
    rows = [r for r in rows if r["n"]]
    for r, adj in zip(rows, sig.holm([r["p"] for r in rows])):
        r["p_holm"] = adj
    out = [r"\begin{longtable}{llllrrrr}", r"\toprule",
           r"Comparison (A vs.\ B) & Model & Task & & $n$ & A & B & $p_{\mathrm{Holm}}$ \\", r"\midrule", r"\endhead"]
    for r in rows:
        p = r["p_holm"]
        ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        star = "$^*$" if p < 0.05 else ""
        out.append(f"{r['claim']} & {r['model']} & {r['task']} & & {r['n']} & {r['acc_a']:.1f} & {r['acc_b']:.1f} & {ps}{star} \\\\")
    out += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(out)


def prompt_table():
    d = "results/v1/Qwen3-VL-8B-Instruct-q4"
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{View} & \multicolumn{2}{c}{Swap} & \multicolumn{2}{c}{Stab.} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Second input & orig. & para. & orig. & para. & orig. & para. \\", r"\midrule"]
    for c, name in (("direct", "Img."), ("oracle", "Oracle"), ("wrong", "Misl.")):
        alt = load(f"{d}-prompt_alt/{c}.jsonl")
        base = load(f"{d}/{c}.jsonl")
        cells = []
        for t, _ in TASKS:
            ids = {i for i in alt if i.startswith(t)}
            cells += [fmt(acc(base, ids)), fmt(acc(alt, ids))]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def egocentric_table():
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Img.} & \multicolumn{2}{c}{Img.\ + oracle} & \multicolumn{2}{c}{Oracle only} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Model & same & flip & same & flip & same & flip \\", r"\midrule"]
    for name, m in MODELS:
        R = {c: load(f"results/v1/{m}/{c}.jsonl") for c in ("direct", "oracle", "oracle_only")}
        flip = {i for i, r in R["direct"].items() if r["task"] == "viewpoint" and r["meta"].get("answer_flips_from_current")}
        same = {i for i, r in R["direct"].items() if r["task"] == "viewpoint"} - flip
        cells = [fmt(acc(R[c], s)) for c in ("direct", "oracle", "oracle_only") for s in (same, flip)]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def small_model_table():
    m = "Qwen2.5-VL-3B-Instruct"
    conds = [("direct", "Img."), ("recon", "Own"), ("oracle", "Oracle"), ("oracle_only", "Oracle only"),
             ("shuffled", "Unrel."), ("wrong", "Misl.")]
    out = [r"\begin{tabular}{l" + "c" * len(conds) + "}", r"\toprule",
           "Task & " + " & ".join(n for _, n in conds) + r" \\", r"\midrule"]
    R = {c: load(f"results/v1/{m}/{c}.jsonl") for c, _ in conds}
    for t, tn in TASKS:
        cells = [fmt(acc({i: r for i, r in R[c].items() if r["task"] == t})) for c, _ in conds]
        out.append(f"{tn} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def camturn_table():
    """Physical outcome: second image from the original camera vs. one turned by 20 degrees."""
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{Oracle} & \multicolumn{3}{c}{Misl.} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"Model & Camera & stands & falls & all & stands & falls & all \\", r"\midrule"]
    for name, m in MODELS[:3]:
        for cam, d in (("same", f"results/v1/{m}"), ("turned", f"results_camturn/v1_camturn/{m}")):
            cells = []
            for c in ("oracle", "wrong"):
                R = {i: r for i, r in load(f"{d}/{c}.jsonl").items() if r["task"] == "stability"}
                for ans in ("stays standing", "falls over"):
                    cells.append(fmt(acc(R, {i for i, r in R.items() if r["answer"] == ans})))
                cells.append(fmt(acc(R)))
            out.append(f"{name if cam == 'same' else ''} & {cam} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main(out):
    parts = {"significance": significance_table(), "prompt": prompt_table(),
             "egocentric": egocentric_table(), "small": small_model_table(),
             "camturn": camturn_table()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for k, v in parts.items():
        open(out.replace(".tex", f"_{k}.tex"), "w").write(v + "\n")
        print(f"== {k}\n{v[:600]}\n")


if __name__ == "__main__":
    main(sys.argv[1])
