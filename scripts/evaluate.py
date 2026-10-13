"""Ask a VLM every question under each condition; one results file per (model, condition).

Conditions (images given, in order):
  direct      original                    answer immediately
  cot         original                    picture the change in words first, then answer
  oracle      original + imagined         the exact state the question asks about
  oracle_only imagined                    is the original even needed?
  wrong       original + wrong            plausible imagined state with a different answer
  shuffled    original + imagined of another question of the same task
  recon       original + the model's own reconstruction, transformed and rendered
              (scripts/recon.py first; if that failed for a question, falls back to direct)
Every two-image condition uses the same wording, so the model cannot tell oracle,
wrong and shuffled apart from the prompt.

Backends:
  hf      transformers on a local GPU   --model Qwen/Qwen3-VL-8B-Instruct
  openai  any OpenAI-compatible server  --base-url http://localhost:8000/v1
  mock    no model; answers from the second image's label (pipeline test only)

    python scripts/evaluate.py data/v1 --backend hf --model Qwen/Qwen3-VL-8B-Instruct \\
        --conditions direct oracle wrong shuffled cot
Resumable: questions already in the results file are skipped.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import time

CONDITIONS = ["direct", "cot", "oracle", "oracle_only", "wrong", "shuffled", "recon"]

SECOND_IMAGE = {
    "viewpoint": "The second image shows the scene from the new viewpoint described in the question.",
    "swap": "The second image shows the scene after the two objects have traded places.",
    "stability": "The second image shows the scene after the stack was released and everything came to rest.",
}
ONLY_IMAGE = {
    "viewpoint": "The image shows the scene from the new viewpoint described in the question.",
    "swap": "The image shows the scene after the two objects have traded places.",
    "stability": "The image shows the scene after the stack was released and everything came to rest.",
}
COT = {
    "viewpoint": "First describe, step by step, how the scene would look from the new viewpoint, "
                 "then answer.",
    "swap": "First describe, step by step, where every object would be after the swap, then answer.",
    "stability": "First reason, step by step, about where the weight of the upper blocks rests "
                 "and what would happen after release, then answer.",
}
MAX_TOKENS = {"cot": 1024}
DEFAULT_MAX_TOKENS = 48

# Paraphrased wording for the prompt-robustness check (--prompt-variant alt). Same
# information, different phrasing, and the options listed in reverse order.
ALT_SECOND_IMAGE = {
    "viewpoint": "Image 2 is a rendering of this scene as it looks from the position the question asks you to walk to.",
    "swap": "Image 2 is a rendering of this scene once the two named objects have swapped positions.",
    "stability": "Image 2 is a rendering of this scene after the blocks were let go and stopped moving.",
}
ALT_ONLY_IMAGE = {k: v.replace("Image 2 is a rendering of this scene", "This image is a rendering of a scene")
                  for k, v in ALT_SECOND_IMAGE.items()}


# ------------------------------------------------------------------ prompts

def build(item, condition, root, shuffle_pool, recon=None, variant="default"):
    img = lambda st, it=item: os.path.join(root, it[st]["image"])  # noqa: E731
    alt = variant == "alt"
    opts = " / ".join(reversed(item["choices"]) if alt else item["choices"])
    task = item["task"]
    if condition == "recon" and not (recon or {}).get(item["id"]):
        condition = "direct"                       # reconstruction failed: no imagined image
    if condition in ("direct", "cot"):
        images, intro = [img("original")], ("Here is a rendered scene." if alt else "The image shows a scene.")
    elif condition == "oracle_only":
        images, intro = [img("imagined")], (ALT_ONLY_IMAGE if alt else ONLY_IMAGE)[task]
    else:
        if condition == "oracle":
            second = img("imagined")
        elif condition == "wrong":
            second = img("wrong")
        elif condition == "shuffled":
            second = img("imagined", shuffle_pool[item["id"]])
        elif condition == "recon":
            second = os.path.join(root, recon[item["id"]])
        else:
            raise ValueError(condition)
        images = [img("original"), second]
        intro = ("Image 1 is the scene as it is now. " + ALT_SECOND_IMAGE[task] if alt else
                 "The first image shows the current scene. " + SECOND_IMAGE[task])
    if condition == "cot":
        tail = f"{COT[task]} End your reply with a final line of the form 'Answer: <option>'."
    elif alt:
        tail = "Respond with exactly one line: 'Answer: ' followed by one of the options."
    else:
        tail = "Reply with only a single line of the form 'Answer: <option>'."
    label = "Choose from" if alt else "Options"
    text = f"{intro}\n\n{item['question']}\n{label}: {opts}\n\n{tail}"
    return images, text


def shuffle_partners(items, seed=0):
    """Each question gets the imagined image of another question of the same task (a derangement)."""
    rng = random.Random(seed)
    partner = {}
    for task in {it["task"] for it in items}:
        group = [it for it in items if it["task"] == task]
        order = group[:]
        rng.shuffle(order)
        for a, b in zip(order, order[1:] + order[:1]):     # rotate by one: nobody keeps their own
            partner[a["id"]] = b
    return partner


def parse(text, choices):
    """Option named on the last 'Answer:' line; else the last option mentioned anywhere."""
    by_len = sorted(choices, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(c) for c in by_len), re.I)
    lines = [ln for ln in text.splitlines() if re.search(r"answer\s*:", ln, re.I)]
    for chunk in ([lines[-1].split(":", 1)[1]] if lines else []) + [text]:
        hits = pat.findall(chunk)
        if hits:
            last = hits[-1].lower()
            return next(c for c in choices if c.lower() == last), chunk is not text
    return None, False


def select(items, tasks=None, limit=None):
    if tasks:
        items = [it for it in items if it["task"] in tasks]
    if limit:
        items = [it for t in dict.fromkeys(it["task"] for it in items)
                 for it in [x for x in items if x["task"] == t][:limit]]
    return items


# ----------------------------------------------------------------- backends

class HF:
    def __init__(self, model, quant="none", max_pixels=1280 * 28 * 28):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model, max_pixels=max_pixels)
        kw = {}
        if quant in ("8", "4"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True) if quant == "8" else
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16))
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="cuda", **kw).eval()

    def __call__(self, images, text, max_tokens):
        from PIL import Image
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
        content.append({"type": "text", "text": text})
        inputs = self.proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True)[0], int(new.shape[1])


class OpenAICompat:
    def __init__(self, model, base_url):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))

    def __call__(self, images, text, max_tokens):
        content = []
        for p in images:
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": text})
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens, temperature=0)
        return r.choices[0].message.content or "", r.usage.completion_tokens if r.usage else 0


class Mock:
    """Answers whatever the second image depicts; random with one image.
    Expected accuracy: oracle 1.0, wrong 0.0, shuffled ~chance, direct ~chance."""

    def __init__(self, items, root):
        self.label = {}
        for it in items:
            for st, ans in (("imagined", it["answer"]), ("wrong", it["wrong_answer"])):
                self.label[os.path.join(root, it[st]["image"])] = ans
        self.choices = {it["question"]: it["choices"] for it in items}
        self.rng = random.Random(0)

    def __call__(self, images, text, max_tokens):
        ans = self.label.get(images[-1])           # None for the original image
        if ans is None:
            q = next(q for q in self.choices if q in text)
            ans = self.rng.choice(self.choices[q])
        return f"Answer: {ans}", 3


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset folder with questions.jsonl")
    p.add_argument("--backend", choices=["hf", "openai", "mock"], default="mock")
    p.add_argument("--model", default="mock")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--quant", default="none", choices=["none", "8", "4"])
    p.add_argument("--conditions", nargs="+", default=["direct", "oracle", "wrong", "shuffled"])
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--limit", type=int, help="first N questions per task (smoke tests)")
    p.add_argument("--out", default="results")
    p.add_argument("--recon", help="recon/<tag> folder to use for the recon condition")
    p.add_argument("--max-tokens", type=int, help="override the generation limit for every condition")
    p.add_argument("--redo-truncated", action="store_true",
                   help="first drop rows that hit the old limit (kept in <cond>.truncated.jsonl), then rerun them")
    p.add_argument("--prompt-variant", choices=["default", "alt"], default="default")
    a = p.parse_args()

    items = [json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))]
    items = select(items, a.tasks, a.limit)
    pool = shuffle_partners(items)
    recon = None
    if "recon" in a.conditions:
        idx = os.path.join(a.root, "recon", a.recon or "", "index.jsonl")
        recon = {r["id"]: r["image"] for r in map(json.loads, open(idx))}

    if a.backend == "hf":
        call = HF(a.model, a.quant)
    elif a.backend == "openai":
        call = OpenAICompat(a.model, a.base_url)
    else:
        call = Mock(items, a.root)

    tag = a.model.split("/")[-1] + ("" if a.quant == "none" else f"-q{a.quant}")
    if a.prompt_variant != "default":
        tag += f"-prompt_{a.prompt_variant}"
    outdir = os.path.join(a.out, os.path.basename(os.path.normpath(a.root)), tag)
    os.makedirs(outdir, exist_ok=True)
    for cond in a.conditions:
        path = os.path.join(outdir, f"{cond}.jsonl")
        limit = a.max_tokens or MAX_TOKENS.get(cond, DEFAULT_MAX_TOKENS)
        if a.redo_truncated and os.path.exists(path):
            rows = [json.loads(line) for line in open(path)]
            cut = [r for r in rows if r["gen_tokens"] >= MAX_TOKENS.get(cond, DEFAULT_MAX_TOKENS)]
            with open(path.replace(".jsonl", ".truncated.jsonl"), "a") as g:
                g.writelines(json.dumps(r) + "\n" for r in cut)
            with open(path, "w") as g:
                g.writelines(json.dumps(r) + "\n" for r in rows if r not in cut)
            print(f"{cond}: rerunning {len(cut)} truncated answers with max_tokens={limit}")
        done = set()
        if os.path.exists(path):
            done = {json.loads(line)["id"] for line in open(path)}
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with open(path, "a") as f:
            for n, it in enumerate(todo, 1):
                images, text = build(it, cond, a.root, pool, recon, a.prompt_variant)
                reply, n_tok = call(images, text, limit)
                pred, strict = parse(reply, it["choices"])
                rec = {"id": it["id"], "task": it["task"], "condition": cond, "pred": pred,
                       "answer": it["answer"], "wrong_answer": it["wrong_answer"],
                       "correct": pred == it["answer"], "parsed_strict": strict,
                       "gen_tokens": n_tok, "max_tokens": limit, "reply": reply, "meta": it.get("meta", {})}
                if cond == "recon":
                    rec["recon_failed"] = not recon.get(it["id"])
                    if cond == "recon" and not rec["recon_failed"]:
                        rec["recon_image"] = recon[it["id"]]
                if cond == "shuffled":
                    rec["partner"] = pool[it["id"]]["id"]
                    pa = pool[it["id"]]["answer"]      # only meaningful when it is one of our options
                    rec["partner_answer"] = pa if pa in it["choices"] else None
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"{tag} {cond:12s} {n}/{len(todo)}  {rate:.2f}s/q", flush=True)
    print("results in", outdir)


if __name__ == "__main__":
    main()
