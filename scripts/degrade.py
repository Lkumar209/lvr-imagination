"""Degraded oracles: the true scene with Gaussian position noise (sigma, metres),
transformed and rendered exactly like the oracle. Sweeping sigma measures how
faithful an imagined state must be before it stops helping.

Each sigma gets a folder in the recon format, so evaluate.py reads it with
`--conditions recon --recon noise_<sigma>`. The index also stores the answer the
noisy imagination itself implies (`implied`), to separate "the imagined image was
wrong" from "the model misread a correct image".

    python scripts/degrade.py data/v1 --sigmas 0 0.1 0.25 0.5 1.0 --workers 48
"""
import argparse
import copy
import json
import os
import re
import sys
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imagine.render import Camera, render  # noqa: E402
from imagine.tasks import fallen, left_right, nearest, stack_failure  # noqa: E402

LR = re.compile(r"is the (.+?) to the left or to the right of the (.+?)\?")


def jitter(objs, sigma, rng, axes=(0, 2)):
    for o in objs:
        for a in axes:
            o["center"][a] += float(rng.normal(0, sigma))


def degrade(item, sigma, seed):
    """-> (scene, camera dict, answer implied by this imagined state)."""
    rng = np.random.default_rng([seed, int(round(sigma * 1000))])
    scene = copy.deepcopy(item["original"]["scene"])
    objs = scene["objects"]
    by = {o["name"]: o for o in objs}
    task = item["task"]
    if task == "viewpoint":
        jitter(objs, sigma, rng)
        cam = item["imagined"]["camera"]
        a, b = LR.search(item["question"]).groups()
        return scene, cam, left_right(Camera.from_dict(cam), by[a], by[b])[0]
    if task == "swap":
        jitter(objs, sigma, rng)
        t, p = by[item["meta"]["target"]], by[item["meta"]["partner"]]
        for k in (0, 2):
            t["center"][k], p["center"][k] = p["center"][k], t["center"][k]
        return scene, item["original"]["camera"], nearest(objs, t)[0][1]
    if task == "stability":
        blocks = [o for o in objs if o["name"].endswith("block")]
        extras = [o for o in objs if not o["name"].endswith("block")]
        jitter(blocks[1:], sigma, rng, axes=(0,))       # misjudged offsets; the base stays put
        level = stack_failure(blocks, margin=0.0)
        if level is not None:
            objs = fallen(blocks, level, rng, extras, strict=False) + extras
        return ({"objects": objs}, item["original"]["camera"],
                "stays standing" if level is None else "falls over")
    raise ValueError(task)


def job(args):
    root, item, sigma, n = args
    tag = f"noise_{sigma:g}"
    path = os.path.join(root, "recon", tag, f"{item['id']}.png")
    scene, cam, implied = degrade(item, sigma, n)
    if not os.path.exists(path):
        img, _ = render(scene, Camera.from_dict(cam), ss=2)
        Image.fromarray(img).save(path)
    return tag, {"id": item["id"], "task": item["task"], "sigma": sigma,
                 "image": os.path.relpath(path, root), "implied": implied,
                 "implied_correct": implied == item["answer"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    items = [json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))]
    for s in a.sigmas:
        os.makedirs(os.path.join(a.root, "recon", f"noise_{s:g}"), exist_ok=True)
    jobs = [(a.root, it, s, n) for s in a.sigmas for n, it in enumerate(items)]
    with Pool(a.workers) as pool:
        out = pool.map(job, jobs, chunksize=4)
    for s in a.sigmas:
        tag = f"noise_{s:g}"
        rows = [r for t, r in out if t == tag]
        with open(os.path.join(a.root, "recon", tag, "index.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        acc = {t: np.mean([r["implied_correct"] for r in rows if r["task"] == t])
               for t in ("viewpoint", "swap", "stability")}
        print(f"sigma={s:<5g} imagined state still implies the right answer: "
              + "  ".join(f"{t}={100 * v:.0f}%" for t, v in acc.items()))


if __name__ == "__main__":
    main()
