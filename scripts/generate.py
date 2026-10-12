"""Generate the benchmark: questions.jsonl plus original / imagined / wrong renders.

    python scripts/generate.py --n 20 --out data/pilot
    python scripts/generate.py --n 200 --out data/v1 --workers 8
"""
import argparse
import json
import os
import sys

# one numpy thread per worker; otherwise the pool oversubscribes the CPU badly
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imagine.render import Camera, render  # noqa: E402
from imagine.tasks import TASKS, generate  # noqa: E402

STATES = ("original", "imagined", "wrong")


def build(args):
    task, seed, out, ss = args
    item = generate(task, seed)
    folder = os.path.join(out, "images", item["id"])
    os.makedirs(folder, exist_ok=True)
    for s in STATES:
        path = os.path.join(folder, f"{s}.png")
        if not os.path.exists(path):
            img, _ = render(item[s]["scene"], Camera.from_dict(item[s]["camera"]), ss=ss)
            Image.fromarray(img).save(path)
        item[s]["image"] = os.path.relpath(path, out)
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="items per task")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--out", default="data/pilot")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--ss", type=int, default=2, help="supersampling")
    a = p.parse_args()

    jobs = [(t, a.seed0 + i, a.out, a.ss) for t in a.tasks for i in range(a.n)]
    with Pool(a.workers) as pool:
        items = pool.map(build, jobs, chunksize=1)
    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    for t in a.tasks:
        sub = [it for it in items if it["task"] == t]
        counts = {}
        for it in sub:
            counts[it["answer"]] = counts.get(it["answer"], 0) + 1
        print(f"{t:10s} n={len(sub):4d} answers={counts}")


if __name__ == "__main__":
    main()
