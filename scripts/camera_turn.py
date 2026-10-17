"""Stability control: render every second image (oracle and misleading) from a camera
turned by --deg, so no second image is a pixel copy of the original.

For stable stacks the oracle otherwise equals the original, and for falling stacks the
misleading image does. A model that only checks whether the picture changed would then
answer "falls over" for every turned image; a model that reads the blocks would not.

    python scripts/camera_turn.py data/v1 data/v1_camturn --deg 20
    python scripts/evaluate.py data/v1_camturn --backend hf --model $M --quant 4 \\
        --conditions oracle wrong --out results_camturn
"""
import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image  # noqa: E402

from imagine.render import Camera, render  # noqa: E402

ELEV, TARGET = 20.0, (0, 0.6, 0)                     # as in tasks.gen_stability
DIST = 8.5                                            # tasks.DEFAULT_DIST


def job(args):
    root, out, item, deg, ss = args
    folder = os.path.join(out, "images", item["id"])
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(root, item["original"]["image"])
    dst = os.path.join(folder, "original.png")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    item["original"]["image"] = os.path.relpath(dst, out)
    cam = Camera.orbit(deg, ELEV, DIST, target=TARGET)
    for s in ("imagined", "wrong"):
        path = os.path.join(folder, f"{s}.png")
        if not os.path.exists(path):
            img, _ = render(item[s]["scene"], cam, ss=ss)
            Image.fromarray(img).save(path)
        item[s]["camera"] = cam.to_dict()
        item[s]["image"] = os.path.relpath(path, out)
    item["meta"]["camera_turn_deg"] = deg
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--deg", type=float, default=20.0)
    p.add_argument("--ss", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    items = [json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))]
    items = [it for it in items if it["task"] == "stability"][:a.limit]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        items = pool.map(job, [(a.root, a.out, it, a.deg, a.ss) for it in items])
    with open(os.path.join(a.out, "questions.jsonl"), "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)
    print(f"{len(items)} stability questions -> {a.out}")


if __name__ == "__main__":
    main()
