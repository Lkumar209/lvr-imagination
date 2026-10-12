"""Stage 1 of the model-built imagination condition: ask the VLM to describe each
original scene as numbers, apply the question's change, render the result.

Writes <root>/recon/<model tag>/<id>.png and index.jsonl. Afterwards run
evaluate.py with `--conditions recon --recon <model tag>`.

    python scripts/recon.py data/v1 --backend hf --model Qwen/Qwen3-VL-8B-Instruct --quant 4
Resumable.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import HF, OpenAICompat, select  # noqa: E402
from imagine.recon import imagine, prompt_for  # noqa: E402
from imagine.render import Camera, render  # noqa: E402
from imagine.tasks import Reject  # noqa: E402


class MockRecon:
    """Describes the true scene with noise: a pipeline test, not a model."""

    def __init__(self, items, root, noise=0.3):
        import numpy as np
        self.np, self.noise = np, noise
        self.by_image = {os.path.join(root, it["original"]["image"]): it for it in items}
        self.rng = np.random.default_rng(0)

    def __call__(self, images, text, max_tokens):
        it = self.by_image[images[0]]
        objs = it["original"]["scene"]["objects"]
        n = lambda: float(self.rng.normal(0, self.noise))  # noqa: E731
        if it["task"] == "stability":
            out = [{"color": o["color"], "width": o["size"][0], "x": o["center"][0] + n()}
                   for o in objs if o["name"].endswith("block")]
        else:
            out = [{"name": o["name"], "x": o["center"][0] + n(), "z": o["center"][2] + n()} for o in objs]
        return json.dumps(out), 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--backend", choices=["hf", "openai", "mock"], default="mock")
    p.add_argument("--model", default="mock")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--quant", default="none", choices=["none", "8", "4"])
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--rescale-from", metavar="TAG",
                   help="no model calls: reuse the replies in recon/TAG, spread each layout to "
                        "scene size, write recon/TAG-rescaled")
    a = p.parse_args()

    items = select([json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))],
                   a.tasks, a.limit)
    if a.rescale_from:
        replies = {r["id"]: r["reply"] for r in map(json.loads, open(
            os.path.join(a.root, "recon", a.rescale_from, "index.jsonl")))}
        items = [it for it in items if it["id"] in replies]
        call = lambda images, text, _: (replies[by_img[images[0]]], 0)  # noqa: E731
        by_img = {os.path.join(a.root, it["original"]["image"]): it["id"] for it in items}
    elif a.backend == "hf":
        call = HF(a.model, a.quant)
    elif a.backend == "openai":
        call = OpenAICompat(a.model, a.base_url)
    else:
        call = MockRecon(items, a.root)

    tag = a.model.split("/")[-1] + ("" if a.quant == "none" else f"-q{a.quant}")
    if a.rescale_from:
        tag = a.rescale_from + "-rescaled"
    outdir = os.path.join(a.root, "recon", tag)
    os.makedirs(outdir, exist_ok=True)
    index = os.path.join(outdir, "index.jsonl")
    done = {json.loads(line)["id"] for line in open(index)} if os.path.exists(index) else set()

    fails = 0
    with open(index, "a") as f:
        for n, it in enumerate([it for it in items if it["id"] not in done], 1):
            reply, _ = call([os.path.join(a.root, it["original"]["image"])], prompt_for(it), 768)
            rec = {"id": it["id"], "task": it["task"], "reply": reply}
            try:
                scene, cam = imagine(it, reply, normalize=bool(a.rescale_from))
                img, _ = render(scene, Camera.from_dict(cam), ss=2)
                path = os.path.join(outdir, f"{it['id']}.png")
                Image.fromarray(img).save(path)
                rec.update(image=os.path.relpath(path, a.root), scene=scene)
            except Reject as e:
                rec.update(image=None, error=str(e))
                fails += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"recon {tag} {n} done, {fails} failed", flush=True)
    print(f"recon {tag}: {fails} failures this run; index at {index}")


if __name__ == "__main__":
    main()
