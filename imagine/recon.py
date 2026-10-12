"""Model-built imagination: the VLM writes the scene down as numbers, we apply the
change described in the question and render the result with our own renderer.

The rendered image then takes the place of the oracle image. Its quality is
limited only by the model's own perception of the scene, which separates
"could it see the scene?" from "could it reason once it had the picture?".

Nothing here reads the ground-truth scene except the question's transformation
(which camera move, which two objects, the stack-or-not task), all of which the
question text gives the model anyway. Sizes are fixed defaults, never copied.
"""
import json
import re

import numpy as np

from .render import COLORS, Camera
from .tasks import DEFAULT_DIST, DEFAULT_ELEV, Reject, fallen, make_object, stack_failure

SCENE_PROMPT = (
    "List every object in the image. For each one give its name as '<color> <shape>' "
    "(shape is one of: cube, sphere, cylinder) and where it stands on the floor, seen from above, "
    "in meters: x runs from left (negative) to right (positive) as seen in the image, and z runs "
    "from far away (negative) to close to the camera (positive). All objects are within about "
    "2.5 m of the center of the scene. Reply with only a JSON list, for example "
    '[{"name": "red cube", "x": 0.5, "z": -1.0}, {"name": "blue sphere", "x": -1.2, "z": 0.8}]'
)
STACK_PROMPT = (
    "Look at the stack of flat blocks (ignore any objects lying apart from the stack). List the "
    "blocks from the bottom of the stack to the top. For each give its color, its width in meters "
    "(blocks are 0.9 to 1.5 m wide), and the horizontal position of its center in meters, from "
    "left (negative) to right (positive) as seen in the image. Reply with only a JSON list, for "
    'example [{"color": "red", "width": 1.2, "x": 0.0}, {"color": "blue", "width": 1.0, "x": 0.3}]'
)
SHAPE_WORDS = {"cube": "cube", "box": "cube", "block": "cube", "sphere": "sphere", "ball": "sphere",
               "cylinder": "cylinder", "can": "cylinder"}
BLOCK_H, BLOCK_D, SCALE = 0.35, 0.85, 0.9


def prompt_for(item):
    return STACK_PROMPT if item["task"] == "stability" else SCENE_PROMPT


def parse_json_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise Reject("no JSON list")
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise Reject(f"bad JSON: {e}")
    if not isinstance(out, list) or not out:
        raise Reject("empty list")
    return out


def _num(v, lo, hi):
    v = float(v)
    if not np.isfinite(v):
        raise Reject("non-finite number")
    return float(np.clip(v, lo, hi))


def rescale(objs, radius=2.0):
    """Spread a reconstructed layout to the arena's size about its centroid. Keeps the
    model's relative layout and fixes only its overall scale, which VLMs badly
    underestimate (the prompt states the objects span about 2.5 m)."""
    xz = np.array([[o["center"][0], o["center"][2]] for o in objs])
    c = xz.mean(0)
    r = np.linalg.norm(xz - c, axis=1).max()
    if r < 1e-6:
        return
    for o, p in zip(objs, (xz - c) * (radius / r)):
        o["center"][0], o["center"][2] = float(p[0]), float(p[1])


def scene_from_reply(text, rng, normalize=False):
    objs, seen = [], set()
    for e in parse_json_list(text):
        words = str(e.get("name", "")).lower().split()
        color = next((w for w in words if w in COLORS), None)
        shape = next((SHAPE_WORDS[w] for w in words if w in SHAPE_WORDS), None)
        if color is None or shape is None or (color, shape) in seen:
            continue
        seen.add((color, shape))
        objs.append(make_object(shape, color, _num(e.get("x", 0), -3, 3), _num(e.get("z", 0), -3, 3),
                                rng, scale=SCALE))
    if not objs:
        raise Reject("no usable objects")
    if normalize:
        rescale(objs)
    return {"objects": objs}


def stack_from_reply(text):
    blocks, y = [], 0.0
    for e in parse_json_list(text):
        color = str(e.get("color", "")).lower().strip()
        if color not in COLORS:
            continue
        w = _num(e.get("width", 1.2), 0.5, 2.0)
        blocks.append({"name": f"{color} block", "shape": "box", "color": color,
                       "center": [_num(e.get("x", 0), -3, 3), y + BLOCK_H / 2, 0.0],
                       "size": [w, BLOCK_H, BLOCK_D], "yaw": 0.0})
        y += BLOCK_H
    if len(blocks) < 2:
        raise Reject("fewer than two blocks")
    return blocks


def imagine(item, reply, seed=0, normalize=False):
    """VLM reply describing the original scene -> (scene, camera dict) of the imagined state."""
    rng = np.random.default_rng(seed)
    task, meta = item["task"], item["meta"]
    if task == "viewpoint":
        cam = Camera.orbit(meta["move"], DEFAULT_ELEV, DEFAULT_DIST)
        return scene_from_reply(reply, rng, normalize), cam.to_dict()
    if task == "swap":
        scene = scene_from_reply(reply, rng, normalize)
        by_name = {o["name"]: o for o in scene["objects"]}
        a, b = meta["swap"]
        if a not in by_name or b not in by_name:
            raise Reject("swapped objects missing from the reconstruction")
        pa, pb = by_name[a]["center"], by_name[b]["center"]
        pa[0], pb[0] = pb[0], pa[0]
        pa[2], pb[2] = pb[2], pa[2]
        return scene, item["original"]["camera"]
    if task == "stability":
        blocks = stack_from_reply(reply)
        level = stack_failure(blocks, margin=0.0)
        objs = blocks if level is None else fallen(blocks, level, rng, strict=False)
        return {"objects": objs}, item["original"]["camera"]
    raise ValueError(task)
