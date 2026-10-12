"""Question generators. Each task samples a scene, a question whose answer is
computed from the scene geometry, and three renderable states:

  original  - what the model is shown
  imagined  - the oracle "imagined" state: exactly what the question asks the
              model to picture (new viewpoint, rearranged scene, aftermath)
  wrong     - a plausible but incorrect imagined state, whose own answer
              differs from the true one (the misleading-imagination control)

Every state is (scene dict, camera dict), so rendering is a separate step.
"""
import copy

import numpy as np

from .render import COLORS, Camera, rot_y, rot_z, visible_fraction

SHAPES = ["cube", "sphere", "cylinder"]
ARENA = 2.4            # objects sit in the disc |xz| <= ARENA
GAP = 0.15             # minimum clearance between footprints
DEFAULT_ELEV, DEFAULT_DIST = 30.0, 8.5


class Reject(Exception):
    """Sample did not meet a quality constraint; the caller resamples."""


# ------------------------------------------------------------------ scenes

def make_object(shape, color, x, z, rng, scale=None):
    s = scale if scale is not None else rng.uniform(0.75, 1.05)
    name = f"{color} {shape}"
    if shape == "cube":
        return {"name": name, "shape": "box", "color": color, "center": [x, s / 2, z],
                "size": [s, s, s], "yaw": float(rng.uniform(0, 90))}
    if shape == "sphere":
        r = s / 2
        return {"name": name, "shape": "sphere", "color": color, "center": [x, r, z], "radius": r}
    if shape == "cylinder":
        r, h = s * 0.42, s * 1.05
        return {"name": name, "shape": "cylinder", "color": color, "center": [x, h / 2, z],
                "radius": r, "height": h}
    raise ValueError(shape)


def footprint(obj):
    """Radius of a disc that covers the object's floor footprint."""
    if obj["shape"] == "box":
        sx, _, sz = obj["size"]
        return 0.5 * float(np.hypot(sx, sz))
    return obj["radius"]


def xz(obj):
    return np.array([obj["center"][0], obj["center"][2]])


def overlaps(a, b, gap=GAP):
    return np.linalg.norm(xz(a) - xz(b)) < footprint(a) + footprint(b) + gap


def sample_scene(rng, n_min=4, n_max=6, tries=200):
    """Random non-overlapping objects with unique (color, shape) names."""
    n = int(rng.integers(n_min, n_max + 1))
    combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(combos)
    used_colors, objects = set(), []
    for color, shape in combos:
        if len(objects) == n:
            break
        if color in used_colors:          # one object per colour keeps names short and distinct
            continue
        for _ in range(tries):
            r, a = ARENA * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            obj = make_object(shape, color, r * np.cos(a), r * np.sin(a), rng)
            if not any(overlaps(obj, o) for o in objects):
                objects.append(obj)
                used_colors.add(color)
                break
    if len(objects) < n_min:
        raise Reject("could not place objects")
    return {"objects": objects}


def the(obj):
    return f"the {obj['name']}"


def state(scene, cam):
    return {"scene": scene, "camera": cam.to_dict()}


def all_visible(scene, cam, thresh=0.35):
    return min(visible_fraction(scene, cam)) >= thresh


# ------------------------------------------------------------ 1. viewpoint

VIEW_MOVES = {
    90: "walk a quarter of the way around the scene to your right",
    180: "walk around to the opposite side of the scene",
    -90: "walk a quarter of the way around the scene to your left",
}


def left_right(cam, a, b):
    ua, ub = cam.project(a["center"])[0], cam.project(b["center"])[0]
    return ("left" if ua < ub else "right"), abs(ua - ub) / cam.width


def gen_viewpoint(rng, variant, margin=0.08):
    """Perspective taking: left/right relation after the viewer moves."""
    scene = sample_scene(rng)
    cam0 = Camera.orbit(0, DEFAULT_ELEV, DEFAULT_DIST)
    move = int(rng.choice(list(VIEW_MOVES)))
    cam1 = Camera.orbit(move, DEFAULT_ELEV, DEFAULT_DIST)
    if not (all_visible(scene, cam0) and all_visible(scene, cam1)):
        raise Reject("occluded object")

    want = ["left", "right"][variant % 2]
    want_flip = bool((variant // 2) % 2)        # does the answer differ from the current view?
    objs = scene["objects"]
    pairs = [(a, b) for a in objs for b in objs if a is not b]
    rng.shuffle(pairs)
    for a, b in pairs:
        ans, gap1 = left_right(cam1, a, b)
        now, gap0 = left_right(cam0, a, b)
        if ans == want and gap1 >= margin and gap0 >= margin and (ans != now) == want_flip:
            break
    else:
        raise Reject("no pair with the wanted answer")

    # wrong imagination: a different move under which the answer flips
    wrong = [m for m in VIEW_MOVES if m != move
             and left_right(Camera.orbit(m, DEFAULT_ELEV, DEFAULT_DIST), a, b)[0] != ans]
    if not wrong:
        raise Reject("no misleading viewpoint")
    wm = int(rng.choice(wrong))
    camw = Camera.orbit(wm, DEFAULT_ELEV, DEFAULT_DIST)

    q = (f"Imagine you {VIEW_MOVES[move]}, keeping your eyes on the center of the scene. "
         f"From that new viewpoint, is {the(a)} to the left or to the right of {the(b)}?")
    return {
        "task": "viewpoint", "question": q, "choices": ["left", "right"], "answer": ans,
        "original": state(scene, cam0), "imagined": state(scene, cam1), "wrong": state(scene, camw),
        "wrong_answer": "left" if ans == "right" else "right",
        "meta": {"move": move, "wrong_move": wm, "answer_flips_from_current": ans != now,
                 "screen_gap": round(gap1, 3)},
    }


# ----------------------------------------------------------------- 2. swap

def nearest(objs, target):
    d = sorted((np.linalg.norm(xz(o) - xz(target)), o["name"]) for o in objs if o is not target)
    return d


def swapped(scene, i, j):
    s = copy.deepcopy(scene)
    a, b = s["objects"][i], s["objects"][j]
    (ax, _, az), (bx, _, bz) = a["center"], b["center"]
    a["center"][0], a["center"][2] = bx, bz
    b["center"][0], b["center"][2] = ax, az
    objs = s["objects"]
    if any(overlaps(objs[k], objs[m]) for k in range(len(objs)) for m in range(k + 1, len(objs))):
        return None
    return s


def gen_swap(rng, variant, ratio=1.4):
    """Mental rearrangement: the target trades places with another object; which
    object is then closest to the target?

    The target is always one of the two that move, and the answer may be neither
    its swap partner nor its nearest neighbour before the swap. Otherwise "pick
    one of the swapped objects" or "pick the current nearest" would be shortcuts.
    """
    scene = sample_scene(rng, 5, 6)
    cam = Camera.orbit(0, DEFAULT_ELEV, DEFAULT_DIST)
    objs = scene["objects"]
    n = len(objs)

    def outcome(k, j):
        after = swapped(scene, k, j)
        if after is None:
            return None, None, None
        d = nearest(after["objects"], after["objects"][k])
        return after, d[0][1], d[1][0] / d[0][0]

    pairs = [(k, j) for k in range(n) for j in range(n) if k != j]
    rng.shuffle(pairs)
    for k, j in pairs:
        after, ans, r = outcome(k, j)
        before = nearest(objs, objs[k])[0][1]
        if after is not None and r >= ratio and ans not in (objs[j]["name"], before):
            break
    else:
        raise Reject("no informative swap")
    if not (all_visible(scene, cam) and all_visible(after, cam)):
        raise Reject("occluded object")

    # wrong imagination: the target trades places with someone else instead
    wrong = wrong_ans = None
    for j2 in rng.permutation(n):
        if j2 in (k, j):
            continue
        w, a2, _ = outcome(k, int(j2))
        if w is not None and a2 != ans:
            wrong, wrong_ans = w, a2
            break
    if wrong is None:
        raise Reject("no misleading swap")

    target, partner = objs[k], objs[j]
    first, second = (target, partner) if variant % 2 == 0 else (partner, target)
    # the partner can never be the answer (it ends up where the target was); listing it
    # anyway drew most answers in a first run, a pure name-salience effect, so leave it out
    choices = [o["name"] for o in objs if o is not target and o is not partner]
    q = (f"Imagine {the(first)} and {the(second)} trade places. "
         f"After the swap, which object would be closest to {the(target)} in its new position?")
    return {
        "task": "swap", "question": q, "choices": choices, "answer": ans,
        "original": state(scene, cam), "imagined": state(after, cam), "wrong": state(wrong, cam),
        "wrong_answer": wrong_ans,
        "meta": {"swap": [first["name"], second["name"]], "target": target["name"],
                 "partner": partner["name"], "answer_before_swap": before, "dist_ratio": round(float(r), 3)},
    }


# ------------------------------------------------------------ 3. stability

BLOCK_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink"]


def stack_failure(blocks, margin):
    """Lowest level whose load's centre of mass lies outside its top face (x only).
    Returns (level or None, min |slack|). A positive slack means inside."""
    vol = [np.prod(b["size"]) for b in blocks]
    worst, fail = np.inf, None
    for k in range(len(blocks) - 1):
        above = range(k + 1, len(blocks))
        com = sum(vol[m] * blocks[m]["center"][0] for m in above) / sum(vol[m] for m in above)
        half = blocks[k]["size"][0] / 2
        slack = half - abs(com - blocks[k]["center"][0])
        worst = min(worst, abs(slack))
        if slack < 0 and fail is None:
            fail = k
    if worst < margin:
        raise Reject("too close to the tipping point")
    return fail


def fallen(blocks, level, rng, obstacles=(), tries=200, strict=True):
    """Aftermath: everything above `level` lies flat on the floor, scattered on
    the overhang side of the remaining stack."""
    out = copy.deepcopy(blocks[: level + 1])
    base = blocks[level]
    vol = [np.prod(b["size"]) for b in blocks]
    above = blocks[level + 1:]
    com = sum(vol[level + 1 + m] * b["center"][0] for m, b in enumerate(above)) / sum(vol[level + 1:])
    side = 1.0 if com > base["center"][0] else -1.0
    edge = max(side * (b["center"][0] - base["center"][0]) + b["size"][0] / 2 for b in out)
    for b in above:
        f = copy.deepcopy(b)
        f.pop("rot", None)
        f["yaw"] = float(rng.uniform(-40, 40))
        for _ in range(tries):
            dx = rng.uniform(0.25, 1.9)
            f["center"] = [base["center"][0] + side * (edge + dx), f["size"][1] / 2, float(rng.uniform(-0.5, 1.3))]
            if abs(f["center"][0]) + footprint(f) <= 3.0 and not any(overlaps(f, o, 0.05) for o in out + list(obstacles)):
                break
        else:
            if strict:
                raise Reject("no room for fallen block")
            # recon path: a cramped aftermath beats no aftermath; keep the last spot tried
        out.append(f)
    return out


def gen_stability(rng, variant, margin=0.08):
    """Physical outcome: does a stack of blocks stay up?"""
    want_stable = variant % 2 == 0
    n = int(rng.integers(3, 5))
    colors = list(rng.choice(BLOCK_COLORS, n, replace=False))
    blocks, y, x = [], 0.0, 0.0
    for m in range(n):
        w, h, d = rng.uniform(0.9, 1.5), rng.uniform(0.28, 0.42), rng.uniform(0.7, 1.0)
        if m:
            x += rng.uniform(-0.55, 0.55) * blocks[-1]["size"][0]
        blocks.append({"name": f"{colors[m]} block", "shape": "box", "color": str(colors[m]),
                       "center": [x, y + h / 2, 0.0], "size": [w, h, d], "yaw": 0.0})
        y += h
    shift = -np.mean([b["center"][0] for b in blocks])  # keep the stack centred in view
    for b in blocks:
        b["center"][0] += shift
    level = stack_failure(blocks, margin)
    if (level is None) != want_stable:
        raise Reject("wrong outcome for balance")

    # a couple of distractors on the floor, well away from the stack
    extras = []
    for color in rng.choice([c for c in COLORS if c not in colors], int(rng.integers(1, 3)), replace=False):
        for _ in range(50):
            o = make_object(str(rng.choice(["cube", "sphere", "cylinder"])), str(color),
                            rng.uniform(-2.6, 2.6), rng.uniform(-2.2, -1.4), rng, scale=0.6)
            if abs(o["center"][0]) > 1.6 and not any(overlaps(o, e) for e in extras):
                extras.append(o)
                break

    cam = Camera.orbit(0, 20.0, DEFAULT_DIST, target=(0, 0.6, 0))
    scene = {"objects": blocks + extras}
    if level is None:
        imagined = scene
        # misleading aftermath: the top block falls off the side it leans towards
        wrong = {"objects": fallen(blocks, n - 2, rng, extras) + extras}
    else:
        imagined = {"objects": fallen(blocks, level, rng, extras) + extras}
        wrong = scene
    for st in (imagined, wrong):                      # fallen blocks must stay in frame
        if max(abs(o["center"][0]) + footprint(o) for o in st["objects"]) > 3.0:
            raise Reject("aftermath out of frame")
    ans = "stays standing" if level is None else "falls over"
    q = ("This stack of blocks is released and nothing else touches it. "
         "Will the stack stay standing, or will it fall over?")
    return {
        "task": "stability", "question": q, "choices": ["stays standing", "falls over"], "answer": ans,
        "original": state(scene, cam), "imagined": state(imagined, cam), "wrong": state(wrong, cam),
        "wrong_answer": "falls over" if level is None else "stays standing",
        "meta": {"n_blocks": n, "failure_level": level},
    }


TASKS = {"viewpoint": gen_viewpoint, "swap": gen_swap, "stability": gen_stability}


def generate(task, seed, max_tries=500):
    """Deterministic: the same (task, seed) always gives the same item. The target
    answer class comes from the seed, not from the rejection loop, so classes stay
    balanced even when one outcome is much easier to sample."""
    rng = np.random.default_rng([list(TASKS).index(task), seed])   # not hash(): it is salted per process
    for _ in range(max_tries):
        try:
            item = TASKS[task](rng, seed)
            item["id"] = f"{task}_{seed:05d}"
            return item
        except Reject:
            continue
    raise RuntimeError(f"{task} seed {seed}: no valid sample in {max_tries} tries")
