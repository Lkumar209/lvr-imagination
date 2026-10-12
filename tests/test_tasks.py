"""Checks that answers are computed from geometry correctly.  Run: python -m pytest tests"""
import numpy as np

from imagine.render import Camera, render
from imagine.tasks import TASKS, fallen, generate, left_right, overlaps, stack_failure


def obj(name, x, z):
    return {"name": name, "shape": "sphere", "color": "red", "center": [x, 0.5, z], "radius": 0.5}


def test_left_right_follows_viewer_motion():
    a, b = obj("a", -1.5, 0), obj("b", 1.5, 0)          # a on the left from the front
    assert left_right(Camera.orbit(0), a, b)[0] == "left"
    assert left_right(Camera.orbit(180), a, b)[0] == "right"
    front, back = obj("f", 0, 1.5), obj("k", 0, -1.5)    # f is nearer the default viewer
    # walking right puts the viewer on +x, looking towards -x: +z is now on their left
    assert left_right(Camera.orbit(90), front, back)[0] == "left"
    assert left_right(Camera.orbit(-90), front, back)[0] == "right"


def test_render_puts_object_on_the_projected_side():
    scene = {"objects": [obj("a", 2.0, 0)]}
    cam = Camera.orbit(0, width=64, height=64)
    _, ids = render(scene, cam, ss=1)
    cols = np.nonzero(ids == 0)[1]
    assert cols.mean() > 32 and abs(cols.mean() - cam.project([2.0, 0.5, 0])[0]) < 3


def block(x, y, w=1.0, h=0.3):
    return {"name": "b", "shape": "box", "color": "red", "center": [x, y, 0], "size": [w, h, 0.8], "yaw": 0}


def test_stack_failure():
    assert stack_failure([block(0, 0.15), block(0.3, 0.45)], 0.05) is None
    assert stack_failure([block(0, 0.15), block(0.7, 0.45)], 0.05) == 0
    # each top block alone is fine, but the two together overhang the bottom one
    tower = [block(0, 0.15), block(0.45, 0.45), block(0.85, 0.75)]
    assert stack_failure(tower, 0.01) == 0


def test_fallen_blocks_lie_on_floor_without_overlap():
    rng = np.random.default_rng(0)
    tower = [block(0, 0.15), block(0.45, 0.45), block(0.85, 0.75)]
    out = fallen(tower, 0, rng)
    assert len(out) == 3
    for f in out[1:]:
        assert abs(f["center"][1] - f["size"][1] / 2) < 1e-9
    assert not any(overlaps(out[i], out[j], 0.0) for i in range(3) for j in range(i + 1, 3))


def test_generation_is_deterministic_and_balanced():
    for task in TASKS:
        a, b = generate(task, 3), generate(task, 3)
        assert a["question"] == b["question"] and a["answer"] == b["answer"]
        assert a["answer"] in a["choices"] and a["wrong_answer"] != a["answer"]
    for task in ("viewpoint", "stability"):
        answers = [generate(task, s)["answer"] for s in range(4)]
        assert len(set(answers)) == 2
