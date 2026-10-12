"""Small vectorised numpy ray tracer for CLEVR-like scenes.

World is y-up, the floor is the plane y = 0. Shapes: sphere, box (any
rotation), vertical cylinder. Rendering is exact and deterministic, so the
"imagined" state of a question can be rendered as ground truth.

A scene is a plain dict, so it can be stored next to each question:
    {"objects": [{"name": "red cube", "shape": "box", "color": "red",
                  "center": [x, y, z], "size": [sx, sy, sz], "yaw": deg}, ...]}
Spheres use "radius"; cylinders use "radius" and "height". Boxes may carry a
full "rot" (3x3, row-major) instead of "yaw".
"""
import numpy as np

COLORS = {
    "red": (0.80, 0.16, 0.14),
    "blue": (0.15, 0.32, 0.82),
    "green": (0.16, 0.62, 0.24),
    "yellow": (0.92, 0.78, 0.12),
    "purple": (0.52, 0.22, 0.70),
    "orange": (0.95, 0.48, 0.10),
    "cyan": (0.12, 0.70, 0.78),
    "brown": (0.50, 0.32, 0.17),
    "gray": (0.50, 0.50, 0.52),
    "pink": (0.93, 0.47, 0.66),
}
FLOOR = np.array([0.80, 0.79, 0.76])
SKY_TOP, SKY_BOTTOM = np.array([0.80, 0.85, 0.92]), np.array([0.95, 0.95, 0.96])
LIGHT = np.array([-0.45, 1.0, 0.65]) / np.linalg.norm([-0.45, 1.0, 0.65])
EPS = 1e-4
FLOOR_ID, SKY_ID = -2, -1


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def object_rot(obj):
    if "rot" in obj:
        return np.asarray(obj["rot"], float)
    return rot_y(obj.get("yaw", 0.0))


class Camera:
    """Pinhole camera looking from `eye` at `target`."""

    def __init__(self, eye, target=(0, 0.4, 0), fov=40.0, width=448, height=448):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.fov, self.width, self.height = fov, width, height
        self.fwd = _normalize(self.target - self.eye)
        self.right = _normalize(np.cross(self.fwd, [0.0, 1.0, 0.0]))
        self.up = np.cross(self.right, self.fwd)
        self.tan = np.tan(np.radians(fov) / 2)

    @classmethod
    def orbit(cls, azimuth, elevation=30.0, distance=9.0, target=(0, 0.4, 0), **kw):
        """Azimuth 0 sits on +z looking towards -z; positive azimuth moves the
        camera towards +x, i.e. to the viewer's right."""
        a, e = np.radians(azimuth), np.radians(elevation)
        t = np.asarray(target, float)
        eye = t + distance * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        return cls(eye, t, **kw)

    def to_dict(self):
        return {"eye": self.eye.round(4).tolist(), "target": self.target.round(4).tolist(),
                "fov": self.fov, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d):
        return cls(d["eye"], d["target"], d["fov"], d["width"], d["height"])

    def rays(self, ss=1):
        w, h = self.width * ss, self.height * ss
        aspect = w / h
        xs = (2 * (np.arange(w) + 0.5) / w - 1) * self.tan * aspect
        ys = (1 - 2 * (np.arange(h) + 0.5) / h) * self.tan
        px, py = np.meshgrid(xs, ys)
        d = self.fwd + px[..., None] * self.right + py[..., None] * self.up
        d = _normalize(d.reshape(-1, 3))
        return np.broadcast_to(self.eye, d.shape), d

    def project(self, p):
        """World point -> (pixel x, pixel y, depth). Pixel x grows to the right."""
        v = np.asarray(p, float) - self.eye
        z = v @ self.fwd
        x = (v @ self.right) / (z * self.tan * self.width / self.height)
        y = (v @ self.up) / (z * self.tan)
        return (x + 1) / 2 * self.width, (1 - y) / 2 * self.height, z


# ---------------------------------------------------------------- intersection

def _hit_sphere(o, d, obj):
    c, r = np.asarray(obj["center"], float), obj["radius"]
    oc = o - c
    b = np.einsum("ij,ij->i", oc, d)
    cc = np.einsum("ij,ij->i", oc, oc) - r * r
    disc = b * b - cc
    t = np.full(len(d), np.inf)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0))
    t0 = -b - sq
    t[ok & (t0 > EPS)] = t0[ok & (t0 > EPS)]
    p = o + d * np.where(np.isfinite(t), t, 0)[:, None]
    return t, (p - c) / r


def _hit_box(o, d, obj):
    c, half, R = np.asarray(obj["center"], float), np.asarray(obj["size"], float) / 2, object_rot(obj)
    lo, ld = (o - c) @ R, d @ R          # world -> local (R columns are local axes)
    ld = np.where(np.abs(ld) < 1e-12, 1e-12, ld)
    t1, t2 = (-half - lo) / ld, (half - lo) / ld
    tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
    tmin, tmax = tn.max(1), tf.min(1)
    hit = (tmax >= tmin) & (tmin > EPS)
    t = np.where(hit, tmin, np.inf)
    axis = tn.argmax(1)
    n = np.zeros_like(lo)
    n[np.arange(len(d)), axis] = -np.sign(ld[np.arange(len(d)), axis])
    return t, n @ R.T


def _hit_cylinder(o, d, obj):
    c, r, hh = np.asarray(obj["center"], float), obj["radius"], obj["height"] / 2
    lo = o - c
    a = d[:, 0] ** 2 + d[:, 2] ** 2
    b = lo[:, 0] * d[:, 0] + lo[:, 2] * d[:, 2]
    cc = lo[:, 0] ** 2 + lo[:, 2] ** 2 - r * r
    disc = b * b - a * cc
    ok = (disc >= 0) & (a > 1e-12)
    ts = (-b - np.sqrt(np.where(ok, disc, 0))) / np.where(a > 1e-12, a, 1)
    ys = lo[:, 1] + ts * d[:, 1]
    side = ok & (ts > EPS) & (np.abs(ys) <= hh)
    t = np.where(side, ts, np.inf)
    n = np.zeros_like(lo)
    ps = lo + d * np.where(side, ts, 0)[:, None]
    n[side] = ps[side] * [1, 0, 1] / r
    dy = np.where(np.abs(d[:, 1]) < 1e-12, 1e-12, d[:, 1])
    for sign in (1.0, -1.0):
        tc = (sign * hh - lo[:, 1]) / dy
        pc = lo + d * tc[:, None]
        cap = (tc > EPS) & (pc[:, 0] ** 2 + pc[:, 2] ** 2 <= r * r) & (tc < t)
        t = np.where(cap, tc, t)
        n[cap] = [0, sign, 0]
    return t, n


HIT = {"sphere": _hit_sphere, "box": _hit_box, "cylinder": _hit_cylinder}


def trace(o, d, objects):
    """Nearest hit over objects and floor -> (t, normal, id). id indexes objects."""
    t = np.full(len(d), np.inf)
    n = np.zeros_like(d)
    ids = np.full(len(d), SKY_ID)
    for i, obj in enumerate(objects):
        ti, ni = HIT[obj["shape"]](o, d, obj)
        closer = ti < t
        t[closer], n[closer], ids[closer] = ti[closer], ni[closer], i
    dy = np.where(np.abs(d[:, 1]) < 1e-12, 1e-12, d[:, 1])
    tf = -o[:, 1] / dy
    floor = (tf > EPS) & (tf < t)
    t[floor], n[floor], ids[floor] = tf[floor], [0, 1, 0], FLOOR_ID
    return t, n, ids


def render(scene, cam, ss=2):
    """Render -> (rgb uint8 HxWx3, id map HxW). ss = supersampling factor."""
    objects = scene["objects"]
    o, d = cam.rays(ss)
    t, n, ids = trace(o, d, objects)
    hit = ids != SKY_ID
    p = o + d * np.where(hit, t, 0)[:, None]

    base = np.zeros_like(d)
    for i, obj in enumerate(objects):
        base[ids == i] = COLORS[obj["color"]]
    fl = ids == FLOOR_ID
    base[fl] = FLOOR

    # hard shadows from the key light (objects only cast onto things)
    sh = np.ones(len(d))
    if objects and hit.any():
        so = p[hit] + n[hit] * 1e-3
        st, _, sids = trace(so, np.broadcast_to(LIGHT, so.shape).copy(), objects)
        sh[np.flatnonzero(hit)[sids >= 0]] = 0.0

    lam = np.clip(n @ LIGHT, 0, 1)
    h = _normalize(LIGHT - d)
    spec = np.clip(np.einsum("ij,ij->i", n, h), 0, 1) ** 40 * (ids >= 0)
    col = base * (0.38 + 0.62 * lam * sh)[:, None] + 0.25 * (spec * sh)[:, None]

    # floor fades into the background with distance, sky is a vertical gradient
    fade = np.clip((t - 6.0) / 14.0, 0, 1)[:, None]
    sky = SKY_BOTTOM + (SKY_TOP - SKY_BOTTOM) * np.clip(d[:, 1:2] * 3, 0, 1)
    col[fl] = col[fl] * (1 - fade[fl]) + SKY_BOTTOM * fade[fl]
    col[~hit] = sky[~hit]

    H, W = cam.height, cam.width
    img = col.reshape(H, ss, W, ss, 3).mean((1, 3))
    img = (np.clip(img, 0, 1) ** (1 / 1.1) * 255).round().astype(np.uint8)
    idmap = ids.reshape(H, ss, W, ss)[:, ss // 2, :, ss // 2]
    return img, idmap


def visible_fraction(scene, cam, size=160):
    """Per-object fraction of its pixels that survive occlusion, at low res.
    Compares each object's pixel count in the full scene against rendering it alone."""
    small = Camera(cam.eye, cam.target, cam.fov, size, size)
    _, ids = render(scene, small, ss=1)
    out = []
    for i, obj in enumerate(scene["objects"]):
        _, alone = render({"objects": [obj]}, small, ss=1)
        total = (alone == 0).sum()
        out.append(float((ids == i).sum() / total) if total else 0.0)
    return out
