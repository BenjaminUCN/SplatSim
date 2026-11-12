"""Grilla de posiciones del tomate para los ensayos y rango de su posición aleatoria
(configs/apple_grid.yaml).

En el sim, run_policy_trials.py pone el tomate en cada posición; en el robot real el panel indica
la posición y el operador lo pone a mano. "Random Position" del panel pone el tomate dentro de
random_range (una caja) o de random_hemisphere (media cáscara esférica alrededor del hombro, para
que quede al alcance del brazo), según random_shape.
"""
import numpy as np
import yaml

APPLE_GRID_PATH = "configs/apple_grid.yaml"
EVAL_GRID_PATH = "configs/eval_grid_4x4.yaml"   # grilla de evaluación (run_policy_trials.py --grid_path)
_HEADER = """\
# Tomato positions (x, y, z) in PyBullet world coordinates used by scripts/xarm5/run_policy_trials.py,
# added with "Save Position to Grid" in the control panel (or by hand), and random_range: the box
# (min, max per axis, m) or random_hemisphere: where "Random Position" puts it, chosen by random_shape
# (panel: Tomato > Random range). The hemisphere: center (m), radius [min, max] (m), facing (deg, the
# horizontal direction of its axis, 180 = -x), z [min, max] (m, cuts it at those heights),
# x [min, max] (m, cuts it along x, e.g. to keep the tomato away from the robot or the wall) and
# y [min, max] (m, cuts it sideways, e.g. to keep the tomato in front of the plant).
"""
RANGE_SHAPES = ("box", "hemisphere")


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def load_grid(path: str = APPLE_GRID_PATH) -> list:
    """Posiciones guardadas (lista vacía si no hay archivo)."""
    return _load(path).get("positions") or []


def load_random_range(default: dict, path: str = APPLE_GRID_PATH) -> dict:
    """{"x": (min, max), "y": ..., "z": ...} guardado, o `default`."""
    saved = _load(path).get("random_range") or {}
    return {a: tuple(saved.get(a, default[a])) for a in "xyz"}


def load_random_shape(default_hemisphere: dict, path: str = APPLE_GRID_PATH) -> tuple[str, dict]:
    """(forma guardada, "box" si no hay; hemisferio guardado, o `default_hemisphere`)."""
    data = _load(path)
    shape = data.get("random_shape", "box")
    saved = data.get("random_hemisphere") or {}
    hemi = {k: saved.get(k, v) for k, v in default_hemisphere.items()}
    return (shape if shape in RANGE_SHAPES else "box"), normalize_hemisphere(**hemi)


def _round(v):
    return [round(float(x), 4) for x in v] if np.ndim(v) else round(float(v), 4)


def save_grid_file(path: str = APPLE_GRID_PATH, positions: list | None = None,
                   random_range: dict | None = None, random_shape: str | None = None,
                   random_hemisphere: dict | None = None) -> None:
    """Reescribe el archivo cambiando solo lo que se pasa (lo demás se conserva)."""
    data = _load(path)
    if positions is not None:
        data["positions"] = positions
    if random_range is not None:
        data["random_range"] = {a: [round(float(v), 4) for v in random_range[a]] for a in "xyz"}
    if random_shape is not None:
        data["random_shape"] = random_shape
    if random_hemisphere is not None:
        data["random_hemisphere"] = {k: _round(v) for k, v in random_hemisphere.items()}
    with open(path, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


# ── hemisferio ───────────────────────────────────────────────────────────────
MIN_SHELL = 0.01   # m: grosor y alto mínimos del hemisferio
NO_X_LIMIT = (-1.5, 1.5)   # m: más allá del alcance del brazo (sin recorte en x)
NO_Y_LIMIT = (-1.5, 1.5)   # m: ídem en y


def _limits(v) -> tuple[float, float]:
    lo, hi = sorted(float(a) for a in v)
    return lo, max(hi, lo + MIN_SHELL)


def normalize_hemisphere(center, radius, facing, z, x=NO_X_LIMIT, y=NO_Y_LIMIT) -> dict:
    """Valores ordenados y válidos: 0 <= r_min < r_max, min < max en z, x e y, facing en
    (-180, 180]."""
    r_lo, r_hi = sorted(max(float(r), 0.0) for r in radius)
    return {"center": tuple(float(v) for v in center),
            "radius": (r_lo, max(r_hi, r_lo + MIN_SHELL)),
            "facing": 180.0 - (180.0 - float(facing)) % 360.0,
            "z": _limits(z), "x": _limits(x), "y": _limits(y)}


def hemisphere_axes(h: dict):
    """Eje del hemisferio (horizontal, hacia `facing`), el otro horizontal y la vertical."""
    f = np.radians(h["facing"])
    return np.array([np.cos(f), np.sin(f), 0.0]), np.array([-np.sin(f), np.cos(f), 0.0]), np.array([0.0, 0.0, 1.0])


def in_hemisphere(points, hemisphere: dict) -> np.ndarray:
    """Qué puntos (N x 3) están dentro: entre los radios, del lado de `facing` y entre los
    límites en z, x e y."""
    pts = np.atleast_2d(points)
    d = pts - np.array(hemisphere["center"])
    r = np.linalg.norm(d, axis=1)
    axis = hemisphere_axes(hemisphere)[0]
    x_lo, x_hi = hemisphere.get("x", NO_X_LIMIT)
    y_lo, y_hi = hemisphere.get("y", NO_Y_LIMIT)
    return ((r >= hemisphere["radius"][0]) & (r <= hemisphere["radius"][1]) & (d @ axis >= 0)
            & (pts[:, 2] >= hemisphere["z"][0]) & (pts[:, 2] <= hemisphere["z"][1])
            & (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
            & (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi))


def sample_hemisphere(h: dict, rng=np.random, batch: int = 2000, tries: int = 50) -> list[float]:
    """Un punto uniforme en el volumen (muestreo por rechazo en la caja que lo contiene)."""
    c, r = np.array(h["center"]), h["radius"][1]
    x_lo, x_hi = h.get("x", NO_X_LIMIT)
    y_lo, y_hi = h.get("y", NO_Y_LIMIT)
    lo = np.array([max(c[0] - r, x_lo), max(c[1] - r, y_lo), max(c[2] - r, h["z"][0])])
    hi = np.array([min(c[0] + r, x_hi), min(c[1] + r, y_hi), min(c[2] + r, h["z"][1])])
    if (hi <= lo).any():
        raise ValueError("the hemisphere is empty: the x, y or z limits leave it out")
    for _ in range(tries):
        pts = rng.uniform(lo, hi, size=(batch, 3))
        inside = pts[in_hemisphere(pts, h)]
        if len(inside):
            return [float(v) for v in inside[0]]
    raise ValueError("the hemisphere is empty: check the radius and the x, y and z limits")


def hemisphere_lines(h: dict, step_deg: float = 10.0) -> np.ndarray:
    """Segmentos (N x 2 x 3) para dibujarlo: meridianos y paralelos de las dos cáscaras, recortados
    a los límites en z, x e y."""
    axis, side, up = hemisphere_axes(h)
    c = np.array(h["center"])
    segs = []
    theta = np.radians(np.arange(0.0, 90.0 + 1e-9, step_deg / 2))     # desde el eje
    phi = np.radians(np.arange(0.0, 360.0 + 1e-9, step_deg / 2))      # alrededor del eje
    for r in h["radius"]:
        for p in np.radians(np.arange(0.0, 360.0, 30.0)):              # meridianos
            pts = c + r * (np.cos(theta)[:, None] * axis
                           + np.sin(theta)[:, None] * (np.cos(p) * side + np.sin(p) * up))
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        for t in np.radians([30.0, 60.0, 90.0]):                       # paralelos (90: el borde)
            pts = c + r * (np.cos(t) * axis + np.sin(t) * (np.cos(phi)[:, None] * side
                                                           + np.sin(phi)[:, None] * up))
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
    segs = np.concatenate(segs)
    x, y, z = segs[:, :, 0], segs[:, :, 1], segs[:, :, 2]
    x_lo, x_hi = h.get("x", NO_X_LIMIT)
    y_lo, y_hi = h.get("y", NO_Y_LIMIT)
    keep = (z >= h["z"][0]) & (z <= h["z"][1]) & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    return segs[keep.all(axis=1)]
