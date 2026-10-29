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
_HEADER = """\
# Tomato positions (x, y, z) in PyBullet world coordinates used by scripts/xarm5/run_policy_trials.py,
# added with "Save Position to Grid" in the control panel (or by hand), and random_range: the box
# (min, max per axis, m) or random_hemisphere: where "Random Position" puts it, chosen by random_shape
# (panel: Tomato > Random range). The hemisphere: center (m), radius [min, max] (m), facing (deg, the
# horizontal direction of its axis, 180 = -x) and z [min, max] (m, cuts it at those heights).
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


def normalize_hemisphere(center, radius, facing, z) -> dict:
    """Valores ordenados y válidos: 0 <= r_min < r_max, z_min < z_max, facing en (-180, 180]."""
    r_lo, r_hi = sorted(max(float(r), 0.0) for r in radius)
    z_lo, z_hi = sorted(float(v) for v in z)
    return {"center": tuple(float(v) for v in center),
            "radius": (r_lo, max(r_hi, r_lo + MIN_SHELL)),
            "facing": 180.0 - (180.0 - float(facing)) % 360.0,
            "z": (z_lo, max(z_hi, z_lo + MIN_SHELL))}


def hemisphere_axes(h: dict):
    """Eje del hemisferio (horizontal, hacia `facing`), el otro horizontal y la vertical."""
    f = np.radians(h["facing"])
    return np.array([np.cos(f), np.sin(f), 0.0]), np.array([-np.sin(f), np.cos(f), 0.0]), np.array([0.0, 0.0, 1.0])


def in_hemisphere(points, h: dict) -> np.ndarray:
    """Qué puntos (N x 3) están dentro: entre los radios, del lado de `facing` y entre las alturas z."""
    d = np.atleast_2d(points) - np.array(h["center"])
    r = np.linalg.norm(d, axis=1)
    axis = hemisphere_axes(h)[0]
    z = np.atleast_2d(points)[:, 2]
    return ((r >= h["radius"][0]) & (r <= h["radius"][1]) & (d @ axis >= 0)
            & (z >= h["z"][0]) & (z <= h["z"][1]))


def sample_hemisphere(h: dict, rng=np.random, batch: int = 2000, tries: int = 50) -> list[float]:
    """Un punto uniforme en el volumen (muestreo por rechazo en la caja que lo contiene)."""
    c, r = np.array(h["center"]), h["radius"][1]
    lo = np.array([c[0] - r, c[1] - r, max(c[2] - r, h["z"][0])])
    hi = np.array([c[0] + r, c[1] + r, min(c[2] + r, h["z"][1])])
    if (hi <= lo).any():
        raise ValueError("the hemisphere is empty: the z limits leave it out")
    for _ in range(tries):
        pts = rng.uniform(lo, hi, size=(batch, 3))
        inside = pts[in_hemisphere(pts, h)]
        if len(inside):
            return [float(v) for v in inside[0]]
    raise ValueError("the hemisphere is empty: check the radius and z limits")


def hemisphere_lines(h: dict, step_deg: float = 10.0) -> np.ndarray:
    """Segmentos (N x 2 x 3) para dibujarlo: meridianos y paralelos de las dos cáscaras, recortados
    a las alturas z."""
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
    z = segs[:, :, 2]
    return segs[((z >= h["z"][0]) & (z <= h["z"][1])).all(axis=1)]
