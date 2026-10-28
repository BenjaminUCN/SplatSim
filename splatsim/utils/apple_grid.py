"""Grilla de posiciones del tomate para los ensayos y rango de su posición aleatoria
(configs/apple_grid.yaml).

En el sim, run_policy_trials.py pone el tomate en cada posición; en el robot real el panel indica
la posición y el operador lo pone a mano. random_range acota "Random Position" del panel.
"""
import yaml

APPLE_GRID_PATH = "configs/apple_grid.yaml"
_HEADER = """\
# Tomato positions (x, y, z) in PyBullet world coordinates used by scripts/xarm5/run_policy_trials.py,
# added with "Save Position to Grid" in the control panel (or by hand), and random_range: the box
# (min, max per axis, m) where "Random Position" puts it (panel: Tomato > Random range).
"""


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


def save_grid_file(path: str = APPLE_GRID_PATH, positions: list | None = None,
                   random_range: dict | None = None) -> None:
    """Reescribe el archivo cambiando solo lo que se pasa (lo demás se conserva)."""
    data = _load(path)
    if positions is not None:
        data["positions"] = positions
    if random_range is not None:
        data["random_range"] = {a: [round(float(v), 4) for v in random_range[a]] for a in "xyz"}
    with open(path, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
