"""Etiquetas de parada por frame para el clasificador: ¿el efector ya está en posición de agarre?

Cada episodio tiene un frame valid_from: desde ahí hasta el final el brazo está en posición
(etiqueta 1). Los frames anteriores son negativos (0), salvo una banda de IGNORE_FRAMES justo
antes de valid_from, ambigua, que no se usa (-1).

- Automático (grabaciones de sim): valid_from = inicio del último tramo con error de agarre
  <= VALID_RADIUS que llega al final del episodio (splatsim/utils/grasp_point.py). Si el último
  frame no está en posición, el episodio no tiene positivos y solo son negativos los frames con
  error >= INVALID_RADIUS. Antes de valid_from, los frames que ya estaban cerca del punto de
  agarre (el brazo pasó por ahí) se ignoran en vez de usarse como negativos.
- Manual (<carpeta>/stop_labels.yaml, lo escribe scripts/xarm5/review_recordings.py): valid_from
  elegido a mano, o exclude para no usar el episodio. Las grabaciones reales no tienen la
  posición del tomate, así que solo se etiquetan a mano.
"""
from pathlib import Path

import numpy as np
import yaml

STOP_LABELS_FILE = "stop_labels.yaml"
VALID_RADIUS = 0.03     # m: error de agarre bajo el que el frame cuenta como "en posición"
INVALID_RADIUS = 0.06   # m: sobre esto, seguro que no está en posición
IGNORE_FRAMES = 3       # frames ambiguos justo antes de valid_from

_HEADER = """\
# Stop labels for the visual classifier (written by scripts/xarm5/review_recordings.py).
# valid_from: first frame where the arm is in grasp position (until the end of the episode);
# exclude: true to leave the episode out. Episodes not listed use the automatic labels.
"""


def auto_valid_from(grasp_err: np.ndarray) -> int | None:
    """Inicio del último tramo en posición que llega al final, o None si termina fuera."""
    ok = np.asarray(grasp_err) <= VALID_RADIUS
    if not len(ok) or not ok[-1]:
        return None
    k = len(ok) - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    return int(k)


def frame_labels(n: int, valid_from: int | None, grasp_err: np.ndarray | None = None) -> np.ndarray:
    """1 = en posición, 0 = no, -1 = no usar."""
    labels = np.full(n, -1, dtype=int)
    if valid_from is not None:
        labels[valid_from:] = 1
        labels[:max(0, valid_from - IGNORE_FRAMES)] = 0
    elif grasp_err is None:
        labels[:] = 0                      # manual sin positivos: todo negativo
    if grasp_err is not None:
        err = np.asarray(grasp_err)
        before = np.arange(n) < (valid_from if valid_from is not None else n)
        labels[before & (err < INVALID_RADIUS)] = -1   # cerca pero antes de llegar: ambiguo
        if valid_from is None:
            labels[err >= INVALID_RADIUS] = 0
    return labels


def load_stop_labels(folder) -> dict[str, dict]:
    path = Path(folder) / STOP_LABELS_FILE
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {str(k): dict(v) for k, v in (data.get("episodes") or {}).items()}


def save_stop_labels(folder, labels: dict[str, dict]) -> None:
    names = sorted(labels, key=lambda n: (len(n), n))
    with open(Path(folder) / STOP_LABELS_FILE, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump({"episodes": {n: labels[n] for n in names}}, f, default_flow_style=None,
                       sort_keys=False)


def episode_labels(n: int, manual: dict | None, grasp_err: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Etiquetas de un episodio y su origen: manual / auto / excluded / none."""
    if manual and manual.get("exclude"):
        return np.full(n, -1, dtype=int), "excluded"
    if manual and "valid_from" in manual:
        return frame_labels(n, manual["valid_from"], grasp_err), "manual"
    if grasp_err is not None:
        return frame_labels(n, auto_valid_from(grasp_err), grasp_err), "auto"
    return np.full(n, -1, dtype=int), "none"
