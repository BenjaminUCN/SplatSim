"""Punto de agarre: dónde queda el tomate en el marco del efector al final de las demos.

obs["state"] = [x, y, z, roll, pitch, yaw, gripper] es la pose del último link, no la punta
del gripper, así que la distancia efector–tomato no baja de ~13 cm ni en una demo perfecta.
Lo que importa es la posición del tomate en el marco del efector: en el último frame de las
demos es muy consistente (≈ 17 cm sobre el eje de la herramienta). El error de agarre de un
frame es la distancia entre esa posición y el punto de agarre medio.

configs/grasp_point.yaml lo escribe scripts/estimate_grasp_point.py.
"""
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

GRASP_POINT_PATH = "configs/grasp_point.yaml"
SUCCESS_RADIUS = 0.04   # m: error de agarre bajo el que un ensayo cuenta como éxito


def tomato_in_ee(state, tomato_pos) -> np.ndarray:
    """Posición del tomate en el marco del efector; state y tomato_pos pueden ser (N, ...)."""
    state = np.asarray(state, dtype=float)
    single = state.ndim == 1
    state = np.atleast_2d(state)
    tomato = np.atleast_2d(np.asarray(tomato_pos, dtype=float))
    # mismos ángulos que getEulerFromQuaternion de PyBullet (xyz extrínseco)
    rel = Rotation.from_euler("xyz", state[:, 3:6]).inv().apply(tomato - state[:, :3])
    return rel[0] if single else rel


def load_grasp_point(path: str = GRASP_POINT_PATH) -> np.ndarray:
    with open(path) as f:
        return np.asarray(yaml.safe_load(f)["point"], dtype=float)


def grasp_error(state, tomato_pos, point=None) -> np.ndarray:
    """Distancia (m) entre el tomate en el marco del efector y el punto de agarre."""
    point = load_grasp_point() if point is None else point
    return np.linalg.norm(tomato_in_ee(state, tomato_pos) - point, axis=-1)
