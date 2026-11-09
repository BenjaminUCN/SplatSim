"""Punto de agarre: dónde queda el tomate en el marco del efector al final de las demos.

obs["state"] = [x, y, z, roll, pitch, yaw, gripper] es la pose del último link, no la punta
del gripper, así que la distancia efector–tomato no baja de ~13 cm ni en una demo perfecta.
Lo que importa es la posición del tomate en el marco del efector: en el último frame de las
demos es muy consistente (≈ 17 cm sobre el eje de la herramienta). El error de agarre de un
frame es la distancia entre esa posición y el punto de agarre medio.

configs/grasp_point.yaml lo escribe scripts/xarm5/estimate_grasp_point.py.
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


FINGER_LINKS = ("left_finger", "right_finger")
EE_LINK = "link_eef"          # el link de obs["state"]
CENTER_LINK = "gripper_center_link"   # hand_held_XL330: mitad del filo del pruner


def gripper_center(urdf_path: str) -> np.ndarray:
    """Centro del espacio entre las pinzas en el marco del efector, desde la URDF: el punto medio
    de las cajas de los dos dedos con el gripper abierto (en el xArm5 ≈ [0, 0, 0.128], 4 cm antes
    de link_tcp, la punta de los dedos)."""
    import itertools

    import pybullet as p
    from pybullet_utils.bullet_client import BulletClient
    c = BulletClient(connection_mode=p.DIRECT)
    try:
        body = c.loadURDF(urdf_path, useFixedBase=True)
        links = {c.getJointInfo(body, i)[12].decode(): i for i in range(c.getNumJoints(body))}
        ee = c.getLinkState(body, links[EE_LINK], computeForwardKinematics=True)
        rot = np.array(c.getMatrixFromQuaternion(ee[5])).reshape(3, 3)
        if CENTER_LINK in links:     # gripper sin dedos: el URDF marca el punto
            center = c.getLinkState(body, links[CENTER_LINK], computeForwardKinematics=True)
            return rot.T @ (np.array(center[4]) - ee[4])
        centers = []
        for name in FINGER_LINKS:
            lo, hi = c.getAABB(body, links[name])
            corners = np.array([rot.T @ (np.array(v) - ee[4]) for v in itertools.product(*zip(lo, hi))])
            centers.append((corners.min(0) + corners.max(0)) / 2)
        return np.mean(centers, axis=0)
    finally:
        c.disconnect()


POINT_HEADER = """\
# Where the tomato must be in the end-effector frame (m): point, from point_source
# (gripper_center: middle of the space between the fingers, from the URDF; demos: median
# of the demos' last frame, demo_point; manual: set in review_recordings.py's Grasp point
# folder). final_*: the demos' last frame vs point (estimate_grasp_point.py only).
"""


def save_grasp_point(point, source: str, path: str = GRASP_POINT_PATH) -> None:
    """Cambia point (y point_source) conservando demo_point y el resto; quita las estadísticas
    final_*, que eran respecto del punto anterior (estimate_grasp_point.py las recalcula)."""
    import os
    from datetime import datetime
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data = {"point": [round(float(v), 4) + 0.0 for v in point], "point_source": source,
            **{k: v for k, v in data.items() if k not in ("point", "point_source") and not k.startswith("final_")}}
    data["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        f.write(POINT_HEADER)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)


def load_demo_point(path: str = GRASP_POINT_PATH) -> np.ndarray | None:
    """Mediana del final de las demos, si el archivo la tiene."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return np.asarray(data["demo_point"], dtype=float) if data.get("demo_point") else None


def load_point_source(path: str = GRASP_POINT_PATH) -> str:
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("point_source", "demos")


def load_grasp_point(path: str = GRASP_POINT_PATH) -> np.ndarray:
    with open(path) as f:
        return np.asarray(yaml.safe_load(f)["point"], dtype=float)


def grasp_error(state, tomato_pos, point=None) -> np.ndarray:
    """Distancia (m) entre el tomate en el marco del efector y el punto de agarre."""
    point = load_grasp_point() if point is None else point
    return np.linalg.norm(tomato_in_ee(state, tomato_pos) - point, axis=-1)
