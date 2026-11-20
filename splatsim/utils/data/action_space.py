"""Espacio del estado y la acción de un dataset LeRobot del xArm5.

- "pose" (lo de siempre): estado y acción son la pose de `link_eef`, [x, y, z, roll, pitch, yaw, pinza],
  con los ángulos de Euler de PyBullet. Al evaluar, el agente pasa la acción por la IK.
- "joints": estado y acción son las 5 articulaciones del brazo más la pinza, sin IK. El estado es
  `joint_positions` de la grabación; la acción, las mismas 5 articulaciones del cuadro más el comando de
  la pinza (`action[6]`), igual que en "pose" la acción es la pose del cuadro con el comando de la pinza.
  Así lo único que cambia entre los dos es la representación.

El espacio queda en los nombres de `observation.state` en meta/info.json: un dataset en articulaciones los
tiene en JOINT_NAMES; los antiguos, `state_0`…`state_6`.
"""
import numpy as np

ACTION_SPACES = ("pose", "joints")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]


def joint_state_action(pkl_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(estado, acción) en articulaciones de un cuadro grabado en simulación, ambos (6,) float32."""
    joints = np.asarray(pkl_data["joint_positions"], dtype=np.float32).flatten()
    if joints.size != 6:
        raise ValueError(f"joint_positions debe tener 6 valores (5 articulaciones + pinza), tiene {joints.size}")
    gripper_command = np.asarray(pkl_data["action"], dtype=np.float32).flatten()[-1]
    return joints, np.append(joints[:5], gripper_command).astype(np.float32)


def feature_names(action_space: str, dim: int, prefix: str) -> list[str]:
    """Nombres de observation.state ("state") o action ("action") en meta/info.json."""
    if action_space == "joints":
        return list(JOINT_NAMES)
    return [f"{prefix}_{i}" for i in range(dim)]


def action_space_of(features: dict) -> str:
    """Espacio de un dataset a partir de sus features (meta/info.json o LeRobotDatasetMetadata.features)."""
    names = list(features["observation.state"].get("names") or [])
    return "joints" if names == JOINT_NAMES else "pose"
