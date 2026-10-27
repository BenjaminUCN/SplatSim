"""Herramientas de depuración: sliders de cámaras base, botón de reset, ventana de la manzana."""
import cv2
import numpy as np
import yaml

from .config import (
    BASE_CAM_RANGE_RPY,
    BASE_CAM_RANGE_XYZ,
    BASE_CAMS,
    CAMERA_POSES_PATH,
)

_POSES_HEADER = """\
# Poses (xyz in meters, rpy in radians) of the fixed base cameras in PyBullet world coordinates.
# Written by the "Save Camera Poses" button of the xArm5 server; they override BASE_CAMS
# in splatsim/robots/xarm/config.py.
"""


def load_base_camera_poses() -> dict:
    """BASE_CAMS con las poses de CAMERA_POSES_PATH encima: nombre -> (xyz, rpy)."""
    poses = dict(BASE_CAMS)
    try:
        with open(CAMERA_POSES_PATH) as f:
            saved = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return poses
    for name, pose in saved.items():
        poses[name] = (tuple(pose["xyz"]), tuple(pose["rpy"]))
        print(f"[xArm5] {name}: pose de {CAMERA_POSES_PATH} xyz={pose['xyz']} rpy={pose['rpy']}")
    return poses


class BaseCameraTuner:
    """Sliders X/Y/Z/Roll/Pitch/Yaw que mueven una cámara base y dibujan sus ejes."""

    def __init__(self, server, camera):
        self.s, self.camera = server, camera
        c = server.pybullet_client
        prefix = camera.name.removesuffix("_rgb")
        init = (*camera.xyz, *camera.rpy)
        names = [f"{prefix} {n}" for n in ("X", "Y", "Z", "Roll", "Pitch", "Yaw")]
        ranges = [BASE_CAM_RANGE_XYZ] * 3 + [BASE_CAM_RANGE_RPY] * 3
        self._sliders = [c.addUserDebugParameter(n, v - r, v + r, v)
                         for n, v, r in zip(names, init, ranges)]
        self._line_ids = [-1, -1, -1]
        self._prev_pose = None
        print(f"Sliders ready. Adjust '{prefix}' XYZ / RPY to tune {camera.name}.")

    def update(self) -> None:
        c = self.s.pybullet_client
        x, y, z, roll, pitch, yaw = [c.readUserDebugParameter(s) for s in self._sliders]
        self.camera.set_pose((x, y, z), (roll, pitch, yaw))
        self._draw_pose((x, y, z), c.getQuaternionFromEuler([roll, pitch, yaw]))

    def _draw_pose(self, pos, quat, length=0.1) -> None:
        pose = (tuple(pos), tuple(quat))
        if pose == self._prev_pose:
            return
        self._prev_pose = pose

        c = self.s.pybullet_client
        R = np.array(c.getMatrixFromQuaternion(quat)).reshape(3, 3)
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            end = np.array(pos) + length * R[:, i]
            self._line_ids[i] = c.addUserDebugLine(
                pos, end.tolist(), colors[i], 2, lifeTime=0,
                replaceItemUniqueId=self._line_ids[i],
            )


class SaveCameraPosesButton:
    """Botón que guarda la pose actual de las cámaras base en CAMERA_POSES_PATH."""

    def __init__(self, server, cameras):
        self.s, self.cameras = server, cameras
        self._btn = server.pybullet_client.addUserDebugParameter("Save Camera Poses", 1, 0, 0)
        self._prev = 0

    def update(self) -> None:
        val = self.s.pybullet_client.readUserDebugParameter(self._btn)
        if val == self._prev:
            return
        self._prev = val
        poses = {cam.name: {"xyz": [round(float(v), 4) for v in cam.xyz],
                            "rpy": [round(float(v), 4) for v in cam.rpy]}
                 for cam in self.cameras}
        with open(CAMERA_POSES_PATH, "w") as f:
            f.write(_POSES_HEADER)
            yaml.safe_dump(poses, f, default_flow_style=None, sort_keys=False)
        print(f"[xArm5] Poses de {', '.join(poses)} guardadas en {CAMERA_POSES_PATH}")


class ResetJointsButton:
    """Botón de PyBullet que devuelve el brazo a la pose inicial."""

    def __init__(self, server):
        self.s = server
        self._btn = server.pybullet_client.addUserDebugParameter("Reset Robot Joints", 1, 0, 0)
        self._prev = 0

    def update(self) -> None:
        s, c = self.s, self.s.pybullet_client
        val = c.readUserDebugParameter(self._btn)
        if val == self._prev:
            return
        self._prev = val
        print("[xArm5] Resetting robot joints to initial positions.")
        for i in range(1, s.num_dofs()):
            c.setJointMotorControl2(
                s.dummy_robot, i, c.POSITION_CONTROL,
                targetPosition=s.initial_joint_state[i - 1] * s.joint_signs[i - 1],
                force=1000,
            )


class AppleWindow:
    """Ventana de OpenCV con la última imagen de la cámara de la manzana."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last = None

    def set(self, img) -> None:
        self._last = img

    def show(self) -> None:
        if not self.enabled or self._last is None:
            return
        img = (self._last.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imshow("Apple Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
