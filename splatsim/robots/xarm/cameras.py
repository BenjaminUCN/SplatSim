"""Cámaras del splat: un helper común + una clase por cámara."""
import math

import numpy as np
import torch
from gaussian_splatting.scene.cameras import Camera
from torchvision.transforms.functional import to_pil_image

from .config import (
    APPLE_CAM_FOV,
    APPLE_CAM_OFFSET_RPY,
    APPLE_CAM_OFFSET_XYZ,
    BASE_CAM_FOV,
    IMAGE_H,
    IMAGE_W,
    WRIST_CAM_FOV,
)


def build_splat_camera(server, T_cam_world, name: str, colmap_id: int, fov) -> Camera:
    """
    Convierte una pose 4x4 de cámara en el mundo PyBullet a un `Camera` del 3DGS.
    Aquí vive todo el bloque que antes estaba repetido 3 veces.
    """
    inv = server.transformations_cache[server.robot_name]["inv_transformation"]
    inv = torch.as_tensor(inv, dtype=torch.float32)
    T_world = torch.as_tensor(T_cam_world, dtype=torch.float32, device=inv.device)

    T_splat = inv @ T_world                      # mundo PyBullet -> espacio del splat
    R, t = T_splat[:3, :3], T_splat[:3, 3]

    scale = torch.pow(torch.linalg.det(R), 1 / 3)
    R_cw, T_cw = R / scale, t / scale

    T_no_scale = torch.eye(4, device=R.device)
    T_no_scale[:3, :3] = R_cw
    T_no_scale[:3, 3] = T_cw
    T_wc = torch.linalg.inv(T_no_scale)[:3, 3]   # 3DGS espera (R c2w, T w2c)

    blank = to_pil_image(torch.zeros((3, IMAGE_H, IMAGE_W)).float())
    return Camera(
        (IMAGE_W, IMAGE_H),
        colmap_id,
        R_cw.detach().cpu().numpy(),
        T_wc.detach().cpu().numpy(),
        *fov,
        None, blank, None,
        name, 0,
        scale=scale.detach().cpu().numpy(),
    )


def scene_fov(server, default):
    """(FoVx, FoVy) en radianes. Si la escena define camera_focal_px (objects.yaml), todas las cámaras usan
    esa focal con píxeles cuadrados a IMAGE_W x IMAGE_H (p.ej. la RGB de la D435i a 640x480); si no, default."""
    focal = server.object_config[server.robot_name].get("camera_focal_px")
    if focal is None:
        return default
    return 2 * math.atan(IMAGE_W / 2 / focal), 2 * math.atan(IMAGE_H / 2 / focal)


def rpy_matrix(rpy) -> np.ndarray:
    """La misma convención que getQuaternionFromEuler de PyBullet: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _pose_matrix(pos, rot_matrix_flat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array(rot_matrix_flat).reshape(3, 3)
    T[:3, 3] = pos
    return T


class BaseCamera:
    """Cámara fija en el mundo (base_rgb, base_left_rgb, ...). Su pose se cambia desde los sliders.

    jitter: corrimiento (dx, dy, dz en m, en el mundo) y giro (roll, pitch, yaw en rad, en los ejes de la
    cámara) sobre la pose nominal, para variar la cámara por episodio; (0,)*6 = la pose nominal."""

    def __init__(self, server, name: str, xyz, rpy):
        self.s = server
        self.name = name
        self.jitter = (0.0,) * 6
        self.set_pose(xyz, rpy)

    def set_pose(self, xyz, rpy):
        self.xyz, self.rpy = tuple(xyz), tuple(rpy)

    def pose(self) -> np.ndarray:
        """4x4 en el mundo, con el jitter."""
        T = np.eye(4)
        T[:3, :3] = rpy_matrix(self.rpy) @ rpy_matrix(self.jitter[3:])
        T[:3, 3] = np.add(self.xyz, self.jitter[:3])
        return T

    def get(self) -> Camera:
        return build_splat_camera(self.s, self.pose(), self.name, 1, scene_fov(self.s, BASE_CAM_FOV))


class WristCamera:
    """Cámara pegada al link `wrist_camera_link` del URDF."""
    name = "wrist_rgb"
    LINK_NAME = "wrist_camera_link"

    def __init__(self, server):
        self.s = server
        self.link_index = None
        client = server.pybullet_client
        for i in range(client.getNumJoints(server.dummy_robot)):
            info = client.getJointInfo(server.dummy_robot, i)
            if info[12].decode("utf-8") == self.LINK_NAME:
                self.link_index = i
                print(f"[xArm5] {self.LINK_NAME} encontrado: índice={i}")
                break
        if self.link_index is None:
            print(f"[xArm5] WARNING: {self.LINK_NAME} no encontrado en el URDF")

    def get(self):
        if self.link_index is None:
            return None
        client = self.s.pybullet_client
        st = client.getLinkState(self.s.dummy_robot, self.link_index,
                                 computeForwardKinematics=True)
        T = _pose_matrix(st[4], client.getMatrixFromQuaternion(st[5]))
        return build_splat_camera(self.s, T, self.name, 1, scene_fov(self.s, WRIST_CAM_FOV))


class AppleCamera:
    """Cámara que sigue a la manzana con un offset fijo."""
    name = "apple_rgb"

    def __init__(self, server, apple):
        self.s = server
        self.apple = apple

    def get(self) -> Camera:
        client = self.s.pybullet_client
        pos, quat = client.getBasePositionAndOrientation(self.apple.body_id)
        R_apple = np.array(client.getMatrixFromQuaternion(quat)).reshape(3, 3)
        R_off = np.array(client.getMatrixFromQuaternion(
            client.getQuaternionFromEuler(APPLE_CAM_OFFSET_RPY))).reshape(3, 3)

        T = np.eye(4)
        T[:3, :3] = R_apple @ R_off
        T[:3, 3] = np.array(pos) + R_apple @ np.array(APPLE_CAM_OFFSET_XYZ)
        return build_splat_camera(self.s, T, self.name, 2, APPLE_CAM_FOV)
