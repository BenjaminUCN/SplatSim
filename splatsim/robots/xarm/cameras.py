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


def build_splat_camera(server, T_cam_world, name: str, colmap_id: int, fov, principal_px=None) -> Camera:
    """
    Convierte una pose 4x4 de cámara en el mundo PyBullet a un `Camera` del 3DGS.
    Aquí vive todo el bloque que antes estaba repetido 3 veces.
    principal_px: (cx, cy) del centro óptico en píxeles (convención OpenCV, sobre scene_image_size); None = el
    centro de la imagen, como el 3DGS.
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

    width, height = scene_image_size(server)
    blank = to_pil_image(torch.zeros((3, height, width)).float())
    camera = Camera(
        (width, height),
        colmap_id,
        R_cw.detach().cpu().numpy(),
        T_wc.detach().cpu().numpy(),
        *fov,
        None, blank, None,
        name, 0,
        scale=scale.detach().cpu().numpy(),
    )
    if principal_px is not None:
        shift_principal_point(camera, principal_px, width, height)
    return camera


def shift_principal_point(camera: Camera, principal_px, width: int, height: int) -> None:
    """Corre el centro óptico de la proyección a principal_px (frustum descentrado). El rasterizador lleva el NDC a
    píxeles como ((ndc + 1) * W - 1) / 2, así que con P[0, 2] = 0 el eje cae en W/2 - 0.5 (OpenCV); el término
    P[0, 2] = 2 (cx + 0.5 - W/2) / W lo lleva a cx (y lo mismo en y). projection_matrix está traspuesta."""
    cx, cy = principal_px
    P = camera.projection_matrix.clone()
    P[2, 0] = 2.0 * (cx + 0.5 - width / 2) / width
    P[2, 1] = 2.0 * (cy + 0.5 - height / 2) / height
    camera.projection_matrix = P
    camera.full_proj_transform = camera.world_view_transform.unsqueeze(0).bmm(P.unsqueeze(0)).squeeze(0)


def scene_image_size(server) -> tuple[int, int]:
    """(ancho, alto) de las imágenes: camera_image_size [alto, ancho] de la escena (objects.yaml) o
    IMAGE_W x IMAGE_H."""
    size = server.object_config[server.robot_name].get("camera_image_size")
    return (IMAGE_W, IMAGE_H) if size is None else (int(size[1]), int(size[0]))


def scene_fov(server, default):
    """(FoVx, FoVy) en radianes. Si la escena define camera_focal_px (objects.yaml), todas las cámaras usan
    esa focal con píxeles cuadrados sobre scene_image_size (p.ej. la RGB de la D435i en 16:9 a 640x360:
    ~462 px, 69 x 42 grados); si no, default."""
    focal = server.object_config[server.robot_name].get("camera_focal_px")
    if focal is None:
        return default
    width, height = scene_image_size(server)
    return 2 * math.atan(width / 2 / focal), 2 * math.atan(height / 2 / focal)


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
        # centro óptico de la wrist real (calibración de fábrica), si la escena lo da
        principal = self.s.object_config[self.s.robot_name].get("wrist_camera_principal_px")
        return build_splat_camera(self.s, T, self.name, 1, scene_fov(self.s, WRIST_CAM_FOV), principal)


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
