"""Geometría de una cámara base fija: proyección a píxeles, frustum y máscara de un cuerpo de PyBullet.

La cámara es la misma que renderiza el splat (splatsim/robots/xarm/cameras.py): pose xyz + rpy en el
mundo de PyBullet (convención de COLMAP: x a la derecha de la imagen, y hacia abajo, z hacia adelante)
y los FOV horizontal y vertical de config.py sobre una imagen IMAGE_W x IMAGE_H (los píxeles no son
cuadrados: el FOV es 16:9 y la imagen 4:3, igual que en el render del splat).

Solo depende de numpy; body_mask importa PyBullet al usarse.
"""
from dataclasses import dataclass

import numpy as np


def rotation_from_rpy(rpy) -> np.ndarray:
    """Matriz de rotación de PyBullet (getQuaternionFromEuler): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class CameraView:
    xyz: tuple
    rpy: tuple
    fov: tuple            # (horizontal, vertical) en rad
    width: int
    height: int
    near: float = 0.02    # m: lo que está más cerca no se ve

    @property
    def rotation(self) -> np.ndarray:
        """Columnas: derecha, abajo y adelante de la cámara, en el mundo."""
        return rotation_from_rpy(self.rpy)

    @property
    def focal(self) -> tuple[float, float]:
        return (self.width / (2 * np.tan(self.fov[0] / 2)), self.height / (2 * np.tan(self.fov[1] / 2)))

    def to_camera(self, points) -> np.ndarray:
        """Puntos (N, 3) del mundo en el marco de la cámara."""
        return (np.atleast_2d(np.asarray(points, float)) - np.asarray(self.xyz, float)) @ self.rotation

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) en píxeles (N, 2) y profundidad (N,). u crece hacia la derecha y v hacia abajo."""
        cam = self.to_camera(points)
        depth = cam[:, 2]
        fx, fy = self.focal
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.column_stack([self.width / 2 + fx * cam[:, 0] / depth,
                                  self.height / 2 + fy * cam[:, 1] / depth])
        return uv, depth

    def uv_fraction(self, point) -> list[float] | None:
        """Proyección de un punto en fracción de la imagen (puede salir de 0-1), None si está detrás."""
        uv, depth = self.project([point])
        if depth[0] <= self.near:
            return None
        return [round(float(uv[0, 0]) / self.width, 4), round(float(uv[0, 1]) / self.height, 4)]

    def in_frame(self, points, margin: float = 0.0) -> np.ndarray:
        """Si cada punto cae dentro del frustum, a más de margin píxeles del borde."""
        uv, depth = self.project(points)
        return ((depth > self.near) & (uv[:, 0] >= margin) & (uv[:, 0] <= self.width - margin)
                & (uv[:, 1] >= margin) & (uv[:, 1] <= self.height - margin))

    # ── render de máscaras con PyBullet ──────────────────────────────────────
    def pybullet_matrices(self, far: float = 10.0) -> tuple[list, list]:
        """View y projection de PyBullet equivalentes (OpenGL mira hacia -z con y hacia arriba)."""
        import pybullet as p
        rot = self.rotation
        eye = np.asarray(self.xyz, float)
        view = p.computeViewMatrix(eye.tolist(), (eye + rot[:, 2]).tolist(), (-rot[:, 1]).tolist())
        aspect = np.tan(self.fov[0] / 2) / np.tan(self.fov[1] / 2)   # da el FOV horizontal exacto
        proj = p.computeProjectionMatrixFOV(np.degrees(self.fov[1]), aspect, self.near, far)
        return view, proj

    def segmentation(self, client, scale: float = 1.0) -> np.ndarray:
        """Id del cuerpo de PyBullet que se ve en cada píxel (-1 = nada), a scale de la resolución.
        Solo la geometría de PyBullet: la planta existe solo en el splat y no tapa nada aquí."""
        import pybullet as p
        w, h = max(1, round(self.width * scale)), max(1, round(self.height * scale))
        view, proj = self.pybullet_matrices()
        _, _, _, _, seg = client.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
        return np.reshape(np.asarray(seg), (h, w))


def mask_stats(mask: np.ndarray) -> dict:
    """Píxeles, centroide y caja de una máscara booleana, en fracción de la imagen (0-1)."""
    h, w = mask.shape
    count = int(mask.sum())
    if not count:
        return {"pixels": 0, "fraction": 0.0, "centroid": None, "bbox": None}
    rows, cols = np.nonzero(mask)
    return {"pixels": count, "fraction": round(count / mask.size, 5),
            "centroid": [round(float(cols.mean()) / w, 4), round(float(rows.mean()) / h, 4)],
            "bbox": [round(cols.min() / w, 4), round(rows.min() / h, 4),
                     round((cols.max() + 1) / w, 4), round((rows.max() + 1) / h, 4)]}
