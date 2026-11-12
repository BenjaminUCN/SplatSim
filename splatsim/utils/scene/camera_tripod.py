"""Trípode de mesa de la D435i (el que viene en la caja) para dibujar las cámaras base en el panel.

Sin CAD: cilindros y una esfera con medidas aproximadas. La cámara va atornillada (1/4") a la rótula, así
que el trípode queda vertical aunque la cámara esté inclinada: se arma en el mundo, bajo el tornillo, hasta
el plano donde apoya (por defecto z = 0, la mesa del robot), y las patas se alargan para llegar.
"""
import numpy as np

# tornillo de 1/4" de la D435i en el marco óptico de su RGB (x a la derecha, y hacia abajo, z adelante):
# al medio del ancho, en la cara de abajo (realsense2_description, _d435.urdf.xacro)
SCREW_IN_RGB_M = (0.0325, 0.0125, -0.0105)
SUPPORT_Z = 0.0
PLATE_R, PLATE_H = 0.012, 0.004          # plato bajo la cámara
BALL_R = 0.009                           # rótula
NECK_R, NECK_H = 0.011, 0.028            # cuerpo bajo la rótula, donde se articulan las patas
LEG_R = 0.004
LEG_SPLAY_DEG = 28.0                     # patas abiertas respecto de la vertical
SIDES = 16


def _cylinder(a, b, r: float, sides: int = SIDES):
    """Cilindro cerrado entre los puntos a y b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = (b - a) / np.linalg.norm(b - a)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    vertices = np.vstack([a + ring, b + ring, a, b])
    i = np.arange(sides)
    j = (i + 1) % sides
    faces = np.vstack([np.c_[i, j, sides + i], np.c_[j, sides + j, sides + i],
                       np.c_[np.full(sides, 2 * sides), j, i], np.c_[np.full(sides, 2 * sides + 1), sides + i, sides + j]])
    return vertices, faces


def _sphere(center, r: float, n: int = 10):
    lat = np.linspace(0, np.pi, n + 1)
    lon = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    vertices = np.asarray(center) + r * np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], -1).reshape(-1, 3)
    faces = []
    m = 2 * n
    for a in range(n):
        for b in range(m):
            p, q = a * m + b, a * m + (b + 1) % m
            faces += [(p, q, p + m), (q, q + m, p + m)]
    return vertices, np.array(faces)


def _merge(parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(vertices).astype(np.float32), np.vstack(faces).astype(np.uint32)


def screw_position(T_world_camera: np.ndarray) -> np.ndarray:
    """Tornillo de la cámara en el mundo, desde la pose 4x4 de su RGB (la de las cámaras del sim)."""
    return T_world_camera[:3, :3] @ np.array(SCREW_IN_RGB_M) + T_world_camera[:3, 3]


def tripod_mesh(T_world_camera: np.ndarray, support_z: float = SUPPORT_Z):
    """Vértices y caras (en el mundo) del trípode bajo la cámara. Una pata apunta hacia atrás de la
    cámara (lejos de lo que mira), las otras dos hacia adelante. None si la cámara está bajo el apoyo."""
    screw = screw_position(T_world_camera)
    plate_bottom = screw - (0, 0, PLATE_H)
    ball = plate_bottom - (0, 0, BALL_R * 0.8)
    neck_top = ball - (0, 0, BALL_R * 0.8)
    hinge = neck_top - (0, 0, NECK_H)
    height = hinge[2] - support_z
    if height <= 0.01:
        return None
    forward = T_world_camera[:3, 2].copy()
    forward[2] = 0
    yaw = np.arctan2(forward[1], forward[0]) if np.linalg.norm(forward) > 1e-6 else 0.0
    reach = height * np.tan(np.radians(LEG_SPLAY_DEG))
    parts = [_cylinder(plate_bottom, screw, PLATE_R), _sphere(ball, BALL_R), _cylinder(hinge, neck_top, NECK_R)]
    for k in range(3):
        angle = yaw + np.pi + k * 2 * np.pi / 3
        foot = np.array([hinge[0] + reach * np.cos(angle), hinge[1] + reach * np.sin(angle), support_z])
        parts.append(_cylinder(hinge, foot, LEG_R))
    return _merge(parts)
