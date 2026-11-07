"""Segmentar una hoja de un splat: elegir un punto con un rayo y crecer por la superficie.

El follaje de un splat es translúcido gaussiana a gaussiana (opacidad mediana ~0.1) y solo tapa
al sumarlas, así que filtrar por opacidad deja trozos sueltos. Por eso la región crece por todas
las gaussianas conectadas, mientras la superficie siga suave y el color se parezca: dos hojas
casi nunca son continuas y del mismo color a la vez.

Solo depende de numpy/scipy/plyfile/opencv: lo usan la interfaz (scripts/xarm5/segment_leaf.py)
y las pruebas, sin GPU. Todo está en coordenadas de mundo (metros), salvo el .ply guardado, que
conserva las gaussianas en el marco del splat para no tocar sus rotaciones ni armónicos; el .yaml
de al lado guarda la transformación a mundo (splatsim/robots/xarm/occluders.py lo usa).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # armónico esférico de grado 0: f_dc -> color
MIN_OPACITY = 0.02           # por debajo no se ven ni al sumarlas: no conectan nada
OCCLUDER_ROOT = "~/data/occluders"


@dataclass
class GrowParams:
    link_mm: float = 5.0
    """dos gaussianas conectan si están a menos de esto"""
    normal_deg: float = 30.0
    """ángulo máximo entre las normales locales de dos gaussianas conectadas"""
    plane_mm: float = 2.0
    """distancia máxima al plano local de la vecina (separa hojas paralelas)"""
    color_tol: float = 20.0
    """distancia de color máxima (a, b de Lab) al color de la semilla"""
    max_radius_cm: float = 18.0
    """radio máximo desde la semilla"""


class SplatCloud:
    """Las gaussianas de un .ply en mundo, con lo necesario para elegir y crecer."""

    def __init__(self, vertex: np.ndarray, transformation):
        self.vertex = vertex  # arreglo estructurado del .ply (todas las columnas)
        self.T = np.asarray(transformation, dtype=float)
        self.world_scale = float(np.cbrt(np.linalg.det(self.T[:3, :3])))
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(float)
        self.xyz = xyz @ self.T[:3, :3].T + self.T[:3, 3]
        self.opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(float)))
        self.rgb = np.clip(0.5 + SH_C0 * np.stack(
            [vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(float), 0, 1)
        self.sigma = np.exp(np.stack(
            [vertex[f"scale_{i}"] for i in range(3)], axis=1).astype(float)) * self.world_scale
        self.lab = cv2.cvtColor(self.rgb.astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0].astype(float)

    @classmethod
    def from_ply(cls, path, transformation) -> "SplatCloud":
        return cls(PlyData.read(str(Path(path).expanduser()))["vertex"].data, transformation)

    def __len__(self) -> int:
        return len(self.xyz)

    def covariances(self, indices=None) -> np.ndarray:
        """Covarianzas (N, 3, 3) en mundo, para dibujar el splat en viser."""
        v = self.vertex if indices is None else self.vertex[indices]
        sigma = self.sigma if indices is None else self.sigma[indices]
        q = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(float)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                     axis=1).reshape(-1, 3, 3)
        R = (self.T[:3, :3] / self.world_scale) @ R
        return (R * (sigma ** 2)[:, None, :]) @ R.transpose(0, 2, 1)


def pick_surface(cloud: SplatCloud, origin, direction, min_alpha: float = 0.5) -> int | None:
    """Índice de la gaussiana que se ve por el rayo: la primera en que el alfa acumulado llega a
    min_alpha, como la mezcla del render (no la más cercana al rayo, que suele ser un velo)."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    rel = cloud.xyz - origin
    depth = rel @ direction
    perp2 = np.sum((rel - depth[:, None] * direction) ** 2, axis=1)
    radius = np.maximum(cloud.sigma.max(axis=1), 1e-4)
    near = np.where((depth > 0) & (perp2 < (3 * radius) ** 2))[0]
    if len(near) == 0:
        return None
    near = near[np.argsort(depth[near])]
    # muchas gaussianas son agujas: el alfa sale de su covarianza proyectada sobre el plano
    # perpendicular al rayo, no de su eje mayor (eso las haría tapar donde no llegan)
    u = np.cross(direction, [0.0, 0.0, 1.0] if abs(direction[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    basis = np.stack([u, v])  # (2, 3)
    cov2 = basis @ cloud.covariances(near) @ basis.T + 1e-12 * np.eye(2)
    r2 = rel[near] @ basis.T
    mahalanobis = np.einsum("ni,ni->n", r2, np.linalg.solve(cov2, r2[..., None])[..., 0])
    alpha = cloud.opacity[near] * np.exp(-0.5 * mahalanobis)
    accumulated = 1.0 - np.cumprod(1.0 - alpha)
    hit = int(np.searchsorted(accumulated, min_alpha))
    return int(near[min(hit, len(near) - 1)])


def local_normals(xyz: np.ndarray, weights: np.ndarray, k: int = 24) -> np.ndarray:
    """Normal por gaussiana: el eje de menor varianza de sus k vecinas, ponderado por opacidad."""
    if len(xyz) < 4:
        return np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    _, nn = cKDTree(xyz).query(xyz, k=min(k, len(xyz)))
    w = weights[nn][..., None]
    pts = xyz[nn]
    centered = pts - (pts * w).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered * w, centered)
    return np.linalg.eigh(cov)[1][:, :, 0]


def grow_region(cloud: SplatCloud, seed: int, params: GrowParams | None = None) -> np.ndarray:
    """Índices de la superficie conectada a seed."""
    p = params or GrowParams()
    radius = p.max_radius_cm / 100
    ball = np.where(np.sum((cloud.xyz - cloud.xyz[seed]) ** 2, axis=1) < radius ** 2)[0]
    ball = ball[(cloud.opacity[ball] >= MIN_OPACITY) | (ball == seed)]
    xyz = cloud.xyz[ball]
    lab = cloud.lab[ball]
    normals = local_normals(xyz, cloud.opacity[ball])
    tree = cKDTree(xyz)
    # el color de referencia es el de la zona, no el de una sola gaussiana (nervios, brillos)
    start = int(np.where(ball == seed)[0][0])
    near = tree.query_ball_point(xyz[start], 0.012)
    weights = cloud.opacity[ball[near]]
    target = np.average(lab[near, 1:], axis=0, weights=weights)
    color_ok = np.linalg.norm(lab[:, 1:] - target, axis=1) < p.color_tol
    cos_min = np.cos(np.radians(p.normal_deg))
    link, plane = p.link_mm / 1000, p.plane_mm / 1000
    # las normales locales suavizan los quiebres: dos hojas que se tocan en ángulo cambian de normal
    # de a poco y el paso a paso no las separa. La curvatura total desde la semilla sí
    # (la normal de la zona, no la de la semilla sola: esa es ruidosa, sobre todo en un brillo)
    zone = xyz[near] - np.average(xyz[near], axis=0, weights=weights)
    seed_normal = np.linalg.eigh((zone * weights[:, None]).T @ zone)[1][:, 0]
    total_bend = np.cos(np.radians(min(2 * p.normal_deg, 89.0)))
    color_ok &= np.abs(normals @ seed_normal) > total_bend
    color_ok[start] = True

    visited = np.zeros(len(ball), dtype=bool)
    visited[start] = True
    frontier = np.array([start])
    while len(frontier):
        found = []
        for i, neighbors in zip(frontier, tree.query_ball_point(xyz[frontier], link)):
            n = np.asarray(neighbors)
            n = n[~visited[n] & color_ok[n]]
            if len(n) == 0:
                continue
            aligned = np.abs(normals[n] @ normals[i]) > cos_min
            coplanar = np.abs((xyz[n] - xyz[i]) @ normals[i]) < plane
            n = n[aligned & coplanar]
            visited[n] = True
            found.append(n)
        frontier = np.concatenate(found) if found else np.array([], dtype=int)
    return ball[visited]


def brush(cloud: SplatCloud, center: int, radius_cm: float) -> np.ndarray:
    """Índices dentro de una esfera (retoque a mano: agregar o borrar lo que el crecimiento erró)."""
    dist2 = np.sum((cloud.xyz - cloud.xyz[center]) ** 2, axis=1)
    return np.where((dist2 < (radius_cm / 100) ** 2) & (cloud.opacity >= MIN_OPACITY))[0]


def leaf_summary(cloud: SplatCloud, indices: np.ndarray) -> dict:
    """Centro (ponderado por opacidad), normal y tamaño de la selección."""
    pts = cloud.xyz[indices]
    w = cloud.opacity[indices]
    center = np.average(pts, axis=0, weights=w)
    rel = pts - center
    axes = np.linalg.eigh((rel * w[:, None]).T @ rel)[1]
    extent = np.ptp(rel @ axes, axis=0)
    return {"center": center, "normal": axes[:, 0], "size_cm": np.sort(extent)[::-1] * 100,
            "count": int(len(indices)), "median_opacity": float(np.median(w))}


class LeafSelection:
    """La selección como una lista de pasos, para deshacer y para recalcular el último crecimiento
    al mover los parámetros sin volver a hacer clic."""

    def __init__(self, cloud: SplatCloud):
        self.cloud = cloud
        self.steps: list[dict] = []

    def grow(self, seed: int, params: GrowParams) -> None:
        self.steps.append({"kind": "add", "seed": seed, "indices": grow_region(self.cloud, seed, params)})

    def regrow(self, params: GrowParams) -> bool:
        """Rehace el último paso si fue un crecimiento."""
        if not self.steps or "seed" not in self.steps[-1]:
            return False
        self.steps[-1]["indices"] = grow_region(self.cloud, self.steps[-1]["seed"], params)
        return True

    def paint(self, center: int, radius_cm: float, erase: bool = False) -> None:
        self.steps.append({"kind": "erase" if erase else "add",
                           "indices": brush(self.cloud, center, radius_cm)})

    def undo(self) -> None:
        if self.steps:
            self.steps.pop()

    def clear(self) -> None:
        self.steps.clear()

    def indices(self) -> np.ndarray:
        mask = np.zeros(len(self.cloud), dtype=bool)
        for step in self.steps:
            mask[step["indices"]] = step["kind"] == "add"
        return np.where(mask)[0]


def crop_cloud(cloud: SplatCloud, box_min, box_max) -> SplatCloud:
    """Las gaussianas visibles dentro de una caja de mundo (lo que se muestra y se puede elegir)."""
    inside = np.all((cloud.xyz > np.asarray(box_min)) & (cloud.xyz < np.asarray(box_max)), axis=1)
    return SplatCloud(cloud.vertex[inside & (cloud.opacity >= MIN_OPACITY)], cloud.T)


def occluder_path(scene: str, name: str, root: str = OCCLUDER_ROOT) -> Path:
    return Path(root).expanduser() / scene / f"{name}.ply"


def save_leaf(cloud: SplatCloud, indices: np.ndarray, path, scene: str) -> Path:
    """Escribe <name>.ply (un splat normal, en su marco) y <name>.yaml (cómo pasarlo a mundo)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.sort(np.asarray(indices))
    PlyData([PlyElement.describe(cloud.vertex[indices], "vertex")]).write(str(path))
    info = leaf_summary(cloud, indices)
    with open(path.with_suffix(".yaml"), "w") as f:
        yaml.safe_dump({
            "scene": scene,
            "count": info["count"],
            "center": [round(float(v), 5) for v in info["center"]],
            "normal": [round(float(v), 4) for v in info["normal"]],
            "size_cm": [round(float(v), 1) for v in info["size_cm"]],
            "transformation": [[float(v) for v in row] for row in cloud.T],
        }, f, default_flow_style=None, sort_keys=False)
    return path


def next_leaf_name(scene: str, root: str = OCCLUDER_ROOT) -> str:
    """hoja_1, hoja_2, ...: el primer nombre libre."""
    n = 1
    while occluder_path(scene, f"hoja_{n}", root).exists():
        n += 1
    return f"hoja_{n}"


def saved_leaves(scene: str, root: str = OCCLUDER_ROOT) -> list[Path]:
    return sorted((Path(root).expanduser() / scene).glob("*.ply"))


def load_leaf_info(ply) -> dict:
    with open(Path(ply).expanduser().with_suffix(".yaml")) as f:
        return yaml.safe_load(f)


def canonical_transformation(transformation, center) -> np.ndarray:
    """Del marco del .ply a un marco con la misma orientación que el mundo y el origen en el centro
    de la hoja: así la pose de un oclusor es dónde queda su centro, y rota sobre sí misma."""
    T = np.array(transformation, dtype=float)
    T[:3, 3] -= np.asarray(center, dtype=float)
    return T


def place_between(camera_xyz, target_xyz, fraction: float) -> np.ndarray:
    """Punto de la recta objetivo-cámara: 0 = en el objetivo, 1 = en la cámara."""
    camera_xyz, target_xyz = np.asarray(camera_xyz, float), np.asarray(target_xyz, float)
    return target_xyz + fraction * (camera_xyz - target_xyz)


def load_occluder_config(path) -> list[dict]:
    """Entradas de configs/occluders.yaml: name, ply, position, wxyz, enabled. Sin archivo: ninguna."""
    path = Path(path).expanduser()
    if not path.exists():
        return []
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("occluders") or []


_OCCLUDERS_HEADER = """\
# Leaves cut from a splat (scripts/xarm5/segment_leaf.py) placed in the scene: position of the
# leaf's center in PyBullet world coordinates (m) and rotation (wxyz, identity = as it was in the
# plant). They only exist in the images: no collisions. Used with --occluders; written by
# "Save Occluders" in the control panel.
"""


def save_occluder_config(path, entries: list[dict]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(_OCCLUDERS_HEADER)
        yaml.safe_dump({"occluders": entries}, f, default_flow_style=None, sort_keys=False)
