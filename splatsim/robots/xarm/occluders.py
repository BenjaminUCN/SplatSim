"""Hojas recortadas del splat (scripts/xarm5/segment_leaf.py) agregadas a la escena como oclusores.

Solo existen en las imágenes: no tienen cuerpo en PyBullet, así que no chocan con el brazo. La
pose se aplica con transform_object, la misma función de los objetos del splat, que también rota
los armónicos esféricos para que el brillo de la hoja gire con ella.
"""
from pathlib import Path

import numpy as np
import torch
from gaussian_splatting.scene.gaussian_model import GaussianModel

from splatsim.utils.scene.leaf_segmentation import (
    canonical_transformation,
    load_leaf_info,
    load_occluder_config,
    save_occluder_config,
)
from splatsim.utils.robot_splat_render_utils import transform_object

# transform_object recorta el objeto a una caja; una hoja se usa entera
NO_AABB = {"bounding_box": [[-1e6] * 3, [1e6] * 3]}
GAUSSIAN_FIELDS = ("_xyz", "_rotation", "_opacity", "_scaling", "_features_dc", "_features_rest")


class Occluder:
    def __init__(self, name: str, ply: str, center, position, wxyz, enabled: bool):
        self.name, self.ply, self.enabled = name, ply, enabled
        self.center = tuple(float(v) for v in center)
        self.position = tuple(float(v) for v in position)
        self.wxyz = tuple(float(v) for v in wxyz)
        self.gaussian = GaussianModel(3)
        self.gaussian.load_ply(str(Path(ply).expanduser()))
        self.placed = None        # tensores en la pose placed_pose (transformar cuesta; casi no cambia)
        self.placed_pose = None

    def entry(self) -> dict:
        return {"name": self.name, "ply": self.ply, "position": [round(v, 4) for v in self.position],
                "wxyz": [round(v, 5) for v in self.wxyz], "enabled": self.enabled}


class OccluderSet:
    def __init__(self, server, path: str | None = None):
        """path: el archivo de las hojas fijas. None: sin archivo, solo hojas por episodio."""
        self.s = server
        self.path = Path(path).expanduser() if path else None
        self.items: dict[str, Occluder] = {}
        self._episode_names: list[str] = []
        if self.path:
            for entry in load_occluder_config(self.path):
                self.add(**entry)
            print(f"[xArm5] {len(self.items)} oclusores de {self.path}")

    def add(self, name: str, ply: str, position=None, wxyz=(1.0, 0.0, 0.0, 0.0), enabled: bool = True) -> Occluder:
        """Sin position queda donde estaba en la planta."""
        info = load_leaf_info(ply)
        item = Occluder(name, ply, info["center"], position if position is not None else info["center"],
                        wxyz, enabled)
        self.s.populate_transformations_cache(
            self._cache_key(name), canonical_transformation(info["transformation"], info["center"]))
        self.items[name] = item
        return item

    def remove(self, name: str) -> None:
        self.items.pop(name)

    def set_episode(self, entries: list[dict]) -> list[str]:
        """Reemplaza las hojas del episodio anterior (las fijas quedan). Devuelve sus nombres."""
        for name in self._episode_names:
            self.items.pop(name, None)
        self._episode_names = [self.add(**{**e, "name": self.free_name(e["name"])}).name for e in entries]
        return self._episode_names

    def free_name(self, base: str) -> str:
        name, n = base, 2
        while name in self.items:
            name, n = f"{base}_{n}", n + 1
        return name

    def set_pose(self, name: str, position, wxyz) -> None:
        item = self.items[name]
        item.position = tuple(float(v) for v in position)
        item.wxyz = tuple(float(v) for v in wxyz)

    def save(self) -> None:
        save_occluder_config(self.path, [item.entry() for item in self.items.values()])

    def append_to(self, gaussian) -> None:
        """Agrega las hojas activas al splat ya armado de este frame."""
        placed = [self._placed(item) for item in list(self.items.values()) if item.enabled]
        if not placed:
            return
        with torch.no_grad():
            for i, field in enumerate(GAUSSIAN_FIELDS):
                setattr(gaussian, field, torch.cat([getattr(gaussian, field)] + [p[i] for p in placed], dim=0))

    @staticmethod
    def _cache_key(name: str) -> str:
        return f"occluder/{name}"

    def _placed(self, item: Occluder) -> tuple:
        pose = (item.position, item.wxyz)
        if item.placed_pose != pose:
            robot = self.s.robot_name
            with torch.no_grad():
                xyz, rot, opacity, scaling, dc, rest = transform_object(
                    item.gaussian,
                    object_config={"aabb": NO_AABB},
                    pos=torch.tensor(np.array(pose[0]), device="cuda").float(),
                    quat=torch.tensor(np.array(pose[1]), device="cuda").float(),
                    robot_transformation=self.s.object_config[robot]["transformation"]["matrix"],
                    object_name=self._cache_key(item.name),
                    robot_name=robot,
                    transformations_cache=self.s.transformations_cache,
                )
            item.placed = (xyz, rot, opacity, scaling, dc, rest)
            item.placed_pose = pose
        return item.placed
