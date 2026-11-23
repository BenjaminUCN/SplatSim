"""URDF de un objeto de objects.yaml con su splat como malla visual, para verlo en PyBullet y en el panel.

    python scripts/xarm5/splat_object_urdf.py tomato_real_tab

La malla sale de las gaussianas del objeto (splatsim/utils/scene/splat_mesh.py), en el marco del cuerpo
(transformation del objeto). Masa, roce y malla de colisión son las del URDF de --template (por defecto
plastic_apple de urdf_models), así que la física no cambia: un tomate real escalado con
real_tomato_object.py ya calza con esa malla de colisión.

Escribe en ~/data/output/<objeto>/urdf/: model.urdf, visual.obj (+ .mtl y la paleta .png), collision.obj,
visual_colors.ply (la misma malla con color por vértice) y preview.png (vistas en PyBullet, para revisarla).
En objects.yaml agrega urdf_path al objeto; la simulación carga ese URDF en vez del de urdf_models. Volver a
correrlo lo reemplaza.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from scripts.xarm5.add_peduncle import replace_block  # noqa: E402
from splatsim.utils.paths import OBJECTS_CONFIG  # noqa: E402
from splatsim.utils.scene.hand_held import read_ply  # noqa: E402
from splatsim.utils.scene.real_object import colors, opacities  # noqa: E402
from splatsim.utils.scene.splat_mesh import splat_mesh, write_object_urdf  # noqa: E402


@dataclass
class Args:
    name: tyro.conf.Positional[str]
    """objeto de objects.yaml (con ply_path, transformation y aabb)"""
    template: str = "plastic_apple"
    """objeto de urdf_models del que salen masa, roce y colisión"""
    voxel: float = 0.0015
    """m por vóxel; el tallo (~3 mm de radio) necesita 1.5 mm o menos"""
    sigma: float = 2.0
    """suavizado de la densidad, en vóxeles"""
    threshold: float = 0.015
    """densidad del borde: más bajo engorda la malla y cierra huecos, más alto la adelgaza"""
    faces: int = 20000
    """caras de la malla simplificada"""


def save_preview(urdf: Path, bounds: np.ndarray, out: Path) -> None:
    """Tres vistas del URDF con el renderer de PyBullet (sin pantalla), encuadradas en bounds (la malla visual;
    getAABB de PyBullet da la de colisión)."""
    import pybullet as p
    from PIL import Image

    client = p.connect(p.DIRECT)
    p.loadURDF(str(urdf), physicsClientId=client)
    center, size = bounds.mean(axis=0), float(np.max(np.ptp(bounds, axis=0)))
    views = []
    for direction in ([1, 0, 0.3], [0, 1, 0.3], [0.6, 0.6, 1]):
        eye = center + 2.2 * size * np.array(direction) / np.linalg.norm(direction)
        view = p.computeViewMatrix(eye.tolist(), center.tolist(), [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(35, 1, 0.01, 5)
        rgba = p.getCameraImage(400, 400, view, projection, renderer=p.ER_TINY_RENDERER, physicsClientId=client)[2]
        views.append(np.reshape(rgba, (400, 400, 4))[..., :3].astype(np.uint8))
    p.disconnect(client)
    Image.fromarray(np.concatenate(views, axis=1)).save(out)


def main(args: Args) -> None:
    import urdf_models.models_data as md

    with open(OBJECTS_CONFIG, encoding="utf-8") as f:
        text = f.read()
    entry = yaml.safe_load(text)[args.name]
    matrix = np.array(entry["transformation"]["matrix"], float)
    vertices = read_ply(str(Path(entry["ply_path"]).expanduser()))
    xyz = np.stack([vertices[k] for k in "xyz"], axis=1).astype(float) @ matrix[:3, :3].T + matrix[:3, 3]
    lo, hi = entry["aabb"]["bounding_box"]
    mesh = splat_mesh(xyz, opacities(vertices), colors(vertices), lo, hi, voxel=args.voxel, sigma=args.sigma,
                      threshold=args.threshold, faces=args.faces)

    folder = Path(entry["ply_path"]).expanduser().parents[2] / "urdf"
    urdf = write_object_urdf(folder, Path(md.model_lib()[args.template]), *mesh)
    save_preview(urdf, np.array([mesh[0].min(axis=0), mesh[0].max(axis=0)]), folder / "preview.png")

    entry["urdf_path"] = f"~/{urdf.relative_to(Path.home())}" if urdf.is_relative_to(Path.home()) else str(urdf)
    text = replace_block(text, args.name, yaml.safe_dump({args.name: entry}, sort_keys=False, default_flow_style=None))
    with open(OBJECTS_CONFIG, "w", encoding="utf-8") as f:
        f.write(text)
    v, f_, _ = mesh
    extent = np.ptp(v, axis=0) * 100
    print(f"malla {len(v)} vértices, {len(f_)} caras, {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} cm")
    print(f"{urdf} (colisión de {args.template}); vistas en {folder / 'preview.png'}")
    print(f"urdf_path de {args.name} en {OBJECTS_CONFIG}")


if __name__ == "__main__":
    main(tyro.cli(Args))
