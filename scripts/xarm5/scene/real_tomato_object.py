"""Agrega a objects.yaml un tomate real a partir de su splat recortado (splat_from_video.py --carve).

    python scripts/xarm5/scene/real_tomato_object.py ~/data/Colmaps/tomate_tab --name tomato_real_tab \\
        --scene xarm5_lite_plant_v3_hand_held_down_cam_down_tilt

Deja el objeto en el marco del tomate sintético (origen en la base del fruto, z a lo largo del
pedúnculo, metros; ver splatsim/utils/scene/real_object.py) y lo escala al diámetro pedido, que por
defecto es el del sintético para que calce con la malla de colisión de plastic_apple.

Escribe:
- ~/data/output/<name>/point_cloud/iteration_<N>/point_cloud.ply: solo la componente del tomate, sin
  los restos sueltos del recorte;
- <name>.png al lado: vistas del objeto con el pedúnculo encontrado, para revisarlo;
- en objects.yaml, el objeto <name> (transformation, aabb, ply_path, peduncle con el cut_range_mm del
  tomate sintético) y, con --scene, la escena <scene>_<name>: una copia de <scene> que usa este tomate
  (object_overrides). Volver a correrlo reemplaza ambos sin duplicarlos.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # raíz del repo
from scripts.xarm5.scene.add_peduncle import replace_block  # noqa: E402
from splatsim.robots.xarm.config import APPLE_NAME  # noqa: E402
from splatsim.utils.paths import OBJECTS_CONFIG  # noqa: E402
from splatsim.utils.scene.hand_held import read_ply, write_ply  # noqa: E402
from splatsim.utils.scene.real_object import colors, fit_tomato, opacities  # noqa: E402

SYNTHETIC = f"{APPLE_NAME}_peduncle"


@dataclass
class Args:
    source: tyro.conf.Positional[Path]
    """carpeta de splat_from_video.py (con output/point_cloud/iteration_N/point_cloud_carved.ply)"""
    name: str = "tomato_real"
    diameter: float = 0.0728
    """m, ancho del fruto (el del tomate sintético por defecto)"""
    yaw: float = 0.0
    """grados de giro sobre el pedúnculo; con 0 la rama se inclina hacia +x"""
    scene: str = ""
    """escena de objects.yaml que se copia como <escena>_<name> usando este tomate"""


def carved_ply(scene: Path) -> Path:
    plys = sorted((scene / "output" / "point_cloud").glob("iteration_*/point_cloud_carved.ply"),
                  key=lambda p: int(p.parent.name.split("_")[1]))
    if not plys:
        raise SystemExit(f"No hay point_cloud_carved.ply en {scene}/output: corre splat_from_video.py --carve")
    return plys[-1]


def save_views(path: Path, vertices, fit) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.stack([vertices[k] for k in "xyz"], axis=1).astype(float)
    body = xyz @ fit.matrix[:3, :3].T + fit.matrix[:3, 3]
    keep = fit.keep & (opacities(vertices) > 0.2)
    body, rgb = body[keep], colors(vertices)[keep]
    stem = fit.peduncle
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, (u, w, label) in zip(axes, [(0, 1, "x"), (1, 0, "y")]):
        order = np.argsort(body[:, w] * (1 if u == 0 else -1))
        ax.scatter(body[order, u], body[order, 2], c=rgb[order], s=3)
        ax.plot([0, 0], [stem["bottom"][2], stem["top"][2]], "c-", lw=1, label="pedúnculo recto")
        ax.plot(0, stem["cut"][2], "y+", ms=14, mew=2, label="cut")
        ax.set_xlabel(f"{label} (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(path.stem)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def main(args: Args) -> None:
    source = carved_ply(args.source.expanduser())
    vertices = read_ply(str(source))
    fit = fit_tomato(vertices, args.diameter, args.yaw)

    out = Path.home() / "data" / "output" / args.name / "point_cloud" / source.parent.name / "point_cloud.ply"
    write_ply(str(out), vertices[fit.keep])
    save_views(out.parents[2] / f"{args.name}.png", vertices, fit)

    with open(OBJECTS_CONFIG, encoding="utf-8") as f:
        text = f.read()
    config = yaml.safe_load(text)
    peduncle = dict(fit.peduncle)
    if "cut_range_mm" in config.get(SYNTHETIC, {}).get("peduncle", {}):
        peduncle["cut_range_mm"] = config[SYNTHETIC]["peduncle"]["cut_range_mm"]
    entry = {"transformation": {"matrix": [[round(float(v), 6) for v in row] for row in fit.matrix]},
             "aabb": {"bounding_box": fit.aabb},
             "ply_path": f"~/{out.relative_to(Path.home())}",
             "peduncle": peduncle}
    text = replace_block(text, args.name, yaml.safe_dump({args.name: entry}, sort_keys=False, default_flow_style=None))
    if args.scene:
        scene = dict(config[args.scene])
        scene["object_overrides"] = {APPLE_NAME: args.name}
        name = f"{args.scene}_{args.name}"
        text = replace_block(text, name, yaml.safe_dump({name: scene}, sort_keys=False, default_flow_style=None))
    with open(OBJECTS_CONFIG, "w", encoding="utf-8") as f:
        f.write(text)

    scale = np.cbrt(np.linalg.det(fit.matrix[:3, :3]))
    print(f"{fit.keep.sum()} de {len(vertices)} gaussianas en {out}")
    print(f"escala {scale:.4f} m por unidad de COLMAP (radio del fruto {fit.radius_colmap:.3f} u)")
    print(f"pedúnculo: {peduncle}")
    print(f"aabb: {fit.aabb}")
    print(f"{args.name}{' y ' + name if args.scene else ''} en {OBJECTS_CONFIG}; vistas en {out.parents[2] / (args.name + '.png')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    sys.stdout.flush()
    os._exit(0)
