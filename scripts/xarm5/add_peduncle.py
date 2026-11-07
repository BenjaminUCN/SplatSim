"""Le agrega al tomate un pedúnculo provisional: un cilindro verde de gaussianas.

    python scripts/xarm5/add_peduncle.py --scene xarm5_lite_plant_v3_hand_held_down [--length 0.05]

Escribe <tomate>_peduncle/.../point_cloud.ply junto al splat original, sin tocarlo, y agrega a
configs/object_configs/objects.yaml el objeto plastic_apple_peduncle con los puntos del tallo, y a la
escena object_overrides, que hace que use ese tomate. Las demás escenas siguen con el tomate original.

Volver a correrlo reemplaza el .ply, el objeto y la escena en objects.yaml sin duplicarlos. Al grabar
un tomate real con pedúnculo, basta cambiar ply_path, aabb y los puntos del objeto.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from splatsim.utils.hand_held import read_ply, write_ply  # noqa: E402
from splatsim.utils.peduncle import LENGTH, RADIUS, SPACING, STEM_RGB, cylinder_points, stem_from_tomato  # noqa: E402

OBJECTS = "configs/object_configs/objects.yaml"
SOURCE = "plastic_apple"
NAME = f"{SOURCE}_peduncle"
SH_C0 = 0.28209479177387814
OPACITY = 0.98


def body_frame(vertices, matrix) -> np.ndarray:
    """Las gaussianas del .ply en el marco del cuerpo del tomate."""
    M = np.asarray(matrix, float)
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    return xyz @ M[:3, :3].T + M[:3, 3]


def stem_gaussians(dtype, points_body, matrix, spacing: float = SPACING, rgb=STEM_RGB) -> np.ndarray:
    """Gaussianas esféricas de color plano, en el marco del .ply."""
    M = np.asarray(matrix, float)
    inv = np.linalg.inv(M)
    scale = np.cbrt(np.linalg.det(M[:3, :3]))
    out = np.zeros(len(points_body), dtype=dtype)
    out["x"], out["y"], out["z"] = (points_body @ inv[:3, :3].T + inv[:3, 3]).T
    dc = (np.asarray(rgb) - 0.5) / SH_C0
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = dc
    out["opacity"] = math.log(OPACITY / (1 - OPACITY))
    for i in range(3):
        out[f"scale_{i}"] = math.log(spacing / scale)
    out["rot_0"] = 1.0
    return out


def replace_block(text: str, key: str, block: str) -> str:
    """text con el bloque de primer nivel key reemplazado por block (o agregado al final). Conserva
    el formato y los comentarios del resto, que yaml.safe_dump perdería."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{key}:")), None)
    if start is None:
        return text + ("" if text.endswith("\n") or not text else "\n") + block
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i].startswith((" ", "-", "#"))), len(lines))
    return "".join(lines[:start]) + block + "".join(lines[end:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="xarm5_lite_plant_v3_hand_held_down")
    parser.add_argument("--length", type=float, default=LENGTH, help="m de tallo visible")
    parser.add_argument("--radius", type=float, default=RADIUS, help="m")
    args = parser.parse_args()

    with open(OBJECTS, encoding="utf-8") as f:
        text = f.read()
    config = yaml.safe_load(text)
    source = config[SOURCE]
    matrix = source["transformation"]["matrix"]
    ply = Path(source["ply_path"]).expanduser()
    vertices = read_ply(str(ply))
    body = body_frame(vertices, matrix)
    low, high = np.asarray(source["aabb"]["bounding_box"])
    tomato = body[np.all((body > low) & (body < high), axis=1)]

    stem = stem_from_tomato(tomato, args.length)
    stem["radius"] = args.radius
    points = cylinder_points(stem["bottom"], stem["top"], args.radius, SPACING, np.random.default_rng(0))
    merged = np.concatenate([vertices, stem_gaussians(vertices.dtype, points, matrix)])

    # <salida>/<tomate>/point_cloud/iteration_N/point_cloud.ply
    iteration = ply.parent.name
    out = ply.parents[2].with_name(NAME) / "point_cloud" / iteration / "point_cloud.ply"
    write_ply(str(out), merged)

    margin = 0.005
    aabb = [[float(v) for v in low], [float(v) for v in high]]
    aabb[1][2] = max(aabb[1][2], stem["top"][2] + margin)
    entry = {"transformation": source["transformation"], "aabb": {"bounding_box": aabb},
             "ply_path": f"~/{out.relative_to(Path.home())}", "peduncle": stem}
    scene = dict(config[args.scene])
    scene["object_overrides"] = {SOURCE: NAME}
    text = replace_block(text, NAME, yaml.safe_dump({NAME: entry}, sort_keys=False, default_flow_style=None))
    text = replace_block(text, args.scene, yaml.safe_dump({args.scene: scene}, sort_keys=False, default_flow_style=None))
    with open(OBJECTS, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{len(points)} gaussianas de tallo en {out}")
    print(f"Tallo (marco del tomate): {stem}")
    print(f"{NAME} y object_overrides de {args.scene} en {OBJECTS}")


if __name__ == "__main__":
    main()
    os._exit(0)
