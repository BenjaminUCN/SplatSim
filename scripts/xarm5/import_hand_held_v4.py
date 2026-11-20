"""Mallas del gripper hand_held_XL330 v4 para el simulador, desde el paquete del CAD (paquete_v4/).

    python scripts/xarm5/import_hand_held_v4.py ~/Downloads/paquete_v4

Escribe splatsim/robot_definitions/meshes/hand_held_xl330_v4/ con el mismo formato que hand_held_xl330/
(mm, marco del ensamble, manifest.json), sin tocar el v3. Respecto del v3:
  - XL330, tornillos M2 y horn: los del v3 movidos 9.30 mm hacia el disco del brazo (el paquete no los trae).
  - bottom_plate con las dos columnas, y el pruner con las medidas reales, de meshes_mm/.
  - Adaptador HN05 v3 (en el horn) y bastón (link propio), puestos con el gripper abierto, que es el 0 de
    los joints del URDF.
  - manifest.json lleva además la geometría que cambió (eje del horn, centro del filo) y el cuatro barras
    (ver splatsim/utils/scene/hand_held_linkage.py).
mount_xarm5 y top_cover son los del v3: el paquete no dice que cambien.
"""
import argparse
import json
import math
import os
import shutil
import struct

import numpy as np

from splatsim.utils.scene.hand_held import MESH_DIR, read_stl
from splatsim.utils.scene.hand_held_linkage import Linkage

OUT_DIR = "splatsim/robot_definitions/meshes/hand_held_xl330_v4"
SERVO_SHIFT_MM = (0.0, -9.30, 0.0)
DZ_SERVO = -15.8                       # z del ensamble = z del servo + DZ_SERVO
ADAPTER_FACE_Z = 9.0                   # cara del adaptador (servo): el bastón se apoya ahí
LINKAGE = {                            # construir_hand_held_XL330_v4.py
    "horn_axis_mm": [12.7, round(13.86524 - 9.30, 5)],
    "pivot_mm": [9.66, 58.97],
    "yoke_mm": [-13.88, 57.64],
    "jaw_travel_rad": 0.615,
    "crank_mm": 11.5,
    "rod_mm": round(68.72 - 2 * (1.5 + 1.5), 2),
    "branch": 1,
}
# mitad del filo: el x e y del v3 (el filo del yunque no se movió) y z a media altura de la mandíbula v4
GRIPPER_CENTER_MM = [-9.9, 97.0, round((-8.74 + 0.31) / 2, 3)]
FROM_V3 = ("mount_xarm5", "top_cover")
SERVO_V3 = ("xl330", "tornillos_m2", "horn")
FROM_PACKAGE = {"bottom_plate": "bottom_plate_v4.stl", "fixed_steel": "fixed_steel.stl", "anvil": "anvil.stl",
                "anvil_lip": "anvil_lip.stl", "pivot_bolt": "pivot_bolt.stl", "moving_jaw": "moving_jaw.stl",
                "moving_plate": "moving_plate.stl"}
NEW_PARTS = {"adaptador_hn05": ("horn", [0.92, 0.92, 0.9], False), "baston": ("rod", [0.3, 0.3, 0.33], False)}


def write_stl(path: str, triangles: np.ndarray) -> None:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(triangles), np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    data["n"], data["v"] = normals, triangles.reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(b"hand_held_XL330 v4, mm, assembly frame".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        data.tofile(f)


def volume_and_com(triangles: np.ndarray) -> tuple[float, list]:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    volume = v.sum()
    com = (v[:, None] * (a + b + c) / 4).sum(0) / volume
    return abs(float(volume)), [round(float(x), 3) for x in com]


def place(triangles: np.ndarray, angle: float, offset) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return triangles @ R.T + np.asarray(offset, float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("package", help="paquete_v4/ folder (meshes_mm/ inside)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    meshes = os.path.join(os.path.expanduser(args.package), "meshes_mm")

    linkage = Linkage.from_manifest(LINKAGE)
    with open(os.path.join(MESH_DIR, "manifest.json"), encoding="utf-8") as f:
        v3 = {p["name"]: p for p in json.load(f)["parts"]}

    triangles = {}
    for name in FROM_V3:
        triangles[name] = read_stl(os.path.join(MESH_DIR, v3[name]["file"]))
    for name in SERVO_V3:
        triangles[name] = read_stl(os.path.join(MESH_DIR, v3[name]["file"])) + np.array(SERVO_SHIFT_MM)
    for name, filename in FROM_PACKAGE.items():
        triangles[name] = read_stl(os.path.join(meshes, filename))
    # abierto (t = 1): el adaptador con una ranura (a 45° en su malla) sobre el pasador del horn
    axis = (*linkage.horn_axis_mm, DZ_SERVO)
    triangles["adaptador_hn05"] = place(read_stl(os.path.join(meshes, "adaptador_hn05_v3.stl")),
                                        linkage.horn_angle(1.0) - math.radians(45), axis)
    pin = linkage.horn_pin(1.0)
    triangles["baston"] = place(read_stl(os.path.join(meshes, "baston_68_72.stl")), linkage.rod_angle(1.0),
                                (pin[0], pin[1], ADAPTER_FACE_Z + DZ_SERVO))

    os.makedirs(args.out_dir, exist_ok=True)
    parts = []
    for name, tris in triangles.items():
        filename = name + ".stl"
        if name in FROM_V3:
            shutil.copy2(os.path.join(MESH_DIR, v3[name]["file"]), os.path.join(args.out_dir, filename))
        else:
            write_stl(os.path.join(args.out_dir, filename), tris)
        volume, com = volume_and_com(tris)
        link, rgb, collision = NEW_PARTS[name] if name in NEW_PARTS else (v3[name]["link"], v3[name]["rgb"],
                                                                           v3[name]["collision"])
        parts.append({"name": name, "file": filename, "link": link, "rgb": rgb, "collision": collision,
                      "volume_mm3": round(volume, 1), "com_mm": com})
        print(f"{name:15s} {link:5s} {len(tris):6d} triangles, {volume:9.1f} mm3")

    manifest = {
        "source": "paquete_v4 (construir_hand_held_XL330_v4.py): v3 servo moved 9.30 mm, measured pruner, "
                  "HN05 adapter v3 and rod; mount_xarm5 and top_cover from v3",
        "units": "mm",
        "gripper": "hand_held_xl330_v4",
        "horn_axis_mm": LINKAGE["horn_axis_mm"],
        "pivot_mm": LINKAGE["pivot_mm"],
        "jaw_travel_rad": LINKAGE["jaw_travel_rad"],
        "gripper_center_mm": GRIPPER_CENTER_MM,
        "linkage": LINKAGE,
        "parts": parts,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"horn travel {math.degrees(linkage.travel_rad):.1f} deg; wrote {args.out_dir}")


if __name__ == "__main__":
    main()
