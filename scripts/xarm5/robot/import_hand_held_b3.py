"""Mallas de la pinza B3 (v6: triple etapa, relación 8.44) para el simulador, desde pinza_trabajo/ del laptop.

    python scripts/xarm5/robot/import_hand_held_b3.py ~/data/pinza_trabajo

Escribe splatsim/robot_definitions/meshes/hand_held_b3/ con el mismo formato que hand_held_xl330/ (mm, marco
de la placa = el del ensamble, manifest.json), sin tocar v3 ni v4:
  - base y tapa de la pinza v6 (b3/pinza_v6_*_ensamble.stl);
  - pruner con el pivote medido O (b3/B3.json): las piezas de v5/ (pruner_*.stl), que son las del CAD corridas
    T = O - pivote_CAD y con la hoja fija y la placa móvil rehechas; las mallas de la mandíbula están cerradas;
  - XL330, tornillos M2 y horn del v3 corridos al horn de la B3 (H de B3.json), con el piñón z19 en el horn;
  - mount_xarm5 del v3: el soporte al brazo no cambia (los builders B2/B3 dejan la muesca del M6 en z 10).
Todo menos mount_xarm5 va corrido PLATE_DZ_MM en z de la placa (medido en el robot, ver abajo).
Los engranajes intermedios (ejes I y J, sector) quedan dentro de la caja cerrada y no se ven: no se incluyen.
La mandíbula sigue al servo en proporción (HandHeldGripper). Su apertura sale de las puntas medidas con pie de
metro (ley de cosenos, JAW_TIPS_MM): 35.5° abierta en el home y 2.7° cerrada (no cruza el yunque), 32.8° de
recorrido. Los 53.7° de v5.json (pasador P) suponían las puntas a ~44 mm del pivote; están a ~70 mm.
"""
import argparse
import json
import math
import os
import shutil

import numpy as np

from splatsim.utils.scene.hand_held import HORN_AXIS_MM, MESH_DIR, read_stl

OUT_DIR = "splatsim/robot_definitions/meshes/hand_held_b3"
# centro del filo del v4 (x, y del yunque del CAD; z a media altura de la mandíbula medida) corrido T, como el yunque
V4_GRIPPER_CENTER_MM = (-9.9, 97.0, -4.215)
# Medido con pie de metro el 24-nov (regla sobre el flange D 75.5, del lado opuesto a la cámara, que en z de la placa
# es -z): la cara de la hoja fija hacia la regla (z -12.7 en las mallas) queda a 13.65 mm del eje del flange
# (z -15.8) y la cara de afuera de la caja (el fondo de la base, z -21.4) a 22.35 mm: las dos dan -16.75 mm
PLATE_DZ_MM = -16.75
# medido el 24-nov: pivote (perno central) -> punta de la hoja 71.1 y -> punta del yunque 70.0; entre puntas 43.1
# abierta (home, servo ~3025) y 3.5 cerrada (servo ~11671)
JAW_TIPS_MM = {"r_jaw": 71.1, "r_anvil": 70.0, "open": 43.1, "closed": 3.5}
FROM_V3 = ("mount_xarm5",)
SERVO_V3 = ("xl330", "tornillos_m2", "horn")
# (archivo en pinza_trabajo, nombre, link, rgb, collision); los colores, los del v3 para la misma pieza
PARTS = [
    ("b3/pinza_v6_base_ensamble.stl", "bottom_plate", "base", [0.85, 0.35, 0.1], True),
    ("b3/pinza_v6_tapa_ensamble.stl", "top_cover", "base", [0.2, 0.5, 0.85], True),
    ("v5/pruner_hoja_fija.stl", "fixed_steel", "base", [0.35, 0.38, 0.4], False),
    ("v5/pruner_anvil.stl", "anvil", "base", [0.95, 0.62, 0.15], True),
    ("v5/pruner_anvil_lip.stl", "anvil_lip", "base", [0.95, 0.62, 0.15], True),
    ("v5/pruner_pivot_bolt.stl", "pivot_bolt", "base", [0.8, 0.8, 0.8], False),
    ("v5/pruner_moving_jaw.stl", "moving_jaw", "jaw", [0.75, 0.75, 0.72], True),
    ("v5/pruner_moving_plate.stl", "moving_plate", "jaw", [0.08, 0.08, 0.09], False),
    ("b3/pinon_z19_m1_ensamble.stl", "pinon_z19", "horn", [0.9, 0.9, 0.88], False),
]


def write_stl(path: str, triangles: np.ndarray) -> None:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(triangles), np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    data["n"], data["v"] = normals, triangles.reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(b"hand_held B3 (pinza v6), mm, plate frame".ljust(80, b"\0"))
        f.write(np.uint32(len(triangles)).tobytes())
        data.tofile(f)


def volume_and_com(triangles: np.ndarray) -> tuple[float, list]:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    com = (v[:, None] * (a + b + c) / 4).sum(0) / v.sum()
    return abs(float(v.sum())), [round(float(x), 3) for x in com]


def tips_angle_rad(d_mm: float, tips=JAW_TIPS_MM) -> float:
    """Ángulo en el pivote entre las puntas de la hoja y del yunque separadas d_mm (ley de cosenos)."""
    r1, r2 = tips["r_jaw"], tips["r_anvil"]
    return math.acos((r1 ** 2 + r2 ** 2 - d_mm ** 2) / (2 * r1 * r2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("pinza", help="pinza_trabajo/ folder (b3/, v5/ inside)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    root = os.path.expanduser(args.pinza)
    with open(os.path.join(root, "b3", "B3.json"), encoding="utf-8") as f:
        b3 = json.load(f)
    with open(os.path.join(root, "v5", "v5.json"), encoding="utf-8") as f:
        v5 = json.load(f)
    with open(os.path.join(MESH_DIR, "manifest.json"), encoding="utf-8") as f:
        v3 = {p["name"]: p for p in json.load(f)["parts"]}

    pivot = [round(float(x), 4) for x in b3["O"]]
    horn = [round(float(x), 4) for x in b3["H"]]
    servo_shift = np.array([horn[0] - HORN_AXIS_MM[0], horn[1] - HORN_AXIS_MM[1], 0.0])
    T = np.array(v5["T"])
    center = [round(float(V4_GRIPPER_CENTER_MM[0] + T[0]), 3), round(float(V4_GRIPPER_CENTER_MM[1] + T[1]), 3),
              round(V4_GRIPPER_CENTER_MM[2] + PLATE_DZ_MM, 3)]
    dz = np.array([0.0, 0.0, PLATE_DZ_MM])
    jaw_open = tips_angle_rad(JAW_TIPS_MM["open"])
    travel = jaw_open - tips_angle_rad(JAW_TIPS_MM["closed"])

    os.makedirs(args.out_dir, exist_ok=True)
    parts = []

    def add(name, tris, link, rgb, collision, copy_from=None):
        filename = name + ".stl"
        if copy_from:
            shutil.copy2(copy_from, os.path.join(args.out_dir, filename))
        else:
            write_stl(os.path.join(args.out_dir, filename), tris)
        volume, com = volume_and_com(tris)
        parts.append({"name": name, "file": filename, "link": link, "rgb": rgb, "collision": collision,
                      "volume_mm3": round(volume, 1), "com_mm": com})
        print(f"{name:13s} {link:5s} {len(tris):6d} triangles, {volume:9.1f} mm3")

    for name in FROM_V3:
        path = os.path.join(MESH_DIR, v3[name]["file"])
        add(name, read_stl(path), v3[name]["link"], v3[name]["rgb"], v3[name]["collision"], copy_from=path)
    for name in SERVO_V3:
        add(name, read_stl(os.path.join(MESH_DIR, v3[name]["file"])) + servo_shift + dz, v3[name]["link"],
            v3[name]["rgb"], v3[name]["collision"])
    for path, name, link, rgb, collision in PARTS:
        add(name, read_stl(os.path.join(root, path)) + dz, link, rgb, collision)

    manifest = {
        "source": "pinza_trabajo (laptop): b3/ pinza v6 base, tapa and z19 pinion; v5/ pruner at the measured pivot O; "
                  "v3 XL330 and horn moved to the B3 horn; mount_xarm5 from v3; inner gears left out",
        "units": "mm",
        "gripper": "hand_held_xl330",
        "horn_axis_mm": horn,
        "pivot_mm": pivot,
        "jaw_travel_rad": round(travel, 5),
        "jaw_open_rad": round(jaw_open, 5),
        "jaw_tips_mm": JAW_TIPS_MM,
        "gripper_center_mm": center,
        "plate_dz_mm": PLATE_DZ_MM,
        "b3": {k: b3[k] for k in ("ratio", "servo_deg", "crank_deg")},
        "parts": parts,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"pivot {pivot}, horn {horn}, jaw open {math.degrees(jaw_open):.1f} deg, travel {math.degrees(travel):.1f} deg, center {center}; wrote {args.out_dir}")


if __name__ == "__main__":
    main()
