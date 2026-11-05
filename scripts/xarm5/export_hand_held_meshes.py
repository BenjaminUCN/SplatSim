# -*- coding: utf-8 -*-
"""Exporta el gripper hand_held_XL330 a las mallas que usa el simulador (se corre con FreeCAD).

    "C:\\Program Files\\FreeCAD 1.1\\bin\\freecadcmd.exe" scripts/xarm5/export_hand_held_meshes.py

Lee ~/Downloads/hand_held_XL330/hand_held_assembly_XL330.step y ~/Downloads/mount_xarm5.FCStd y
escribe splatsim/robot_definitions/meshes/hand_held_xl330/: un STL por pieza, en mm y en el marco
del ensamble, y manifest.json. Solo hay que volver a correrlo si cambia el CAD.
"""
import json
import os

import FreeCAD as App
import Import
import Mesh
import Part

REPO = os.environ.get("SPLATSIM_REPO", os.path.expanduser("~/Downloads/SplatSim"))
CAD = os.path.expanduser("~/Downloads")
OUT = os.path.join(REPO, "splatsim/robot_definitions/meshes/hand_held_xl330")
TOL = 0.3    # mm: desviación de la malla

# parte -> (link, color, colisión, sólidos del STEP)
PARTS = {
    "mount_xarm5": ("base", (0.35, 0.35, 0.38), True, None),
    "bottom_plate": ("base", (0.85, 0.35, 0.10), True, ["bottom_plate"]),
    "top_cover": ("base", (0.20, 0.50, 0.85), True, ["top_cover"]),
    "xl330": ("base", (0.14, 0.15, 0.17), True,
              ["Cuerpo_central", "Tapa_frontal", "Tapa_trasera", "Pieza_trasera", "Horn_idler",
               "Buje_idler", "Tornillo_idler", "Conector_1", "Conector_2", "Tornillo_carcasa_1",
               "Tornillo_carcasa_2", "Tornillo_carcasa_3", "Tornillo_carcasa_4"]),
    "tornillos_m2": ("base", (0.15, 0.35, 0.75), False,
                     ["Tornillo_M2x8_1", "Tornillo_M2x8_2", "Tornillo_M2x8_3", "Tornillo_M2x8_4"]),
    "horn": ("horn", (0.62, 0.64, 0.68), False, ["Horn_XL330", "Tornillo_central"]),
}


def mount_xarm5_shape():
    """El disco de mount_xarm5, atornillado directo a bottom_plate: sus agujeros ciegos (±20, 8.7)
    calzan con los de la placa (x = ±20, z = -7.1), con el centro en z = -15.8 y su cara z = 0
    contra la cara y = -41.63 de la placa."""
    doc = App.openDocument(os.path.join(CAD, "mount_xarm5.FCStd"))
    disc = doc.getObject("Body001").Shape.copy()
    m = App.Matrix(1, 0, 0, 0,
                   0, 0, -1, -41.63,
                   0, 1, 0, -15.8,
                   0, 0, 0, 1)
    return disc.transformGeometry(m)


def main():
    os.makedirs(OUT, exist_ok=True)
    doc = App.newDocument("HandHeld")
    Import.insert(os.path.join(CAD, "hand_held_XL330/hand_held_assembly_XL330.step"), doc.Name)
    shapes = {o.Label: o.Shape for o in doc.Objects if o.TypeId == "Part::Feature"}
    disc = mount_xarm5_shape()

    # el pruner está trazado desde fotos, no sale del STEP: se conservan sus piezas
    with open(os.path.join(OUT, "manifest.json")) as f:
        old = json.load(f)
    manifest = [p for p in old["parts"] if p.get("traced")]
    for name, (link, rgb, collision, solids) in PARTS.items():
        shape = disc if solids is None else Part.makeCompound([shapes[s] for s in solids])
        Mesh.Mesh(shape.tessellate(TOL)).write(os.path.join(OUT, name + ".stl"))
        com = _compound_com(shape)
        bb = shape.BoundBox
        manifest.append({"name": name, "file": name + ".stl", "link": link, "rgb": list(rgb),
                         "collision": collision, "volume_mm3": round(shape.Volume, 1),
                         "com_mm": [round(com.x, 3), round(com.y, 3), round(com.z, 3)]})
        print("EXP %-13s %-5s vol %9.1f X %7.2f..%7.2f Y %7.2f..%7.2f Z %7.2f..%7.2f %6.0f kB" % (
            name, link, shape.Volume, bb.XMin, bb.XMax, bb.YMin, bb.YMax, bb.ZMin, bb.ZMax,
            os.path.getsize(os.path.join(OUT, name + ".stl")) / 1000))
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"source": old["source"], "units": "mm",
                   "parts": manifest}, f, indent=1)

    # el disco solo debe tocar a bottom_plate por la cara, sin penetrar nada
    for other in ("bottom_plate", "top_cover", "xl330"):
        print("EXP disco vs %s: comun %.4f mm3, distancia %.4f mm" % (
            other, disc.common(shapes[other]).Volume, disc.distToShape(shapes[other])[0]))


def _compound_com(shape):
    total = sum(s.Volume for s in shape.Solids)
    v = App.Vector()
    for s in shape.Solids:
        v += s.CenterOfMass * (s.Volume / total)
    return v


main()
