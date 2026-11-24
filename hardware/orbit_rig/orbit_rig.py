"""Soporte orbital impreso para filmar un tomate real con la RealSense D435i (para splat_from_video.py).

El tomate queda quieto en un poste central y la cámara gira alrededor de él:
- azimut: un rotor sobre bolitas (BBs de 6 mm) gira alrededor del poste, que está fijo a la base;
  la base tiene una escala cada 5° y agujeros cada 15° para fijar el giro con un pasador de 3 mm;
- elevación: un brazo lleva un arco de radio ARC_R centrado en el tomate, de -15° (desde abajo) a
  80° (casi cenital); un carro desliza por el arco y se fija con un tornillo M4, y la cámara (rosca
  1/4"-20) va en una ranura que ajusta la distancia al tomate (ver la salida del script).

Se mueve la cámara y no el tomate porque COLMAP necesita que el fondo sea consistente con el objeto
(splat_from_video.py entrena con el fondo y recorta por máscaras después).

Pensado para la Bambu Lab A1 mini (cama de 180 x 180 x 180 mm) y para imprimir rápido: base y rotor
calados, arco en cuatro tramos cortos, ninguna pieza necesita soportes.

    ~/.venvs/cad/bin/python hardware/orbit_rig/orbit_rig.py [--out ~/data/cad/orbit_rig]

Escribe un STL por pieza (ya orientado para imprimir), el ensamblado en STEP y una vista previa.
Medidas en mm; el origen está en la mesa, en el eje de giro.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from build123d import (Align, Box, Circle, Compound, Cone, Cylinder, Polygon, Pos, Rectangle,
                       RegularPolygon, Rot, Sphere, Text, Torus, export_step, export_stl, extrude)

# ---------------------------------------------------------------- parámetros
BED = (180, 180, 180)        # Bambu Lab A1 mini
BASE_R, BASE_T, PLATE = 88, 5, 3      # base: radio, alto de la pista, alto del anillo y los rayos
SCALE_IN = 74                # anillo exterior de la base (escala) desde este radio
ROTOR_R, ROTOR_T = 68, 5
BB = 6.0                     # bolitas del rodamiento axial (airsoft o acero)
RACE_R = 60                  # radio de la pista
GAP = 1.6                    # luz entre base y rotor (las ranuras se comen el resto de la bolita)
BOSS_R, HUB_R, HUB_H = 10, 16, 14
FIT = 0.2                    # holgura radial de piezas que deslizan
ARM_W, ARM_H, ARM_FROM = 26, 16, 18
ARM_BOLTS = (30, 48)         # radios de los M4 brazo -> rotor (lejos de la pista)
ARC_R = 230                  # radio del arco (línea media), centrado en el tomate
ARC_RAD, ARC_AX = 12, 18     # sección del arco: radial x axial
ARC_FROM, ARC_TO = -20, 88   # grados; la pata baja desde ARC_FROM
JOINTS = (7.5, 32.5, 57.5)   # uniones entre los tramos del arco (lejos de las marcas cada 5°)
TOMATO_ABOVE_ROTOR = 110     # centro del tomate sobre la cara superior del rotor
POST_R = 7
HAND_POST_H, FOOT_R, FOOT_T = 120, 55, 4   # pedestal suelto (sin la base giratoria), para filmar a mano
CAM = (90, 25, 25)           # D435i: largo, alto, profundidad
M3, M3_HEAD, M3_NUT = 3.3, 6.2, 5.8 / math.sqrt(3)     # nut: radio circunscrito de la tuerca
M4, M4_HEAD, M4_NUT = 4.4, 8.2, 7.1 / math.sqrt(3)
QUARTER = 6.6                # tornillo de cámara 1/4"-20
ENGRAVE = 0.6

Z_ROTOR = BASE_T + GAP                    # cara inferior del rotor
Z_ROTOR_TOP = Z_ROTOR + ROTOR_T
Z_RACE = BASE_T + GAP / 2
HUB_TOP = Z_ROTOR_TOP + HUB_H
Z_ARM_TOP = Z_ROTOR_TOP + ARM_H
Z_C = Z_ROTOR_TOP + TOMATO_ABOVE_ROTOR    # centro del tomate = centro del arco
Z_BOSS_TOP = HUB_TOP + 0.3
Z_POST_TOP = Z_C - 62                     # con la punta-pedestal metida a fondo, el tomate queda ~10 mm bajo Z_C
LEG_DEPTH = 12                            # cuánto entra la pata del arco en el brazo

ARC_IN, ARC_OUT = ARC_R - ARC_RAD / 2, ARC_R + ARC_RAD / 2
LEG_X = ARC_R * math.cos(math.radians(ARC_FROM))
ARM_END = LEG_X + ARC_RAD / 2 + 7

# carro: manga alrededor del arco y placa de la cámara del lado del tomate
CLEAR = FIT * 2.5                          # holgura de la manga (el arco es curvo)
SLEEVE_WALL = 3
PLATE_T, PLATE_W = 6, 36
SLEEVE_HALF = CAM[1] / 2 + PLATE_T
SLEEVE_IN = ARC_IN - CLEAR - SLEEVE_WALL   # cara interior de la manga (radio)
SLOT = (SLEEVE_IN - CAM[2] / 2 - 50, SLEEVE_IN - CAM[2] / 2)   # radios posibles del tornillo 1/4"
PLATE_IN = SLOT[0] - 7


# ---------------------------------------------------------------- utilidades
def sector(ri, ro, a0, a1, z0, z1):
    """Sector anular en el plano XY (grados), extruido de z0 a z1."""
    angles = np.radians(np.linspace(a0, a1, 32))
    wedge = Polygon((0, 0), *[(3 * ro * math.cos(a), 3 * ro * math.sin(a)) for a in angles], align=None)
    face = (Circle(ro) - Circle(ri)) & wedge
    return Pos(0, 0, z0) * extrude(face, z1 - z0)


def cyl(r, z0, z1, x=0, y=0):
    return Pos(x, y, z0) * Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))


def ring(ri, ro, z0, z1):
    return cyl(ro, z0, z1) - cyl(ri, z0 - 1, z1 + 1)


def box(x0, x1, y0, y1, z0, z1):
    return Pos(x0, y0, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=(Align.MIN, Align.MIN, Align.MIN))


def hex_pocket(r, z0, z1, x=0, y=0):
    return Pos(x, y, z0) * extrude(RegularPolygon(r, 6), z1 - z0)


def polar(r, deg):
    return r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))


def engrave_ticks(r_out, z_top, angles, long_every, short_len, long_len, labels=None, label_r=None, font=3.5,
                  label_rot=-90):
    """Marcas radiales (y números) grabadas en una cara plana z = z_top."""
    cut = None
    for a in angles:
        length = long_len if a % long_every == 0 else short_len
        tick = Rot(0, 0, a) * Pos(r_out - length / 2, 0) * Rectangle(length, 0.7)
        part = Pos(0, 0, z_top - ENGRAVE) * extrude(tick, ENGRAVE + 1)
        cut = part if cut is None else cut + part
    for a in labels or []:
        text = Rot(0, 0, a) * Pos(label_r, 0) * Rot(0, 0, label_rot) * Text(f"{a}", font)
        cut += Pos(0, 0, z_top - ENGRAVE) * extrude(text, ENGRAVE + 1)
    return cut


def arc_to_world(shape, theta=0.0):
    """Del plano del arco (x radial, y alto sobre el tomate, z axial) al ensamblado."""
    return Pos(0, 0, Z_C) * Rot(90, 0, 0) * Rot(0, 0, theta) * shape


INDEX = [7.5 + 15 * k for k in range(24)]   # agujeros del pasador, entre las marcas
INDEX_R = 79
PIN_AT = 187.5                              # el pasador del rotor, lejos del brazo


# ---------------------------------------------------------------- piezas
def base():
    """Calada: pista, anillo con la escala, rayos y el eje fijo."""
    b = ring(SCALE_IN, BASE_R, 0, PLATE) + ring(RACE_R - 5.5, RACE_R + 5.5, 0, BASE_T) + cyl(BOSS_R, 0, Z_BOSS_TOP)
    for a in range(30, 360, 120):                                     # tres rayos, lejos de los agujeros
        b += Rot(0, 0, a) * box(BOSS_R - 1, SCALE_IN + 1, -4, 4, 0, PLATE)
    b -= Pos(0, 0, Z_RACE) * Torus(RACE_R, BB / 2 + 0.1)
    b -= cyl(POST_R - 1 + FIT, 3, Z_BOSS_TOP + 1)                    # encaje del poste
    for a in INDEX:
        b -= cyl(1.65, -1, PLATE + 1, *polar(INDEX_R, a))
    b -= engrave_ticks(BASE_R - 0.5, PLATE, range(0, 360, 5), 15, 4, 7,
                       labels=range(0, 360, 30), label_r=SCALE_IN + 2.8, font=3.5, label_rot=-90)
    return b


def rotor():
    r = cyl(ROTOR_R, Z_ROTOR, Z_ROTOR_TOP) + cyl(HUB_R, Z_ROTOR_TOP - 1, HUB_TOP)
    # ventanas para ahorrar plástico (no bajo el brazo ni bajo la copa)
    for a0, a1 in ((25, 80), (100, 145), (215, 260), (280, 335)):
        r -= sector(HUB_R + 4, RACE_R - 6, a0, a1, Z_ROTOR - 1, Z_ROTOR_TOP + 1)
    # copa del contrapeso (tuercas, perdigones...) opuesta al brazo
    cx, cy = polar(46, 180)
    r += cyl(24.5, Z_ROTOR_TOP - 1, Z_ROTOR_TOP + 25, cx, cy) - cyl(22, Z_ROTOR_TOP, Z_ROTOR_TOP + 26, cx, cy)
    # puntero hacia el brazo sobre la escala, y lengüeta del pasador
    r += Pos(0, 0, Z_ROTOR) * extrude(Polygon((ROTOR_R - 3, -4), (ROTOR_R + 8, 0), (ROTOR_R - 3, 4), align=None), 2)
    r += Rot(0, 0, PIN_AT) * box(ROTOR_R - 4, INDEX_R + 5, -5, 5, Z_ROTOR, Z_ROTOR + 3)
    r -= cyl(1.7, Z_ROTOR - 1, Z_ROTOR + 4, *polar(INDEX_R, PIN_AT))
    r -= cyl(BOSS_R + FIT * 2, Z_ROTOR - 1, HUB_TOP + 1)
    r -= Pos(0, 0, Z_RACE) * Torus(RACE_R, BB / 2 + 0.1)
    for x in ARM_BOLTS:                                               # tuercas M4 del brazo
        for y in (-7, 7):
            r -= hex_pocket(M4_NUT, Z_ROTOR_TOP - 3.4, Z_ROTOR_TOP + 1, x, y)
            r -= cyl(M4 / 2, Z_ROTOR - 1, Z_ROTOR_TOP, x, y)
    return r


def post():
    flange = Z_BOSS_TOP + 3
    p = cyl(POST_R - 1, 3.3, Z_BOSS_TOP) + cyl(HUB_R - 1, Z_BOSS_TOP, flange)
    p += Pos(0, 0, flange) * Cone(HUB_R - 1, POST_R, HUB_R - 1 - POST_R, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p += cyl(POST_R, flange, Z_POST_TOP)
    p -= cyl(4 + FIT, Z_POST_TOP - 28, Z_POST_TOP + 1)                 # encaje de la punta
    p -= Pos(0, 0, Z_POST_TOP - 6) * Rot(0, 90, 0) * Cylinder(1.3, 2 * POST_R + 2)   # prisionero M3 (autorrosca)
    return p


def pedestal_tip():
    """Copa en la que se apoya el fruto, pedúnculo arriba."""
    z = Z_POST_TOP
    t = cyl(4, z - 26, z + 6) + Pos(0, 0, z + 4) * Cone(5, 15, 9, align=(Align.CENTER, Align.CENTER, Align.MIN))
    t += cyl(15, z + 13, z + 17)
    t -= cyl(11, z + 12, z + 18)
    return t


def hand_pedestal():
    """Poste con pie propio, para filmar a mano sin la base giratoria: el mismo encaje (y prisionero M3) que
    el poste, así que lleva cualquiera de las dos puntas. Se imprime de pie, sin soportes."""
    top = HAND_POST_H
    p = cyl(FOOT_R, 0, FOOT_T) + cyl(POST_R, 0, top)
    p += Pos(0, 0, FOOT_T) * Cone(22, POST_R, 16, align=(Align.CENTER, Align.CENTER, Align.MIN))   # refuerzo
    p -= cyl(4 + FIT, top - 28, top + 1)                              # encaje de la punta
    p -= Pos(0, 0, top - 6) * Rot(0, 90, 0) * Cylinder(1.3, 2 * POST_R + 2)   # prisionero M3 (autorrosca)
    return p


def hook_tip():
    """Para colgar el tomate de su rama: sube por el costado y la pinza con un M3."""
    z = Z_POST_TOP
    top = Z_C + 70 - 10
    h = cyl(4, z - 26, z + 2) + box(-5, 58, -4, 4, z, z + 8) + box(50, 58, -4, 4, z, top + 8)
    h += box(-8, 58, -4, 4, top, top + 8) + box(-10, 10, -8, 8, top - 2, top + 10)
    h -= cyl(4, top - 3, top + 11)                                     # rama hasta 8 mm
    h -= box(-11, -2, -1, 1, top - 3, top + 11)                        # ranura de la pinza
    h -= Pos(-6, 0, top + 4) * Rot(90, 0, 0) * Cylinder(M3 / 2, 20)
    h -= Pos(-6, 8, top + 4) * Rot(90, 0, 0) * extrude(RegularPolygon(M3_NUT, 6), 3)
    return h


def arm():
    a = box(ARM_FROM, ARM_END, -ARM_W / 2, ARM_W / 2, Z_ROTOR_TOP, Z_ARM_TOP)
    for x in ARM_BOLTS:
        for y in (-7, 7):
            a -= cyl(M4 / 2, Z_ROTOR_TOP - 1, Z_ARM_TOP + 1, x, y)
            a -= cyl(M4_HEAD / 2, Z_ROTOR_TOP + 5, Z_ARM_TOP + 1, x, y)
    # cavidad de la pata del arco y un M3 que la atraviesa
    a -= box(LEG_X - ARC_RAD / 2 - FIT, LEG_X + ARC_RAD / 2 + FIT, -ARC_AX / 2 - FIT, ARC_AX / 2 + FIT,
             Z_ARM_TOP - LEG_DEPTH, Z_ARM_TOP + 1)
    zb = Z_ARM_TOP - LEG_DEPTH / 2
    a -= Pos(LEG_X, 0, zb) * Rot(90, 0, 0) * Cylinder(M3 / 2, ARM_W + 2)
    a -= Pos(LEG_X, ARM_W / 2 - 5, zb) * Rot(-90, 0, 0) * Cylinder(M3_HEAD / 2, 6, align=(
        Align.CENTER, Align.CENTER, Align.MIN))
    a -= Pos(LEG_X, -ARM_W / 2 + 5, zb) * Rot(90, 0, 0) * extrude(RegularPolygon(M3_NUT, 6), 6)
    return a


TONGUE = dict(ri=ARC_R - 3, ro=ARC_R + 3, z0=-ARC_AX / 2, z1=2, length=22)
EDGES = (ARC_FROM, *JOINTS, ARC_TO)


def arc_segment(i):
    """Tramo i del arco en el plano del arco (cara +z grabada, hacia arriba al imprimir)."""
    a0, a1 = EDGES[i], EDGES[i + 1]
    last = i == len(EDGES) - 2
    z0, z1 = -ARC_AX / 2, ARC_AX / 2
    s = sector(ARC_IN, ARC_OUT, a0 - (2 if i == 0 else 0), a1, z0, z1)
    if i == 0:                                                         # pata hasta el brazo
        s += box(LEG_X - ARC_RAD / 2, LEG_X + ARC_RAD / 2, Z_ARM_TOP - LEG_DEPTH - Z_C,
                 ARC_IN * math.sin(math.radians(ARC_FROM)) + 2, z0, z1)
        s -= Pos(LEG_X, Z_ARM_TOP - LEG_DEPTH / 2 - Z_C, 0) * Cylinder(M3 / 2, ARC_AX + 2)
    span = math.degrees(TONGUE["length"] / ARC_R)
    if not last:                                                       # lengüeta hacia el tramo de arriba
        s += sector(TONGUE["ri"], TONGUE["ro"], a1 - 0.5, a1 + span, TONGUE["z0"], TONGUE["z1"])
    if i > 0:                                                          # ranura para la de abajo
        s -= sector(TONGUE["ri"] - FIT, TONGUE["ro"] + FIT, a0 - 1, a0 + span + 0.1, TONGUE["z0"] - 1,
                    TONGUE["z1"] + FIT)
    for joint in [a1] * (not last) + [a0] * (i > 0):
        for da in (1.2, 3.6):
            x, y = polar(ARC_R, joint + da)
            s -= Pos(x, y, 0) * Cylinder(M3 / 2, ARC_AX + 2)
            if joint == a0:                                            # tramo de arriba: tuerca en la cara grabada
                s -= hex_pocket(M3_NUT, z1 - 4.5, z1 + 1, x, y)
            else:                                                      # tramo de abajo: cabeza por debajo
                s -= cyl(M3_HEAD / 2, z0 - 1, z0 + 4, x, y)
    if last:                                                           # tope para que el carro no se salga
        s += sector(ARC_IN - 3, ARC_OUT + 3, a1 - 2, a1, z0, z1)
    ticks = [a for a in range(-15, 90, 5) if a0 + 0.5 < a < a1 - 0.5]
    labels = [a for a in ticks if a % 15 == 0]
    s -= engrave_ticks(ARC_OUT - 0.5, z1, ticks, 15, 3.5, 6, labels=labels, label_r=ARC_IN + 2.6, font=3.2)
    return s


def carriage():
    """Carro en el plano del arco a 0°; la cámara mira hacia el centro con su eje óptico en y = 0."""
    zw = ARC_AX / 2 + CLEAR
    out = ARC_OUT + CLEAR + SLEEVE_WALL
    s = sector(SLEEVE_IN, out, -8, 8, -zw - SLEEVE_WALL, zw + SLEEVE_WALL)
    s -= sector(ARC_IN - CLEAR, ARC_OUT + CLEAR, -9, 9, -zw, zw)
    s &= box(0, 400, -SLEEVE_HALF, SLEEVE_HALF, -50, 50)
    # placa con ranura para el 1/4"-20; arriba de ella queda la cámara
    s += box(PLATE_IN, SLEEVE_IN + 2, -CAM[1] / 2 - PLATE_T, -CAM[1] / 2, -PLATE_W / 2, PLATE_W / 2)
    s -= box(SLOT[0] - QUARTER / 2, SLOT[1] + QUARTER / 2, -CAM[1] / 2 - PLATE_T - 1, -CAM[1] / 2 + 1,
             -QUARTER / 2, QUARTER / 2)
    for x in SLOT:
        s -= Pos(x, -CAM[1] / 2 - PLATE_T / 2, 0) * Rot(90, 0, 0) * Cylinder(QUARTER / 2, PLATE_T + 2)
    # tornillo M4 que aprieta contra la cara exterior del arco
    s += box(out - 1, out + 6, -7, 7, -7, 7)
    s -= Pos(out - 4, 0, 0) * Rot(0, 90, 0) * Cylinder(M4 / 2, 14, align=(Align.CENTER, Align.CENTER, Align.MIN))
    s -= Pos(out + 1, 0, 0) * Rot(0, 90, 0) * extrude(RegularPolygon(M4_NUT, 6), 3.4)
    s -= box(out + 1, out + 4.4, -8, 0, -M4_NUT * 0.87, M4_NUT * 0.87)  # entrada de la tuerca desde abajo
    # ventana de lectura de la escala (cara +z) en el eje óptico
    s -= box(ARC_OUT - 7, ARC_OUT + CLEAR + 0.5, -1, 1, zw - 1, zw + SLEEVE_WALL + 1)
    return s


def camera_dummy():
    x = (SLOT[0] + SLOT[1]) / 2
    return box(x - CAM[2] / 2, x + CAM[2] / 2, -CAM[1] / 2, CAM[1] / 2, -CAM[0] / 2, CAM[0] / 2)


def printable_parts() -> dict:
    """Las piezas, ya giradas como se imprimen (la cara grande abajo, sin voladizos)."""
    return {
        "base": base(), "rotor": rotor(), "brazo": arm(), "poste": Rot(180, 0, 0) * post(),
        "punta_pedestal": Rot(180, 0, 0) * pedestal_tip(), "punta_colgante": Rot(90, 0, 0) * hook_tip(),
        "pedestal_mano": hand_pedestal(),
        **{f"arco_{i + 1}": arc_segment(i) for i in range(len(EDGES) - 1)}, "carro": Rot(90, 0, 0) * carriage(),
    }


# ---------------------------------------------------------------- salida
def on_bed(shape, diagonal=False):
    """Centrada en la cama; en diagonal si así cabe y derecha no."""
    bb = shape.bounding_box()
    if diagonal:
        shape = Rot(0, 0, 45) * shape
        bb = shape.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * shape


def fits(shape) -> str:
    bb = shape.bounding_box()
    if bb.size.Z > BED[2]:
        return "NO CABE (alto)"
    if bb.size.X <= BED[0] and bb.size.Y <= BED[1]:
        return "ok"
    d = Rot(0, 0, 45) * shape
    db = d.bounding_box()
    return "ok en diagonal" if db.size.X <= BED[0] and db.size.Y <= BED[1] else "NO CABE"


def render(parts, path, lims=((-150, 250), (-200, 200), (0, 400))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 8))
    views = [(18, -60), (0, -90)]
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for shape, color in parts:
            verts, tris = shape.tessellate(0.5, 0.5)
            v = np.array([tuple(p) for p in verts])
            ax.add_collection3d(Poly3DCollection(v[np.array(tris)], facecolors=color, edgecolor="none",
                                                 linewidth=0, alpha=1.0, shade=True))
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_box_aspect([hi - lo for lo, hi in lims])
        ax.view_init(elev, azim)
        ax.set_axis_off()
    fig.savefig(path, dpi=80, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / "data" / "cad" / "orbit_rig")
    parser.add_argument("--elevation", type=float, default=30, help="elevación del carro en la vista previa")
    args = parser.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    for name, shape in printable_parts().items():
        verdict = fits(shape)
        shape = on_bed(shape, diagonal=verdict == "ok en diagonal")
        export_stl(shape, str(out / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.15)
        bb = shape.bounding_box()
        print(f"{name:15s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} mm  {shape.volume / 1000:6.1f} cm3"
              f"  {verdict}")

    fruit = Pos(0, 0, Z_C - 10 + 2) * Sphere(36.4)
    assembly = [
        (base(), "#9aa4ad"), (rotor(), "#3b4148"), (arm(), "#3b4148"), (post(), "#d8d8d8"),
        (pedestal_tip(), "#d8d8d8"), (fruit, "#d23a2a"),
        *[(arc_to_world(arc_segment(i)), "#3b4148") for i in range(len(EDGES) - 1)],
        (arc_to_world(carriage(), args.elevation), "#e0a030"),
        (arc_to_world(camera_dummy(), args.elevation), "#202020"),
    ]
    export_step(Compound([s for s, _ in assembly]), str(out / "ensamblado.step"))
    render(assembly, out / "vista_previa.png")
    # el pedestal suelto con su punta y el tomate
    lift = HAND_POST_H - Z_POST_TOP
    render([(hand_pedestal(), "#d8d8d8"), (Pos(0, 0, lift) * pedestal_tip(), "#b0b0b0"),
            (Pos(0, 0, lift + Z_C - 10 + 2) * Sphere(36.4), "#d23a2a")], out / "pedestal_mano.png",
           lims=((-110, 110), (-110, 110), (0, 220)))
    print(f"pedestal_mano: centro del tomate a ~{lift + Z_C - 10 + 2:.0f} mm de la mesa, con la punta-pedestal a fondo")
    lens = [round(x - CAM[2] / 2) for x in SLOT]
    print(f"tomate a {Z_C:.0f} mm de la mesa; frente de la cámara a {lens[0]}–{lens[1]} mm del tomate")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
