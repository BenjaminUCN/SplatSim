"""Placa adaptadora para correr de lado la D415 de la muñeca (xArm5 + pinza B3), sobre la cuña de la cámara.

Por qué: con la D415 sobre la cuña de 15°, la lente RGB queda 35 mm a la izquierda del eje de la brida, porque la
RGB está a 35.2 mm del centro de la carcasa y la carcasa está centrada. La hoja de la B3 corta en y = +13.8 mm de
link_eef, así que el punto de corte aparece en la esquina derecha de wrist_rgb (u = 0.76 del ancho). Barrido del
25-nov: docs en ~/data/experiments/261125_camsweep.

La placa se atornilla en los dos M3 de la pestaña del soporte (y de la cuña), a ±22.5 mm, y lleva la D415
**invertida** (girada 180° sobre su eje óptico) con la carcasa corrida SHIFT mm. Invertida, la RGB pasa al otro
extremo: con SHIFT = -20 la lente queda en y = -20 + 35.2 = +15.2 mm, sobre el plano de la hoja, y el punto de corte
queda centrado (u = 0.49). La imagen se endereza girándola 180° en el nodo de la cámara.

Ejes de la placa (mm): x a lo largo de la D415 (= +y de link_eef; los agujeros de la cuña en x = ±22.5), y a lo alto
de la D415 (y = 0 en la línea de los agujeros, x = 67.5 del soporte), z = espesor (la cuña en z = 0).

Montaje:
  1. Tuercas M3 en los dos hexágonos de arriba (agujeros de la cuña).
  2. La D415 invertida (la RGB hacia el lado marcado con la muesca) sobre la placa, con dos M3 x CAM_SCREW desde
     abajo, cabezas en los rebajes.
  3. La placa sobre la cuña con los M3 de siempre desde abajo del soporte, a las tuercas. Si no alcanzan, son
     PLATE_T mm más largos que antes.

    ~/.venvs/cad/bin/python hardware/camera_shift/camera_shift_plate.py [--shift -20] [--out ~/data/cad/camera_shift]

Escribe placa_d415_<shift>.stl (orientado para imprimir, sin soportes), .step y una vista previa .png.
"""

import argparse
import math
from pathlib import Path

from build123d import (
    Align,
    Box,
    Cylinder,
    Location,
    Pos,
    RegularPolygon,
    Rot,
    export_step,
    export_stl,
    extrude,
)

# ---------------------------------------------------------------- parámetros
WEDGE_HOLES = (-22.5,22.5)  # M3 de la pestaña del soporte y de la cuña (x de la placa = y de link_eef)
CAM_HOLES = (45.0)  # separación de los M3 de la espalda de la D415, centrados en la carcasa
CAM_LEN, CAM_H = 99.0, 23.0  # carcasa D415
RGB_FROM_CENTER = (35.2)  # la RGB respecto del centro de la carcasa (hand_held.D415_CENTER_IN_RGB_M)
PLATE_T = 5.0
MARGIN = 9.0  # placa más allá del agujero extremo
M3 = 3.4  # pasante
M3_HEAD, M3_HEAD_H = 6.2, 3.2  # rebaje de la cabeza (Allen M3: 5.5 x 3)
M3_NUT_AF, M3_NUT_H = 5.9, 2.6  # hexágono de la tuerca (5.5 entre caras + holgura)
CAM_SCREW = (6)  # largo del M3 a la cámara: PLATE_T - M3_HEAD_H = 1.8 de placa + ~4 en la rosca
NOTCH = 3.0  # muesca del lado de la RGB


def plate(shift: float):
    cam = [shift - CAM_HOLES / 2, shift + CAM_HOLES / 2]
    xs = [*WEDGE_HOLES, *cam]
    x0, x1 = min(xs) - MARGIN, max(xs) + MARGIN
    rgb = shift + RGB_FROM_CENTER  # invertida: la RGB al lado +x

    body = Pos((x0 + x1) / 2, 0, 0) * Box(
        x1 - x0, CAM_H, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )

    for x in WEDGE_HOLES:  # a la cuña: pasante + tuerca arriba
        body -= Pos(x, 0, 0) * Cylinder(
            M3 / 2, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        nut = extrude(RegularPolygon(M3_NUT_AF / math.sqrt(3), 6), M3_NUT_H)
        body -= Pos(x, 0, PLATE_T - M3_NUT_H) * nut

    for x in cam:  # a la cámara: pasante + cabeza abajo
        body -= Pos(x, 0, 0) * Cylinder(
            M3 / 2, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        body -= Pos(x, 0, 0) * Cylinder(
            M3_HEAD / 2, M3_HEAD_H, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )

    # muesca en V en el canto del lado de la RGB, para no montarla al revés
    notch = (
        Pos(x1, 0, 0)
        * Rot(0, 0, 45)
        * Box(
            NOTCH * math.sqrt(2),
            NOTCH * math.sqrt(2),
            PLATE_T,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    )
    body -= notch
    for a, b in ((x, y) for x in cam for y in WEDGE_HOLES):
        assert abs(a - b) >= M3_HEAD / 2 + M3_NUT_AF / math.sqrt(3) + 1.5, (
            f"agujeros muy juntos: {a}, {b}"
        )
    return body, {
        "x0": x0, 
        "x1": x1, 
        "cam": cam, 
        "rgb": rgb, 
        "body": (shift - CAM_LEN / 2, shift + CAM_LEN / 2)
    }


def preview(part, path: Path, info: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(10, 3.6))
    x0, x1 = info["x0"], info["x1"]
    ax.add_patch(
        Rectangle(
            (-30, -CAM_H / 2 - 4),
            60,
            CAM_H + 8,
            fc="#f3c38b",
            ec="#b0702a",
            label="cuña (60 mm)",
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, -CAM_H / 2),
            x1 - x0,
            CAM_H,
            fc="#9db8d6",
            ec="#2f5d8a",
            alpha=0.8,
            label="placa",
        )
    )
    b0, b1 = info["body"]
    ax.add_patch(
        Rectangle(
            (b0, -CAM_H / 2),
            b1 - b0,
            CAM_H,
            fc="none",
            ec="k",
            ls="--",
            label="D415 (invertida)",
        )
    )
    for x in WEDGE_HOLES:
        ax.add_patch(Circle((x, 0), M3_NUT_AF / 2, fc="w", ec="#b0702a"))
    for x in info["cam"]:
        ax.add_patch(Circle((x, 0), M3_HEAD / 2, fc="w", ec="#2f5d8a"))
        
    ax.plot([info["rgb"]], [0], "ro", label=f"RGB en y = {info['rgb']:+.1f} mm")
    ax.axvline(13.8, color="g", ls=":", label="plano de la hoja (+13.8)")
    ax.axvline(0, color="0.5", lw=0.8)

    ax.set_xlim(min(b0, -30) - 5, max(b1, 30) + 5)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")

    ax.set_xlabel("y de link_eef (mm)")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.set_title("Placa adaptadora D415 — vista desde la cámara hacia la cuña")

    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shift",
        type=float,
        default=-20.0,
        help="centro de la carcasa en y de link_eef (mm)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / "data" / "cad" / "camera_shift"
    )
    args = parser.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    part, info = plate(args.shift)

    name = f"placa_d415_{args.shift:+g}".replace("+", "p").replace("-", "m")

    export_stl(part, str(out / f"{name}.stl"))
    export_step(part, str(out / f"{name}.step"))

    preview(part, out / f"{name}.png", info)

    bb = part.bounding_box()

    print(
        f"{name}: {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm, {part.volume / 1000:.1f} cm3"
    )
    print(
        f"  RGB en y = {info['rgb']:+.1f} mm de link_eef (antes -35.2): corrimiento de la lente "
        f"{info['rgb'] + 35.2:+.1f} mm en x de la imagen y +{PLATE_T:g} mm hacia adelante "
        f"(build_hand_held_urdf.py --camera-shift-mm {info['rgb'] + 35.2:.1f} 0 {PLATE_T:g})"
    )
    print(
        f"  tornillos: 2 x M3 x {CAM_SCREW} a la cámara; 2 x M3 a la cuña, {PLATE_T:g} mm más largos que los de ahora"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
