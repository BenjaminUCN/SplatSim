"""Crea el URDF del xArm5 con el gripper hand_held_XL330 a partir del URDF actual.

    python scripts/xarm5/build_hand_held_urdf.py
    python scripts/xarm5/build_hand_held_urdf.py --roll-deg 180                  # girado en la brida
    python scripts/xarm5/build_hand_held_urdf.py --moving-part hoja_movil.stl    # gira con el horn

Salida: xarm5_hand_held.urdf y la carpeta hand_held_xl330/ junto al URDF de entrada.
"""
import argparse
import os

from splatsim.robots.station.urdf_view import HAND_HELD_TRAVEL_RAD
from splatsim.utils.hand_held import build_urdf

DEFAULT_SRC = "~/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", help="default: xarm5_hand_held.urdf next to --src")
    parser.add_argument("--roll-deg", type=float, default=0.0,
                        help="rotation about the tool axis; default: GoPro towards the link's +x")
    parser.add_argument("--moving-part", help="STL in mm, in the CAD assembly's frame")
    args = parser.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst) if args.dst else os.path.join(os.path.dirname(src), "xarm5_hand_held.urdf")
    info = build_urdf(src, dst, HAND_HELD_TRAVEL_RAD, roll_deg=args.roll_deg, extra_horn_mesh=args.moving_part)
    print(f"Wrote {dst}")
    print(f"Removed links: {', '.join(info['removed'])}")
    print(f"Kept and moved to {info['parent']}: {', '.join(info['reparented']) or '-'}")
    print(f"Estimated gripper mass: {info['mass_kg']:.3f} kg (solid PLA, not weighed)")


if __name__ == "__main__":
    main()
