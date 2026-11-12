"""Crea el URDF del xArm5 con el gripper hand_held_XL330 a partir del URDF actual.

    python scripts/xarm5/build_hand_held_urdf.py
    python scripts/xarm5/build_hand_held_urdf.py --roll-deg 0                    # girado en la brida
    python scripts/xarm5/build_hand_held_urdf.py --moving-part hoja_movil.stl    # gira con el horn
    python scripts/xarm5/build_hand_held_urdf.py --camera-down --dst .../xarm5_hand_held_down_cam_down.urdf

Salida: xarm5_hand_held.urdf y la carpeta hand_held_xl330/ junto al URDF de entrada.
"""
import argparse
import os

from splatsim.robots.station.urdf_view import HAND_HELD_TRAVEL_RAD
from splatsim.utils.scene.hand_held import build_urdf

DEFAULT_SRC = "~/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5.urdf"
# la tapa azul (top_cover) hacia abajo con joint5 en 0; a 0° quedaba hacia arriba
DEFAULT_ROLL_DEG = 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", help="default: xarm5_hand_held.urdf next to --src")
    parser.add_argument("--roll-deg", type=float, default=DEFAULT_ROLL_DEG,
                        help="rotation about the tool axis; 0 = the assembly's +z towards the link's +x")
    parser.add_argument("--moving-part", help="STL in mm, in the CAD assembly's frame")
    parser.add_argument("--camera-down", action="store_true",
                        help="camera stand turned 180 deg on the flange (tab towards -x of link_eef) and the "
                             "camera remounted so its image stays upright; wrist_camera_link moves with it")
    args = parser.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst) if args.dst else os.path.join(os.path.dirname(src), "xarm5_hand_held.urdf")
    info = build_urdf(src, dst, HAND_HELD_TRAVEL_RAD, roll_deg=args.roll_deg, extra_horn_mesh=args.moving_part,
                      camera_down=args.camera_down)
    print(f"Wrote {dst}")
    print(f"Removed links: {', '.join(info['removed'])}")
    print(f"Kept and moved to {info['parent']}: {', '.join(info['reparented']) or '-'}")
    print(f"Estimated gripper mass: {info['mass_kg']:.3f} kg (not weighed)")


if __name__ == "__main__":
    main()
