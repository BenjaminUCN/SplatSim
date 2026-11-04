"""Prueba y calibra el gripper XL330 + OpenRB-150, sin el brazo.

    python scripts/xarm5/xl330_gripper.py status
    python scripts/xarm5/xl330_gripper.py calibrate     # guarda open/closed en configs/xarm5_real.yaml
    python scripts/xarm5/xl330_gripper.py move 0.5      # 0 = abierto, 1 = cerrado
    python scripts/xarm5/xl330_gripper.py open | close | torque-off | reboot

calibrate, torque-off y el final de open/close/move dejan el servo sin torque.
Haz las primeras pruebas sin la hoja de corte.
"""
import argparse
import re
import time

import yaml

from splatsim.robots.xl330_gripper import ADDR_TORQUE_ENABLE, XL330Gripper

DEFAULT_CONFIG = "configs/xarm5_real.yaml"
MOVE_TIMEOUT = 6.0     # s para llegar antes de darlo por terminado


def save_calibration(path: str, open_position: int, closed_position: int) -> None:
    """Cambia solo esas dos líneas, para no perder los comentarios del archivo."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for key, value in (("open_position", open_position), ("closed_position", closed_position)):
        text, n = re.subn(rf"(?m)^(\s+{key}:)[^\S\r\n]*[^\s#]*", rf"\g<1> {value}", text)
        if n != 1:
            raise SystemExit(f"{path}: expected one '{key}:' line, found {n}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def calibrate(gripper: XL330Gripper, path: str) -> None:
    gripper.connect()
    gripper._write(ADDR_TORQUE_ENABLE, 1, 0)
    input("Torque off. Move the gripper by hand to fully OPEN and press Enter... ")
    opened = gripper.read_raw_position()
    input("Now move it to fully CLOSED and press Enter... ")
    closed = gripper.read_raw_position()
    span = abs(closed - opened)
    print(f"open = {opened}, closed = {closed}: travel {span} pulses ({span * 360 / 4096:.1f} deg)")
    if not 50 <= span <= 3596:
        raise SystemExit("That travel is not usable (50-3596 pulses): nothing saved")
    save_calibration(path, opened, closed)
    print(f"Saved in {path}. Set the same travel in HAND_HELD_TRAVEL_RAD if it differs from the sim's.")


def move(gripper: XL330Gripper, value: float) -> None:
    gripper.start()
    if gripper.problem:
        raise SystemExit(gripper.problem)
    gripper.command(value)
    start = time.monotonic()
    while time.monotonic() - start < MOVE_TIMEOUT:
        time.sleep(0.2)
        print(f"\r{gripper.status_text()}    ", end="", flush=True)
        if gripper.problem or gripper.stalled or abs(gripper.get_state() - value) < 0.01:
            break
    print()
    gripper.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("action", choices=["status", "calibrate", "open", "close", "move", "torque-off", "reboot"])
    parser.add_argument("value", nargs="?", type=float, help="for move: 0 = open, 1 = closed")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--port", help="default: gripper.port in the config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = dict((yaml.safe_load(f) or {}).get("gripper") or {})
    if args.port:
        cfg["port"] = args.port
    gripper = XL330Gripper.from_config(cfg)

    if args.action == "calibrate":
        calibrate(gripper, args.config)
    elif args.action in ("open", "close", "move"):
        if args.action == "move" and args.value is None:
            parser.error("move needs a value between 0 and 1")
        move(gripper, {"open": 0.0, "close": 1.0}.get(args.action, args.value))
    else:
        gripper.connect()
        if args.action == "torque-off":
            gripper._write(ADDR_TORQUE_ENABLE, 1, 0)
        elif args.action == "reboot":
            gripper._reboot()
        print(f"port {gripper.port}, ID {gripper.dxl_id}: position {gripper.read_raw_position()}, "
              f"mode {gripper._read(11, 1)}, torque {gripper._read(ADDR_TORQUE_ENABLE, 1)}, "
              f"hardware error {gripper._read(70, 1)}, input {gripper._read(144, 2) / 10:.1f} V; "
              f"calibrated: open {gripper.open_position}, closed {gripper.closed_position}")


if __name__ == "__main__":
    main()
