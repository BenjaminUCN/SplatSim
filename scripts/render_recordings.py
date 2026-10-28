"""Renderiza las imágenes de grabaciones hechas con sim_clock=realtime (solo estado).

    python scripts/render_recordings.py ~/data/bc_data/gello_teleop --fps 10 [--arrangement 2cam]

Al teleoperar con sim_clock=realtime el servidor graba un frame por comando (~30 Hz) con el
estado, sim_time y las posiciones de todas las articulaciones, sin imágenes. Este script
remuestrea cada episodio por sim_time a un fps fijo (interpola las articulaciones al instante
exacto k/fps y recalcula el estado desde esa pose; el comando del gripper sale del frame más
cercano) y renderiza esos frames con las cámaras pedidas, en el formato del recorder
(<salida>/<episodio>/<n>.pkl + image_1/<cam>_<n>.png). Así el dataset tiene un dt exacto, el
mismo con que se evalúa la política (sim_clock=fixed, --hz igual a --fps), y se puede volver a
renderizar con otras cámaras sin grabar de nuevo.

La salida (por defecto <carpeta>_<fps>hz) sirve para review_recordings.py y el convertidor
(lerobot_transform_v2.py --fps igual al de aquí). Los episodios ya renderizados se saltan
(--overwrite para rehacerlos); selection.yaml se copia. render_info.yaml anota la carpeta de
origen, el fps y las cámaras: con eso review_recordings.py superpone la trayectoria original
(~30 Hz) a la remuestreada.
"""
import os
import pickle
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splatsim.robots.station.recorder import save_frame  # noqa: E402
from splatsim.utils.camera_setup import arrangement_cameras  # noqa: E402
from splatsim.utils.episode_selection import SELECTION_FILE  # noqa: E402


@dataclass
class Args:
    folder: tyro.conf.Positional[str]
    """carpeta de grabaciones con sim_clock=realtime (episodios 000/, 001/, ...)"""
    fps: int = 10
    cameras: tuple[str, ...] = ("base_rgb", "base_left_rgb", "base_right_rgb", "wrist_rgb")
    arrangement: str | None = None
    """arreglo de configs/camera_poses.yaml; reemplaza a --cameras"""
    out: str | None = None
    """carpeta de salida (por defecto <folder>_<fps>hz)"""
    robot_name: str = "xarm5_lite_plant_v3"
    episodes: tuple[str, ...] = ()
    """solo estos episodios (por defecto todos)"""
    overwrite: bool = False


RENDER_INFO_FILE = "render_info.yaml"

# campos que se recalculan desde la pose interpolada (el resto viene del frame más cercano)
RECOMPUTED = ("joint_positions", "all_joint_positions", "joint_positions_dummy", "ee_pos_quat",
              "state", "action", "gripper_position", "sim_joint_positions")


def resample(times: np.ndarray, fps: int) -> list[tuple[float, int, int, float]]:
    """Instantes t0 + k/fps hasta el último frame, cada uno como (t, i, i+1, peso de i+1)."""
    out = []
    for t in np.arange(times[0], times[-1] + 1e-9, 1 / fps):
        j = int(np.searchsorted(times, t).clip(1, len(times) - 1))
        i = j - 1
        w = float(np.clip((t - times[i]) / max(times[j] - times[i], 1e-9), 0, 1))
        out.append((float(t), i, j, w))
    return out


def interpolated_obs(server, frames: list[dict], t: float, i: int, j: int, w: float) -> dict:
    """Observación en el instante t: pose interpolada entre los frames i y j, estado recalculado."""
    near = frames[j] if w >= 0.5 else frames[i]
    q = (1 - w) * np.asarray(frames[i]["sim_joint_positions"]) + w * np.asarray(frames[j]["sim_joint_positions"])
    server.restore_state({**near, "sim_joint_positions": q})
    server.current_gripper_action = near["action"][-1]   # comando del gripper: discreto
    fresh = server.state_observation()
    obs = dict(near)
    obs.update({k: fresh[k] for k in RECOMPUTED if k in fresh})
    obs["sim_time"] = t
    obs["source_frames"] = (i, j, round(w, 3))
    return obs


def load_episode(episode: Path) -> list[dict]:
    frames = []
    for p in sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem)):
        try:
            with open(p, "rb") as f:
                frames.append(pickle.load(f))
        except Exception:  # noqa: BLE001  .pkl cortado al final de la grabación
            print(f"  ⚠ {p.name} ilegible, se omite")
    return frames


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_server(robot_name: str, cameras: list[str]):
    """Servidor xArm5 en este proceso, sin ZMQ ni serve_loop: solo para renderizar."""
    from splatsim.robots.xarm.server import XArm5PybulletRobotServerCamera
    with open("configs/object_configs/objects.yaml") as f:
        use_gripper = yaml.safe_load(f)[robot_name].get("use_gripper", True)
    server = XArm5PybulletRobotServerCamera(
        port=free_port(), host="127.0.0.1", camera_names=cameras, robot_name=robot_name,
        cam_i=347, use_gripper=use_gripper, gui="pybullet", show_apple_window=False)
    server.setup_scene()
    return server


def main(args: Args):
    folder = Path(args.folder).expanduser()
    out_root = Path(args.out).expanduser() if args.out else folder.with_name(f"{folder.name}_{args.fps}hz")
    cameras = arrangement_cameras(args.arrangement) if args.arrangement else list(args.cameras)
    episodes = sorted(e for e in folder.iterdir() if e.is_dir() and e.name.isdigit()
                      and (not args.episodes or e.name in args.episodes))
    if not episodes:
        raise SystemExit(f"No hay episodios en {folder}")
    out_root.mkdir(parents=True, exist_ok=True)
    if (folder / SELECTION_FILE).exists():
        shutil.copy(folder / SELECTION_FILE, out_root / SELECTION_FILE)
    with open(out_root / RENDER_INFO_FILE, "w") as f:
        yaml.safe_dump({"source": str(folder.resolve()), "fps": args.fps, "cameras": cameras,
                        "rendered": datetime.now().strftime("%Y-%m-%d %H:%M")}, f, sort_keys=False)
    print(f"{len(episodes)} episodios → {out_root} a {args.fps} fps, cámaras {cameras}")

    server = None
    for n, episode in enumerate(episodes, 1):
        target = out_root / episode.name
        if target.exists() and not args.overwrite:
            print(f"[{n}/{len(episodes)}] {episode.name}: ya existe, se salta")
            continue
        frames = load_episode(episode)
        if not frames or "sim_joint_positions" not in frames[0]:
            print(f"[{n}/{len(episodes)}] {episode.name}: sin sim_joint_positions "
                  "(no es una grabación con sim_clock=realtime), se salta")
            continue
        times = np.array([f["sim_time"] for f in frames])
        points = resample(times, args.fps)
        if server is None:
            server = make_server(args.robot_name, cameras)

        start = time.time()
        tmp = out_root / f".{episode.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        if (episode / "meta.json").exists():
            shutil.copy(episode / "meta.json", tmp / "meta.json")
        for k, point in enumerate(points, 1):
            obs = interpolated_obs(server, frames, *point)
            images = server.render_cameras(cameras)
            obs.update({name: img.cpu() for name, img in images.items()})
            save_frame(str(tmp), k, obs)
        shutil.rmtree(target, ignore_errors=True)
        os.rename(tmp, target)
        gaps = np.diff(times)
        print(f"[{n}/{len(episodes)}] {episode.name}: {len(frames)} frames grabados "
              f"({times[-1] - times[0]:.1f} s, dt máx {gaps.max() * 1000:.0f} ms) → "
              f"{len(points)} a {args.fps} fps, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)  # el servidor deja hilos (teclado, PyBullet) que no terminan solos
