"""Renderiza las imágenes de grabaciones hechas con sim_clock=realtime (solo estado).

    python scripts/xarm5/render_recordings.py ~/data/bc_data/gello_teleop --fps 10 [--arrangement 2cam]

Al teleoperar con sim_clock=realtime el servidor graba un frame por comando (~30 Hz) con el
estado, sim_time y las posiciones de todas las articulaciones, sin imágenes. Este script
remuestrea cada episodio por sim_time a un fps fijo (interpola las articulaciones al instante
exacto k/fps y recalcula el estado desde esa pose; el comando del gripper sale del frame más
cercano) y renderiza esos frames con las cámaras pedidas, en el formato del recorder
(<salida>/<episodio>/<n>.pkl + image_1/<cam>_<n>.png). Así el dataset tiene un dt exacto, el
mismo con que se evalúa la política (sim_clock=fixed, --hz igual a --fps), y se puede volver a
renderizar con otras cámaras sin grabar de nuevo.

La salida (por defecto <carpeta>_<fps>hz) sirve para review_recordings.py y el convertidor
(lerobot_conversion.py --fps igual al de aquí). Los episodios ya renderizados se saltan
(--overwrite para rehacerlos); selection.yaml se copia. render_info.yaml anota la carpeta de
origen, el fps y las cámaras: con eso review_recordings.py superpone la trayectoria original
(~30 Hz) a la remuestreada.
"""
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from splatsim.robots.station.recorder import save_frame  # noqa: E402
from splatsim.utils.camera_setup import arrangement_cameras  # noqa: E402
from splatsim.utils.episode_selection import SELECTION_FILE  # noqa: E402
from splatsim.utils.leaf_segmentation import load_occluder_config  # noqa: E402
from splatsim.utils.leaf_variation import (  # noqa: E402
    DEFAULT_TARGETS,
    arm_path,
    draw_episode,
    place_episode_leaf,
)
from splatsim.utils.parallel_episodes import own_argv, run_workers  # noqa: E402


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
    occluders: str | None = None
    """hojas recortadas del splat que se agregan a todos los episodios (configs/occluders.yaml)"""
    leaf_dir: str | None = None
    """carpeta de hojas recortadas: cada episodio lleva una al azar, ajustada a un objetivo de
    --occlusion_targets (~/data/occluders/<escena>). Se anota en el meta.json de cada episodio"""
    occlusion_targets: tuple[float, ...] = DEFAULT_TARGETS
    """porcentajes objetivo que se sortean por episodio; 0 = sin hoja"""
    leaf_cameras: tuple[str, ...] = ()
    """cámaras en que se ajusta la hoja (por defecto las que no son de muñeca)"""
    seed: int = 0
    """semilla del sorteo de hojas"""
    episodes: tuple[str, ...] = ()
    """solo estos episodios (por defecto todos)"""
    overwrite: bool = False
    workers: int = 1
    """procesos en paralelo, cada uno con su parte de los episodios y su escena en la GPU
    (~8 GB cada uno: en una GPU de 16 GB cabe solo uno)"""


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


def make_server(robot_name: str, cameras: list[str], **kwargs):
    """Servidor xArm5 en este proceso, sin ZMQ ni serve_loop: solo para renderizar (o, sin
    cámaras, para simular sin GPU). kwargs pasa al servidor, p.ej. control_hz."""
    from splatsim.robots.xarm.server import XArm5PybulletRobotServerCamera
    with open("configs/object_configs/objects.yaml") as f:
        use_gripper = yaml.safe_load(f)[robot_name].get("use_gripper", True)
    server = XArm5PybulletRobotServerCamera(
        port=free_port(), host="127.0.0.1", camera_names=cameras, robot_name=robot_name,
        cam_i=347, use_gripper=use_gripper, gui="pybullet", show_apple_window=False, **kwargs)
    server.setup_scene()
    return server


def portable_path(path: Path) -> Path:
    """Con ~: el meta.json de un render en un pod también debe servir en el lab."""
    return Path("~") / path.relative_to(Path.home()) if path.is_relative_to(Path.home()) else path


def write_leaf_meta(path: Path, entries: list[dict], leaf_fit: dict) -> None:
    """Agrega la hoja al meta.json del episodio (measure_occlusion.py la vuelve a poner desde ahí)."""
    meta = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**meta, "occluders": entries, "leaf_fit": leaf_fit}, indent=2))


def main(args: Args):
    folder = Path(args.folder).expanduser()
    out_root = Path(args.out).expanduser() if args.out else folder.with_name(f"{folder.name}_{args.fps}hz")
    cameras = arrangement_cameras(args.arrangement) if args.arrangement else list(args.cameras)
    episodes = sorted(e for e in folder.iterdir() if e.is_dir() and e.name.isdigit()
                      and (not args.episodes or e.name in args.episodes))
    if not episodes:
        raise SystemExit(f"No hay episodios en {folder}")
    leaves = [portable_path(p) for p in sorted(Path(args.leaf_dir).expanduser().glob("*.ply"))] if args.leaf_dir else []
    if args.leaf_dir and not leaves:
        raise SystemExit(f"No hay hojas en {args.leaf_dir}: recorta alguna con segment_leaf.py")
    leaf_cameras = list(args.leaf_cameras) or [c for c in cameras if "wrist" not in c]
    out_root.mkdir(parents=True, exist_ok=True)
    if (folder / SELECTION_FILE).exists():
        shutil.copy(folder / SELECTION_FILE, out_root / SELECTION_FILE)
    with open(out_root / RENDER_INFO_FILE, "w") as f:
        yaml.safe_dump({"source": str(folder.resolve()), "fps": args.fps, "cameras": cameras,
                        "occluders": load_occluder_config(args.occluders) if args.occluders else [],
                        "leaf_variation": {"leaves": [str(p) for p in leaves],
                                           "targets": list(args.occlusion_targets),
                                           "cameras": leaf_cameras, "seed": args.seed} if leaves else None,
                        "rendered": datetime.now().strftime("%Y-%m-%d %H:%M")}, f, sort_keys=False)
    print(f"{len(episodes)} episodios → {out_root} a {args.fps} fps, cámaras {cameras}")
    if args.workers > 1:
        sys.exit(run_workers(own_argv(sys.argv), [e.name for e in episodes], args.workers,
                             total=len(episodes)))

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
            server = make_server(args.robot_name, cameras, occluders=args.occluders)

        start = time.time()
        tmp = out_root / f".{episode.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        if (episode / "meta.json").exists():
            shutil.copy(episode / "meta.json", tmp / "meta.json")
        entries, leaf_fit = [], None
        if leaves:
            path = arm_path(server, frames)
            server.restore_state(frames[0])
            draw = draw_episode(episode.name, leaves, args.occlusion_targets, args.seed)
            entries, leaf_fit = place_episode_leaf(server, draw, leaf_cameras, path)
        for k, point in enumerate(points, 1):
            obs = interpolated_obs(server, frames, *point)
            images = server.render_cameras(cameras)
            obs.update({name: img.cpu() for name, img in images.items()})
            save_frame(str(tmp), k, obs, images_in_pkl=False)   # 14.7 MB por frame si no
        if leaf_fit:
            write_leaf_meta(tmp / "meta.json", entries, leaf_fit)
        shutil.rmtree(target, ignore_errors=True)
        os.rename(tmp, target)
        gaps = np.diff(times)
        leaf_note = ""
        if leaf_fit:
            leaf_note = f" | objetivo {leaf_fit['target']:.0f}%: {leaf_fit['status']}"
            if entries:
                leaf_note += f", {leaf_fit['leaf']} en {leaf_fit['camera']} → {leaf_fit['occlusion']:.0f}%"
        print(f"[{n}/{len(episodes)}] {episode.name}: {len(frames)} frames grabados "
              f"({times[-1] - times[0]:.1f} s, dt máx {gaps.max() * 1000:.0f} ms) → "
              f"{len(points)} a {args.fps} fps, {time.time() - start:.0f} s{leaf_note}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)  # el servidor deja hilos (teclado, PyBullet) que no terminan solos
