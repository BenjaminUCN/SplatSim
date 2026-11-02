"""Duplica un dataset de grabaciones con su versión espejada y lo renderiza de nuevo.

    python scripts/xarm5/mirror_recordings.py ~/data/bc_data/gello_3plants [--robot-name xarm5_lite_plant_v3]

La salida tiene los episodios originales con sus mismos números (000-205) y a continuación sus
espejos (el espejo de n es n + 206, o en general n + el mayor número de origen + 1), todos
renderizados con las mismas cámaras: así el dataset no mezcla dos encuadres distintos.
--no-with-original deja solo los espejos, con los números de origen.

Refleja cada episodio respecto del plano xz del robot (y -> -y): el tomate pasa a (x, -y, z) y
el brazo hace el movimiento simétrico, que para el xArm5 es exacto cambiando el signo de joint1
(giro de la base) y joint5 (giro de la muñeca); joint2-4 y el gripper no cambian. La pose del
efector (ee_pos_quat, state, action) se recalcula en PyBullet con la pose espejada, y las
cámaras se renderizan de nuevo en vez de voltear las imágenes: la cámara de muñeca está montada
a un lado y la escena no es simétrica, así que cada imagen corresponde al estado espejado.

Sirve para grabaciones antiguas (joint_positions de 5 joints + 1, sin sim_joint_positions) y
actuales. Cada frame se renderiza tal cual, sin remuestrear; la numeración de los .pkl se
conserva y las imágenes quedan por frame (image_1/<cam>_<n>.png), solo en los .png.

La salida (por defecto <carpeta>_mirrored) sirve para review_recordings.py y el convertidor.
Los episodios ya hechos se saltan (--overwrite para rehacerlos); los descartes de selection.yaml
pasan a los originales y a sus espejos, y mirror_info.yaml anota el origen, la escena, las
cámaras y el desplazamiento de los espejos.
"""
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from render_recordings import make_server  # noqa: E402
from splatsim.robots.station.recorder import save_frame  # noqa: E402
from splatsim.utils.camera_setup import arrangement_cameras  # noqa: E402
from splatsim.utils.episode_selection import load_discarded, save_discarded  # noqa: E402
from splatsim.utils.parallel_episodes import own_argv, run_workers  # noqa: E402


@dataclass
class Args:
    folder: tyro.conf.Positional[str]
    """carpeta de grabaciones (episodios 000/, 001/, ...)"""
    out: str | None = None
    """carpeta de salida (por defecto <folder>_mirrored)"""
    robot_name: str = "xarm5_lite_plant_v3"
    """escena con que se renderiza (la de las tres plantas de gello_3plants)"""
    cameras: tuple[str, ...] | None = None
    """cámaras a renderizar (por defecto las que tiene la grabación)"""
    arrangement: str | None = None
    """arreglo de configs/camera_poses.yaml; reemplaza a --cameras"""
    with_original: bool = True
    """incluye también los originales, renderizados sin espejar con las mismas cámaras"""
    episodes: tuple[str, ...] = ()
    """solo estos episodios (por defecto todos)"""
    overwrite: bool = False
    workers: int = 1
    """procesos en paralelo, cada uno con su parte de los episodios y su escena en la GPU
    (~8 GB cada uno: en una GPU de 16 GB cabe solo uno)"""


MIRROR_INFO_FILE = "mirror_info.yaml"
ARM_JOINTS = 5
MIRRORED_JOINTS = (0, 4)   # joint1 y joint5 en joint_positions (sin la articulación fija 0)


def mirror_joints(q, offset: int = 0) -> np.ndarray:
    """Cambia el signo de joint1 y joint5; offset=1 si el vector incluye la articulación fija 0."""
    q = np.array(q, dtype=float)
    for i in MIRRORED_JOINTS:
        if i + offset < len(q):
            q[i + offset] *= -1
    return q


def mirror_position(p) -> tuple:
    return (p[0], -p[1], p[2])


def mirror_quaternion(q) -> tuple:
    """Rotación reflejada por y -> -y (M R M con M = diag(1, -1, 1)), cuaternión xyzw."""
    return (-q[0], q[1], -q[2], q[3])


def rendered_obs(server, frame: dict, base_q: np.ndarray, cameras: list[str], mirror: bool) -> dict:
    """El frame (espejado si mirror) con la pose del efector recalculada y las cámaras renderizadas."""
    obs = {k: v for k, v in frame.items() if not k.endswith("_rgb")}
    flip = mirror_joints if mirror else (lambda q, offset=0: np.array(q, dtype=float))
    if "sim_joint_positions" in frame:
        q = flip(frame["sim_joint_positions"], offset=1)
    else:   # grabación antigua: solo los joints del brazo, el resto queda como en el reset;
        # all_joint_positions es la pose de PyBullet (joint_positions a veces es el comando)
        q = base_q.copy()
        q[1:1 + ARM_JOINTS] = flip(frame["all_joint_positions"])[:ARM_JOINTS]
    if mirror:
        for key in ("joint_positions", "all_joint_positions", "joint_positions_dummy", "joint_command"):
            if key in frame:
                obs[key] = mirror_joints(frame[key])
        if "joint_velocities" in frame:
            obs["joint_velocities"] = mirror_joints(frame["joint_velocities"], offset=1)
        for key in frame:
            if key.endswith("_position") and key != "gripper_position":
                name = key[:-len("_position")]
                obs[key] = mirror_position(frame[key])
                if f"{name}_orientation" in frame:
                    obs[f"{name}_orientation"] = mirror_quaternion(frame[f"{name}_orientation"])

    server.restore_state({**obs, "sim_joint_positions": q})
    fresh = server.state_observation()
    # pose del efector recalculada; el gripper (estado y comando) es el grabado
    pose = fresh["state"][:-1]
    obs["state"] = np.concatenate([pose, np.asarray(frame["state"])[-1:]])
    obs["action"] = np.concatenate([pose, np.asarray(frame["action"])[-1:]])
    obs["ee_pos_quat"] = fresh["ee_pos_quat"]
    if "sim_joint_positions" in frame:
        obs["sim_joint_positions"] = q
    obs["mirrored"] = mirror
    obs.update({name: img.cpu() for name, img in server.render_cameras(cameras).items()})
    return obs


def read_frame(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001  .pkl cortado al final de la grabación
        print(f"  ⚠ {path.name} ilegible, se omite")
        return None


def main(args: Args):
    folder = Path(args.folder).expanduser()
    out_root = Path(args.out).expanduser() if args.out else folder.with_name(f"{folder.name}_mirrored")
    episodes = sorted(e for e in folder.iterdir() if e.is_dir() and e.name.isdigit()
                      and (not args.episodes or e.name in args.episodes))
    if not episodes:
        raise SystemExit(f"No hay episodios en {folder}")
    out_root.mkdir(parents=True, exist_ok=True)
    # el espejo de n va en n + offset, después del mayor número de origen
    offset = max(int(e.name) for e in folder.iterdir() if e.is_dir() and e.name.isdigit()) + 1 \
        if args.with_original else 0
    mirror_name = lambda name: f"{int(name) + offset:0{len(name)}d}"  # noqa: E731
    discarded = load_discarded(folder)
    if discarded:
        save_discarded(out_root, {mirror_name(d) for d in discarded}
                       | (discarded if args.with_original else set()))
    jobs = [(e, False, e.name) for e in episodes] if args.with_original else []
    jobs += [(e, True, mirror_name(e.name)) for e in episodes]
    if args.workers > 1:   # cada proceso hace sus originales y sus espejos
        sys.exit(run_workers(own_argv(sys.argv), [e.name for e in episodes], args.workers,
                             total=len(jobs)))

    server = cameras = base_q = None
    for n, (episode, mirror, name) in enumerate(jobs, 1):
        label = f"[{n}/{len(jobs)}] {episode.name}{f' espejado → {name}' if mirror else ''}"
        target = out_root / name
        if target.exists() and not args.overwrite:
            print(f"{label}: ya existe, se salta")
            continue
        paths = sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem))
        first = next((f for f in map(read_frame, paths[:3]) if f is not None), None)
        if first is None:
            print(f"{label}: sin frames, se salta")
            continue
        if server is None:
            if args.arrangement:
                cameras = arrangement_cameras(args.arrangement)
            elif args.cameras:
                cameras = list(args.cameras)
            else:
                cameras = sorted({k for k in first if k.endswith("_rgb")}) or ["base_rgb", "wrist_rgb"]
            with open(out_root / MIRROR_INFO_FILE, "w") as f:
                yaml.safe_dump({"source": str(folder.resolve()), "robot_name": args.robot_name,
                                "cameras": cameras, "mirror": "y -> -y (joint1, joint5)",
                                "with_original": args.with_original,
                                "mirror_offset": offset,   # el espejo del episodio n es n + offset
                                "rendered": datetime.now().strftime("%Y-%m-%d %H:%M")},
                               f, sort_keys=False)
            print(f"{len(episodes)} episodios ({len(jobs)} con espejos) → {out_root}, "
                  f"escena {args.robot_name}, cámaras {cameras}")
            server = make_server(args.robot_name, cameras)
            server.reset_robot()
            base_q = np.asarray(server.state_observation()["sim_joint_positions"])

        start = time.time()
        tmp = out_root / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        if (episode / "meta.json").exists():
            shutil.copy(episode / "meta.json", tmp / "meta.json")
        count = 0
        for path in paths:
            if (frame := read_frame(path)) is None:
                continue
            save_frame(str(tmp), int(path.stem), rendered_obs(server, frame, base_q, cameras, mirror),
                       images_in_pkl=False)
            count += 1
        shutil.rmtree(target, ignore_errors=True)
        os.rename(tmp, target)
        print(f"{label}: {count} frames, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)  # el servidor deja hilos (teclado, PyBullet) que no terminan solos
