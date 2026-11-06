"""Mide cuánto del tomate tapa la escena en cada episodio de una carpeta de grabaciones.

    python scripts/xarm5/measure_occlusion.py ~/data/bc_data/gello_3plants [--frames 8] [--arrangement 2cam]

Por episodio mide --frames instantes repartidos entre el primero y el último, con las poses de cámara
actuales (configs/camera_poses.yaml) y las hojas de --occluders, y escribe en la carpeta (o en --out):

- occlusion.csv: episode, frame, camera, occlusion (%, vacío si el tomate no está en cuadro), level,
  silhouette_px, cut_by_frame.
- occlusion_summary.csv: por episodio y cámara, start (el primer instante con el tomate en cuadro),
  mean, end, level, el nivel de start (en el resto del episodio el brazo también puede tapar), y
  start_cut: el borde de la imagen cortaba al tomate en start, así que el % es sobre lo que quedaba
  en cuadro. En la muñeca el gripper cuenta como oclusión.

Sirve para grabaciones sin imágenes y antiguas. Cuesta 3 renders por cámara e instante.
"""
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_recordings import make_server  # noqa: E402
from splatsim.utils.camera_setup import arrangement_cameras  # noqa: E402
from splatsim.utils.occlusion import LEVELS, OUT_OF_VIEW, occlusion_level  # noqa: E402

ARM_JOINTS = 5
APPLE_KEY = "plastic_apple_position"


@dataclass
class Args:
    folder: tyro.conf.Positional[str]
    """carpeta de grabaciones (episodios 000/, 001/, ...)"""
    frames: int = 8
    """instantes por episodio, repartidos entre el primero y el último"""
    cameras: tuple[str, ...] = ("base_rgb", "base_left_rgb", "base_right_rgb", "wrist_rgb")
    arrangement: str | None = None
    """arreglo de configs/camera_poses.yaml; reemplaza a --cameras"""
    robot_name: str = "xarm5_lite_plant_v3"
    occluders: str | None = None
    """hojas recortadas que se agregan a la escena (configs/occluders.yaml)"""
    episodes: tuple[str, ...] = ()
    """solo estos episodios (por defecto todos)"""
    out: str | None = None
    """carpeta de los .csv (por defecto la de las grabaciones)"""


def spaced_frames(paths: list, count: int) -> list[int]:
    """Índices de count instantes repartidos, con el primero y el último, sin repetir."""
    return sorted({int(i) for i in np.linspace(0, len(paths) - 1, max(count, 1)).round()})


def episode_summary(values: list[float | None], cut: list[bool] | None = None) -> dict:
    """values: % por instante, None si el tomate no estaba en cuadro. cut: si el borde de la imagen
    lo cortaba; start_cut avisa que el nivel es sobre un trozo del tomate, que puede ser muy poco."""
    cut = cut or [False] * len(values)
    seen = [(v, c) for v, c in zip(values, cut) if v is not None]
    if not seen:
        return {"start": None, "mean": None, "end": None, "level": OUT_OF_VIEW, "start_cut": False}
    numbers = [v for v, _ in seen]
    return {"start": numbers[0], "mean": round(float(np.mean(numbers)), 1), "end": numbers[-1],
            "level": occlusion_level(numbers[0]), "start_cut": seen[0][1]}


def restore_frame(server, frame: dict, rest_q: np.ndarray) -> None:
    """Grabaciones actuales: sim_joint_positions. Antiguas: all_joint_positions es la pose del brazo."""
    if "sim_joint_positions" in frame:
        server.restore_state(frame)
        return
    q = rest_q.copy()
    q[1:1 + ARM_JOINTS] = np.asarray(frame["all_joint_positions"])[:ARM_JOINTS]
    server.restore_state({**frame, "sim_joint_positions": q})


def main(args: Args) -> None:
    folder = Path(args.folder).expanduser()
    out = Path(args.out).expanduser() if args.out else folder
    out.mkdir(parents=True, exist_ok=True)
    cameras = arrangement_cameras(args.arrangement) if args.arrangement else list(args.cameras)
    episodes = sorted(e for e in folder.iterdir() if e.is_dir() and e.name.isdigit()
                      and (not args.episodes or e.name in args.episodes))
    if not episodes:
        raise SystemExit(f"No hay episodios en {folder}")

    server = make_server(args.robot_name, cameras, occluders=args.occluders)
    rest_q = np.asarray(server.state_observation()["sim_joint_positions"])
    levels = {cam: dict.fromkeys((*LEVELS, OUT_OF_VIEW), 0) for cam in cameras}
    cut_counts = dict.fromkeys(cameras, 0)
    with open(out / "occlusion.csv", "w", newline="") as detail_file, \
            open(out / "occlusion_summary.csv", "w", newline="") as summary_file:
        detail = csv.DictWriter(detail_file, ["episode", "frame", "camera", "occlusion", "level",
                                              "silhouette_px", "cut_by_frame"])
        summary = csv.DictWriter(summary_file, ["episode", "camera", "start", "mean", "end", "level", "start_cut"])
        detail.writeheader()
        summary.writeheader()
        for n, episode in enumerate(episodes, 1):
            paths = sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem))
            values = {cam: [] for cam in cameras}
            cuts = {cam: [] for cam in cameras}
            measured = 0
            for i in spaced_frames(paths, args.frames) if paths else []:
                try:
                    with open(paths[i], "rb") as f:
                        frame = pickle.load(f)
                except Exception:  # noqa: BLE001  .pkl cortado al final de la grabación
                    print(f"  ⚠ {paths[i].name} ilegible, se omite")
                    continue
                if APPLE_KEY not in frame:
                    break              # robot real: no hay posición del tomate
                restore_frame(server, frame, rest_q)
                for cam, result in server.measure_occlusion(cameras).items():
                    values[cam].append(result["occlusion"])
                    cuts[cam].append(result["cut_by_frame"])
                    detail.writerow({"episode": episode.name, "frame": paths[i].stem, "camera": cam,
                                     **{k: result[k] for k in ("occlusion", "level", "silhouette_px", "cut_by_frame")}})
                measured += 1
            if not measured:
                print(f"[{n}/{len(episodes)}] {episode.name}: sin posición del tomate, se salta")
                continue
            parts = []
            for cam in cameras:
                row = episode_summary(values[cam], cuts[cam])
                summary.writerow({"episode": episode.name, "camera": cam, **row})
                if not row["start_cut"]:
                    levels[cam][row["level"]] += 1
                else:
                    cut_counts[cam] += 1
                parts.append(f"{cam} {row['start'] if row['start'] is not None else '-'}")
            detail_file.flush()
            summary_file.flush()
            print(f"[{n}/{len(episodes)}] {episode.name}: " + ", ".join(parts))
    print(f"\nEpisodios por nivel (al inicio) → {out}/occlusion_summary.csv")
    for cam, counts in levels.items():
        print(f"  {cam}: " + "  ".join(f"{level}%: {c}" if level != OUT_OF_VIEW else f"fuera: {c}"
                                       for level, c in counts.items())
              + f"  cortado por el borde: {cut_counts[cam]}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)  # el servidor deja hilos (teclado, PyBullet) que no terminan solos
