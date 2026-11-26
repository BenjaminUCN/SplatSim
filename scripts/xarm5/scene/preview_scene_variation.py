"""Mosaicos para revisar una escena antes de re-renderizar un dataset con ella (24-nov, transferencia sim2real).

    python scripts/xarm5/scene/preview_scene_variation.py ~/data/bc_data/cut_400tr --robot_name <escena> --out <carpeta>

Para algunos episodios grabados, en el primer frame, en el pregrasp del planificador y en el frame de corte
(el primero con peduncle_cut, o el último):
- wrist_<escena>.jpg: wrist_rgb en los tres momentos, una fila por episodio;
- cameras_<escena>.jpg: cada cámara base en el pregrasp, nominal y en las 8 esquinas de la variación
  (signos de x/roll, y/pitch, z/yaw), e imprime el promedio de |nominal - esquina| para ver cuánto se mueve;
- tomato_<escena>.jpg: el tomate girado tomato_turns veces alrededor del tallo, en base_left_rgb y en wrist_rgb
  en el pregrasp.
Comparar dos escenas: correrlo una vez por escena con la misma --out.
"""
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # raíz del repo
from scripts.xarm5.recordings.render_recordings import load_episode, make_server, pregrasp_frame  # noqa: E402

BASE = ["base_left_rgb", "base_rgb", "base_right_rgb"]
TILE = (320, 240)


@dataclass
class Args:
    folder: tyro.conf.Positional[str]
    robot_name: str
    out: str
    episodes: tuple[str, ...] = ("000", "005", "010", "015")
    camera_xyz_m: float = 0.01
    camera_rpy_deg: float = 1.0
    tomato_turns: int = 8


def to_image(tensor) -> Image.Image:
    a = tensor.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def label(image: Image.Image, text: str) -> Image.Image:
    image = image.resize(TILE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 8 * len(text) + 8, 16), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return image


def grid(rows: list[list[Image.Image]]) -> Image.Image:
    width = max(len(r) for r in rows)
    out = Image.new("RGB", (TILE[0] * width, TILE[1] * len(rows)), (40, 40, 40))
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            out.paste(tile, (j * TILE[0], i * TILE[1]))
    return out


def moments(episode: Path, frames: list[dict]) -> dict:
    times = np.array([f["sim_time"] for f in frames])
    meta = json.loads((episode / "meta.json").read_text()) if (episode / "meta.json").exists() else {}
    cut = next((k for k, f in enumerate(frames) if f.get("peduncle_cut")), len(frames) - 1)
    return {"start": 0, "pregrasp": pregrasp_frame(meta, times), "cut": cut}


def main(args: Args) -> None:
    folder = Path(args.folder).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    server = make_server(args.robot_name, BASE + ["wrist_rgb"])
    global TILE
    from splatsim.robots.xarm.cameras import scene_image_size
    width, height = scene_image_size(server)
    TILE = (320, round(320 * height / width))   # la proporción de la escena
    tag = args.robot_name.replace("xarm5_lite_plant_v3_hand_held_down_cam_down_", "")

    wrist_rows, camera_rows, tomato_rows = [], [], []
    corners = list(itertools.product((-1, 1), repeat=3))   # signos de (x, roll), (y, pitch), (z, yaw)
    for name in args.episodes:
        episode = folder / name
        frames = load_episode(episode)
        when = moments(episode, frames)

        row = []
        for moment, k in when.items():
            server.tomato_yaw = 0.0
            server.set_camera_jitter({})
            server.restore_state(frames[k])
            row.append(label(to_image(server.render_cameras(["wrist_rgb"])["wrist_rgb"]), f"{name} {moment}"))
        wrist_rows.append(row)

        server.restore_state(frames[when["pregrasp"]])
        nominal = {c: to_image(img) for c, img in server.render_cameras(BASE).items()}
        for camera in BASE:
            row = [label(nominal[camera], f"{name} {camera} nominal")]
            diffs = []
            for signs in corners:
                xyz = [s * args.camera_xyz_m for s in signs]
                rpy = [s * math.radians(args.camera_rpy_deg) for s in signs]
                server.set_camera_jitter({camera: [*xyz, *rpy]})
                image = to_image(server.render_cameras([camera])[camera])
                diffs.append(np.abs(np.asarray(image, float) - np.asarray(nominal[camera], float)).mean())
                row.append(label(image, "signos " + "".join("+" if v > 0 else "-" for v in signs)))
            server.set_camera_jitter({})
            print(f"{name} {camera}: |nominal - esquina| medio {np.mean(diffs):.1f}/255")
            camera_rows.append(row)

        for camera in ("base_left_rgb", "wrist_rgb"):
            row = []
            for t in range(args.tomato_turns):
                yaw = 2 * math.pi * t / args.tomato_turns
                server.tomato_yaw = yaw
                server.restore_state(frames[when["pregrasp"]])
                row.append(label(to_image(server.render_cameras([camera])[camera]),
                                 f"{name} {camera} {math.degrees(yaw):.0f}deg"))
            tomato_rows.append(row)
        server.tomato_yaw = 0.0

    for kind, rows in (("wrist", wrist_rows), ("cameras", camera_rows), ("tomato", tomato_rows)):
        path = out / f"{kind}_{tag}.jpg"
        grid(rows).save(path, quality=85)
        print(f"→ {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)
