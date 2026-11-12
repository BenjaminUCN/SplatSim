"""Copia un dataset LeRobot (v3) con los videos reescalados, para entrenar más rápido.

    python scripts/xarm5/resize_dataset.py ~/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010 \
        --height 240 --width 320
Escribe <dataset>_<alto>x<ancho> al lado. Reescala cada mp4 con ffmpeg (los mismos frames y tiempos,
mismo códec que LeRobot: AV1, yuv420p, GOP 2) y cambia la forma en meta/info.json. Los parquet, los
episodios y las estadísticas se copian tal cual: las de imagen son por canal y no dependen del tamaño.

Para evaluar un modelo entrenado con estas imágenes sin tocar el agente, el modelo tiene que reescalar
las del sim (480x640): `train_diffusion.py --resize-shape <alto> <ancho>`, o agregar
"resize_shape": [alto, ancho] al config.json del checkpoint si se entrenó sin esa opción.
"""
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    dataset: tyro.conf.Positional[str]   # carpeta del dataset LeRobot
    height: int = 240
    width: int = 320
    out: str | None = None        # None -> <dataset>_<alto>x<ancho>
    crf: int = 30                 # el de LeRobot
    preset: int = 8               # SVT-AV1: más alto = más rápido


def pts(path: Path) -> list[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    return sorted(out.split(), key=float)


def main(a: Args):
    src = Path(a.dataset).expanduser()
    dst = Path(a.out).expanduser() if a.out else src.with_name(f"{src.name}_{a.height}x{a.width}")
    if dst.exists():
        raise SystemExit(f"{dst} ya existe")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))
    videos = sorted((src / "videos").rglob("*.mp4"))
    t0 = time.time()
    for video in videos:
        out = dst / video.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={a.width}:{a.height}:flags=area",
                        "-fps_mode", "passthrough", "-enc_time_base", "demux",
                        "-c:v", "libsvtav1", "-preset", str(a.preset), "-crf", str(a.crf), "-g", "2",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        if pts(video) != pts(out):
            raise SystemExit(f"{out}: los tiempos de los frames no coinciden con el original")
        print(f"{video.relative_to(src)}: {video.stat().st_size / 1e6:.0f} -> {out.stat().st_size / 1e6:.0f} MB")
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, feature in info["features"].items():
        if feature.get("dtype") == "video":
            feature["shape"] = [feature["shape"][0], a.height, a.width]
            feature["info"]["video.height"], feature["info"]["video.width"] = a.height, a.width
    info_path.write_text(json.dumps(info, indent=4))
    print(f"{len(videos)} videos en {time.time() - t0:.0f} s -> {dst}")


if __name__ == "__main__":
    main(tyro.cli(Args))
