"""Cuadros de una cámara guardados como video H.264 sin pérdida en vez de un PNG por cuadro (~12x menos).

Una carpeta de imágenes (p. ej. <ensayo>/image_1/) con `<cam>_<n>.png` pasa a:
  - `<cam>.mp4`: libx264rgb -qp 0 (RGB sin pérdida; al decodificar da exactamente los mismos píxeles);
  - `videos.json`: {cam: [n, ...]}, los números de los PNG originales, para restaurarlos con su nombre.

pack() verifica cuadro por cuadro (framemd5 de ffmpeg) antes de borrar los PNG; restore() los vuelve a escribir.
VideoFrames se usa como la lista de rutas de PNG en el revisor: frames[k] es una PIL.Image.

Solo se empaquetan cámaras con números consecutivos y PNG RGB de 8 bits (sin alfa); el resto se deja como está.
"""
import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST = "videos.json"
PNG = re.compile(r"^(.+_rgb)_(\d+)\.png$")
ENCODE = ["-c:v", "libx264rgb", "-qp", "0", "-preset", "slow"]


def png_frames(image_dir: Path) -> dict[str, list[tuple[int, str]]]:
    """Cámara -> [(número, nombre de archivo)] de los PNG de la carpeta, en orden."""
    per_cam = {}
    for f in os.scandir(image_dir):
        if m := PNG.match(f.name):
            per_cam.setdefault(m.group(1), []).append((int(m.group(2)), f.name))
    return {cam: sorted(v) for cam, v in sorted(per_cam.items())}


def load_manifest(image_dir: Path) -> dict[str, list[int]]:
    p = Path(image_dir) / MANIFEST
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def _save_manifest(image_dir: Path, manifest: dict):
    tmp = Path(image_dir) / (MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, Path(image_dir) / MANIFEST)


def _pattern(image_dir: Path, cam: str, files: list[tuple[int, str]]) -> list[str]:
    """Argumentos de entrada de ffmpeg para leer los PNG de la cámara en orden."""
    width = len(files[0][1]) - len(cam) - len("_.png")
    return ["-framerate", "10", "-start_number", str(files[0][0]),
            "-i", str(Path(image_dir) / f"{cam}_%0{width}d.png")]


def _framemd5(args: list[str]) -> list[str]:
    out = subprocess.run(["ffmpeg", "-loglevel", "error", *args, "-pix_fmt", "rgb24", "-f", "framemd5", "-"],
                         check=True, capture_output=True, text=True).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def packable(files: list[tuple[int, str]], image_dir: Path) -> str | None:
    """Motivo por el que la cámara no se puede empaquetar sin pérdida (None si se puede)."""
    nums = [n for n, _ in files]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        return "números no consecutivos"
    if len({len(name) for _, name in files}) != 1:
        return "nombres de distinto largo"
    with Image.open(Path(image_dir) / files[0][1]) as im:
        if im.mode != "RGB":
            return f"modo {im.mode}"
    return None


def pack_camera(image_dir: Path, cam: str, files: list[tuple[int, str]], threads: int = 4) -> str | None:
    """Escribe <cam>.mp4, lo verifica contra los PNG, lo anota en videos.json y borra los PNG.
    Devuelve el motivo si no se empaqueta (los PNG quedan intactos)."""
    image_dir = Path(image_dir)
    if why := packable(files, image_dir):
        return why
    src = _pattern(image_dir, cam, files)
    video = image_dir / f"{cam}.mp4"
    tmp = image_dir / f"{cam}.tmp.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *src, *ENCODE, "-threads", str(threads), str(tmp)],
                   check=True)
    if _framemd5(src) != _framemd5(["-i", str(tmp)]):
        tmp.unlink()
        return "el video no coincide con los PNG"
    os.replace(tmp, video)
    manifest = load_manifest(image_dir)
    manifest[cam] = [n for n, _ in files]
    _save_manifest(image_dir, manifest)
    for _, name in files:
        (image_dir / name).unlink()
    return None


def pack(image_dir: Path, threads: int = 4) -> dict[str, str | None]:
    """Empaqueta cada cámara de la carpeta. Reanudable: una cámara ya en videos.json cuyos PNG siguen ahí
    (corte entre escribir el manifiesto y borrar) se vuelve a verificar y se borra. Devuelve {cam: motivo}."""
    image_dir = Path(image_dir)
    done = load_manifest(image_dir)
    result = {}
    for cam, files in png_frames(image_dir).items():
        if cam in done and (image_dir / f"{cam}.mp4").is_file():
            if [n for n, _ in files] == done[cam] and \
                    _framemd5(_pattern(image_dir, cam, files)) == _framemd5(["-i", str(image_dir / f"{cam}.mp4")]):
                for _, name in files:
                    (image_dir / name).unlink()
                result[cam] = None
            else:
                result[cam] = "ya hay video y PNG distintos"
            continue
        result[cam] = pack_camera(image_dir, cam, files, threads)
    return result


def restore(image_dir: Path, width: int = 5):
    """Vuelve a escribir los PNG desde los videos (mismos nombres y píxeles) y borra videos y manifiesto."""
    image_dir = Path(image_dir)
    for cam, nums in load_manifest(image_dir).items():
        video = image_dir / f"{cam}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-start_number", str(nums[0]),
                        str(image_dir / f"{cam}_%0{width}d.png")], check=True)
        video.unlink()
    (image_dir / MANIFEST).unlink(missing_ok=True)


@lru_cache(maxsize=8)
def _decode(path: str, mtime: float) -> tuple[np.ndarray, ...]:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(f[:, :, ::-1]))
    cap.release()
    return tuple(frames)


class VideoFrames:
    """Los cuadros de <cam>.mp4 como secuencia de PIL.Image (se decodifica entero al primer acceso)."""

    def __init__(self, path: Path, count: int):
        self.path = Path(path)
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, k: int) -> Image.Image:
        frames = _decode(str(self.path), self.path.stat().st_mtime)
        return Image.fromarray(frames[k])


def video_cameras(image_dir: Path) -> dict[str, VideoFrames]:
    """Cámara -> VideoFrames de las cámaras empaquetadas de la carpeta."""
    image_dir = Path(image_dir)
    return {cam: VideoFrames(image_dir / f"{cam}.mp4", len(nums))
            for cam, nums in load_manifest(image_dir).items() if (image_dir / f"{cam}.mp4").is_file()}


def open_frame(item) -> Image.Image:
    """Un elemento de la lista de cuadros (ruta de PNG o cuadro de video) como PIL.Image."""
    return item if isinstance(item, Image.Image) else Image.open(item)
