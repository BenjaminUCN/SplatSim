"""Imágenes de un render tomadas del dataset LeRobot convertido desde él (25-nov, espacio en disco).

Un render de scripts/xarm5/recordings/render_recordings.py guarda un PNG por cámara y cuadro: ~80 GB para una escena de 383
episodios. El dataset de lerobot_conversion.py trae las mismas imágenes en AV1 (~2.7 GB).

prune() borra los PNG de un render cuyo dataset está completo y verificado, y deja <render>/dataset_images.yaml, que
dice de qué dataset y episodio sale cada uno. Con ese archivo, scripts/xarm5/recordings/review_recordings.py muestra los cuadros
del dataset. Ojo: son AV1, con pérdida, no los PNG exactos.

Los .pkl (el estado por cuadro, ~1 KB cada uno) y meta.json se quedan, así que trayectorias, selección y etiquetas
siguen funcionando. Los episodios descartados en selection.yaml no están en el dataset y conservan sus PNG.
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

POINTER = "dataset_images.yaml"
IMAGE_DIRS = ("image_1", "images")
PNG = re.compile(r"^(.+_rgb)_(\d+)\.png$")
KEY = "observation.images."
MIN_PSNR_DB = 33.0   # AV1 de lerobot_conversion: 38-42 dB contra su PNG (cut_400tr); otro episodio da 12-24 dB en las
                     # cámaras que ven moverse al robot


# ── índice del dataset ───────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _episodes(dataset: str):
    import pandas as pd
    files = sorted((Path(dataset) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    cols = None
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if cols is None:
            cols = ["episode_index", "length"] + [c for c in df.columns if c.startswith("videos/")]
        frames.append(df[cols])
    return pd.concat(frames).set_index("episode_index").sort_index()


def _info(dataset: Path) -> dict:
    import json
    return json.loads((Path(dataset) / "meta" / "info.json").read_text())


def dataset_cameras_of(dataset: Path) -> list[str]:
    """Cámaras con video en el dataset (sin el prefijo observation.images.)."""
    feats = _info(dataset)["features"]
    return [k[len(KEY):] for k, v in feats.items() if k.startswith(KEY) and v.get("dtype") == "video"]


def find_dataset(render: Path, roots: list[Path]) -> tuple[Path, dict[str, int]] | None:
    """El dataset convertido desde este render, y {episodio del render: episodio del dataset}. Si hay varios, el más
    reciente con DONE (los datasets de antes del 17-nov no lo tienen: check() verifica igual que estén completos)."""
    render = Path(render)
    found = []
    for root in roots:
        for src in Path(root).expanduser().glob("*/meta/splatsim_sources.yaml"):
            dataset = src.parent.parent
            for s in (yaml.safe_load(src.read_text()) or {}).get("sources", []):
                if Path(s["folder"]).name == render.name:
                    mapping = {str(e): int(i) for e, i in zip(s["episodes"], s["dataset_episodes"])}
                    found.append(((dataset / "DONE").exists(), src.stat().st_mtime, dataset, mapping))
    if not found:
        return None
    _, _, dataset, mapping = max(found, key=lambda f: f[:2])
    return dataset, mapping


# ── cuadros ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def _decode(path: str, first: int, count: int, fps: float) -> tuple[np.ndarray, ...]:
    """Los cuadros first..first+count-1 (por índice = round(t * fps)) de un video del dataset."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(first / fps / stream.time_base), stream=stream, backward=True, any_frame=False)
        out = {}
        for frame in container.decode(stream):
            k = round(frame.time * fps)
            if k >= first + count:
                break
            if k >= first:
                out[k] = frame.to_ndarray(format="rgb24")
    if len(out) != count:
        raise RuntimeError(f"{path}: {len(out)} de {count} cuadros desde el {first}")
    return tuple(out[first + i] for i in range(count))


class DatasetFrames:
    """Los cuadros de un episodio y una cámara del dataset como secuencia de PIL.Image (como VideoFrames de
    splatsim/utils/data/frame_video.py): se decodifica el episodio entero al primer acceso."""

    def __init__(self, dataset: Path, episode: int, camera: str):
        row = _episodes(str(dataset)).loc[episode]
        info = _info(dataset)
        key = KEY + camera
        self.fps = float(info["fps"])
        self.path = Path(dataset) / info["video_path"].format(
            video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        self.first = round(float(row[f"videos/{key}/from_timestamp"]) * self.fps)
        self.count = int(row["length"])

    def __len__(self):
        return self.count

    def array(self, k: int) -> np.ndarray:
        return _decode(str(self.path), self.first, self.count, self.fps)[k]

    def __getitem__(self, k: int) -> Image.Image:
        if k < 0:
            k += self.count
        if not 0 <= k < self.count:
            raise IndexError(k)
        return Image.fromarray(self.array(k))


def load_pointer(render: Path) -> dict | None:
    p = Path(render) / POINTER
    return yaml.safe_load(p.read_text()) if p.is_file() else None


def dataset_cameras(render: Path, episode: str) -> dict[str, DatasetFrames]:
    """Cámara -> DatasetFrames del episodio, si el render tiene dataset_images.yaml y el episodio está en él."""
    pointer = load_pointer(render)
    if not pointer or str(episode) not in pointer["episodes"] or not Path(pointer["dataset"]).is_dir():
        return {}
    index = pointer["episodes"][str(episode)]
    return {cam: DatasetFrames(Path(pointer["dataset"]), index, cam) for cam in pointer["cameras"]}


# ── verificar y borrar ───────────────────────────────────────────────────────
def _image_dir(episode: Path) -> Path | None:
    return next((episode / d for d in IMAGE_DIRS if (episode / d).is_dir()), None)


def png_files(episode: Path) -> dict[str, list[Path]]:
    """Cámara -> PNG del episodio en orden."""
    image_dir = _image_dir(episode)
    per_cam = {}
    if image_dir is not None:
        for f in os.scandir(image_dir):
            if m := PNG.match(f.name):
                per_cam.setdefault(m.group(1), []).append((int(m.group(2)), Path(f.path)))
    return {cam: [p for _, p in sorted(v)] for cam, v in per_cam.items()}


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def check(render: Path, dataset: Path, mapping: dict[str, int], samples: int = 5) -> list[str]:
    """Problemas que impiden borrar los PNG (lista vacía si el dataset tiene todo):
    - cada episodio del mapeo con tantos cuadros en el dataset como .pkl en el render, y cada cámara del dataset
      con un PNG por cuadro (o ya sin PNG, si se borraron antes);
    - en `samples` episodios repartidos, 6 cuadros de cada cámara parecidos a su PNG (PSNR >= MIN_PSNR_DB), y en
      total más parecidos a su PNG que los cuadros vecinos (±1): descarta un dataset de otro render con los mismos
      nombres de episodio y cuadros corridos (a 10 Hz un cuadro de diferencia apenas baja el PSNR de cada uno)."""
    render, dataset = Path(render), Path(dataset)
    problems = []
    cameras = dataset_cameras_of(dataset)
    episodes = _episodes(str(dataset))
    names = sorted(mapping)
    if len(set(mapping.values())) != len(mapping):
        problems.append("episodios del dataset repetidos en el mapeo")
    for name in names:
        ep = render / name
        if not ep.is_dir():
            problems.append(f"{name}: no está en el render")
            continue
        if mapping[name] not in episodes.index:
            problems.append(f"{name}: el episodio {mapping[name]} no está en el dataset")
            continue
        n_pkl = sum(1 for _ in ep.glob("*.pkl"))
        length = int(episodes.loc[mapping[name], "length"])
        if n_pkl != length:
            problems.append(f"{name}: {n_pkl} .pkl y {length} cuadros en el dataset")
        pngs = png_files(ep)
        for cam in cameras:
            if cam in pngs and len(pngs[cam]) != length:
                problems.append(f"{name} {cam}: {len(pngs[cam])} PNG y {length} cuadros")
    if problems:
        return problems
    with_png = [n for n in names if png_files(render / n)]
    picks = [with_png[i] for i in np.linspace(0, len(with_png) - 1, min(samples, len(with_png))).round().astype(int)] \
        if with_png else []
    for name in dict.fromkeys(picks):
        pngs = png_files(render / name)
        score = {-1: 0.0, 0: 0.0, 1: 0.0}
        for cam in cameras:
            if cam not in pngs:
                continue
            frames = DatasetFrames(dataset, mapping[name], cam)
            for k in np.linspace(1, len(frames) - 2, 6).round().astype(int):
                with Image.open(pngs[cam][k]) as im:
                    png = np.asarray(im.convert("RGB"))
                if frames.array(k).shape != png.shape:
                    problems.append(f"{name} {cam} cuadro {k}: {frames.array(k).shape} en el dataset y "
                                    f"{png.shape} en el PNG")
                    break
                for d in score:
                    score[d] += _psnr(frames.array(k + d), png)
                if (db := _psnr(frames.array(k), png)) < MIN_PSNR_DB:
                    problems.append(f"{name} {cam} cuadro {k}: PSNR {db:.1f} dB < {MIN_PSNR_DB}")
        if score[0] < max(score[-1], score[1]):
            problems.append(f"{name}: los cuadros del dataset se parecen más a los PNG vecinos que a los suyos "
                            f"(PSNR total {score[-1]:.0f}/{score[0]:.0f}/{score[1]:.0f} con -1/0/+1)")
    return problems


def prune(render: Path, dataset: Path, mapping: dict[str, int], delete: bool = False) -> tuple[int, int, list[str]]:
    """Escribe dataset_images.yaml y borra los PNG de las cámaras del dataset en los episodios del mapeo.
    Devuelve (bytes borrados o por borrar, episodios con PNG que se conservan, cámaras que no están en el dataset).
    Con delete=False solo cuenta. Llamar después de check() sin problemas."""
    render, dataset = Path(render), Path(dataset)
    cameras = dataset_cameras_of(dataset)
    if delete:
        pointer = {"dataset": str(dataset.resolve()), "cameras": cameras,
                   "episodes": {n: int(i) for n, i in sorted(mapping.items())}}
        tmp = render / (POINTER + ".tmp")
        tmp.write_text("# PNG borrados por scripts/xarm5/upkeep/prune_render_images.py; los cuadros están en este dataset\n"
                       + yaml.safe_dump(pointer, sort_keys=False))
        os.replace(tmp, render / POINTER)
    size, kept, extra = 0, 0, set()
    for ep in sorted(d for d in render.iterdir() if d.is_dir()):
        pngs = png_files(ep)
        if ep.name not in mapping:
            kept += bool(pngs)
            continue
        for cam, files in pngs.items():
            if cam not in cameras:
                extra.add(cam)
                continue
            for f in files:
                size += f.stat().st_size
                if delete:
                    f.unlink()
    return size, kept, sorted(extra)
