"""Quita las imágenes float32 que las grabaciones antiguas de GELLO guardaban dentro de cada .pkl.

Esas grabaciones (gello_v2_dual, gello_v3_, gello_3plants, gello_model) traen `base_rgb` y `wrist_rgb`
como tensores de torch (3, 480, 640) en cuda:0 dentro de cada frame, ~7 MB por .pkl, además de los
PNG de `image_1/`. Nada del código actual lee esos tensores (el conversor y el replay usan `image_1/`).

Por cada frame:
  - la imagen pasa a uint8 (round(x * 255));
  - si ya hay un PNG igual en `image_1/` (diferencia ≤ 1 por canal) no se escribe nada;
  - si no, se guarda en `pkl_images/<cámara>_<frame>.png`;
  - `pkl_images.json` del episodio anota de dónde sacar cada imagen ({frame: {cámara: ruta relativa}});
  - el .pkl se reescribe sin los tensores (atómico: tmp + os.replace).

Reanudable: un .pkl sin tensores se salta. Los tensores se cargan en CPU (no toca la GPU).

    python scripts/xarm5/recordings/compact_recordings.py ~/data/bc_data/gello_3plants ~/data/bc_data/gello_model --workers 6
"""
import argparse
import io
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MANIFEST = "pkl_images.json"
OUT_DIR = "pkl_images"
TOLERANCE = 1


def _cpu_storage(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


torch.storage._load_from_bytes = _cpu_storage   # tensores guardados en cuda:0 se cargan en CPU


def image_keys(frame: dict) -> list[str]:
    return [k for k, v in frame.items() if torch.is_tensor(v) and v.ndim == 3]


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float en [0, 1] -> (H, W, 3) uint8."""
    a = t.detach().cpu().float().clamp(0, 1).numpy().transpose(1, 2, 0)
    return np.round(a * 255).astype(np.uint8)


def candidates(cam: str, pos: int) -> list[int]:
    """Índices de image_1/<cam>_<n>.png que pueden ser el frame `pos` (0-based), los más probables
    primero: en SplatSim base_rgb es 2*pos+1 y wrist_rgb 2*pos+2; otras grabaciones van corridas."""
    first = 2 * pos + (1 if cam == "base_rgb" else 2)
    near = [first, first - 1, first + 1, first - 2, first + 2, first - 3, first + 3]
    return [n for n in near if n >= 0]


def same_png(path: Path, img: np.ndarray) -> bool:
    if not path.exists():
        return False
    png = np.asarray(Image.open(path))[..., :3]
    return png.shape == img.shape and int(np.abs(png.astype(np.int16) - img).max()) <= TOLERANCE


def write_atomic(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compact_episode(episode: Path) -> tuple[int, int, int, list[str]]:
    """(frames compactados, imágenes ya en image_1, imágenes escritas en pkl_images, .pkl que no se pueden
    leer). Un .pkl dañado queda como está."""
    pkls = sorted(episode.glob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    manifest_path = episode / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    done = reused = written = 0
    broken = []
    for pos, pkl in enumerate(pkls):
        try:
            with open(pkl, "rb") as f:
                frame = pickle.load(f)
        except Exception as e:
            broken.append(f"{pkl}: {type(e).__name__}: {e}")
            continue
        keys = image_keys(frame)
        if not keys:
            continue
        refs = {}
        for cam in keys:
            img = to_uint8(frame.pop(cam))
            ref = next((f"image_1/{cam}_{n:05d}.png" for n in candidates(cam, pos)
                        if same_png(episode / "image_1" / f"{cam}_{n:05d}.png", img)), None)
            if ref:
                reused += 1
            else:
                (episode / OUT_DIR).mkdir(exist_ok=True)
                ref = f"{OUT_DIR}/{cam}_{pkl.stem}.png"
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                write_atomic(episode / ref, buf.getvalue())
                written += 1
            refs[cam] = ref
        manifest[pkl.stem] = refs
        write_atomic(manifest_path, json.dumps(manifest, indent=1, sort_keys=True).encode())
        data = pickle.dumps(frame)
        if image_keys(pickle.loads(data)) or set(pickle.loads(data)) != set(frame):
            raise RuntimeError(f"{pkl}: el pkl compactado no se relee bien")
        write_atomic(pkl, data)
        done += 1
    return done, reused, written, broken


def episodes(roots: list[Path]) -> list[Path]:
    return sorted(p for root in roots for p in root.iterdir()
                  if p.is_dir() and not p.is_symlink() and any(p.glob("*.pkl")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="carpetas de grabaciones (con episodios 000, 001, ...)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    eps = episodes([r.expanduser() for r in args.roots])
    print(f"{len(eps)} episodios", flush=True)
    totals = np.zeros(3, dtype=int)
    broken = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (ep, result) in enumerate(zip(eps, pool.map(compact_episode, eps)), 1):
            totals += result[:3]
            broken += result[3]
            print(f"[{i}/{len(eps)}] {ep}: {result[0]} frames, {result[1]} en image_1, {result[2]} nuevas"
                  + (f", {len(result[3])} dañados" if result[3] else ""), flush=True)
    print(f"total: {totals[0]} frames, {totals[1]} imágenes ya en image_1, {totals[2]} escritas en {OUT_DIR}/")
    for line in broken:
        print("dañado (sin tocar):", line)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
