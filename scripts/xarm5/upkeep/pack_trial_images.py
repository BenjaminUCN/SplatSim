"""Pasa los PNG de los ensayos (o de cualquier grabación) a un video H.264 sin pérdida por cámara (~12x menos).

Cada <episodio>/image_1/<cam>_<n>.png queda como image_1/<cam>.mp4 + image_1/videos.json
(splatsim/utils/data/frame_video.py). Antes de borrar los PNG se comprueba que el video da exactamente los
mismos píxeles en todos los cuadros. El revisor (review_recordings.py) muestra igual los episodios empaquetados.
Ojo: lerobot_conversion.py y otros conversores leen PNG; para ellos, --restore los vuelve a escribir.

Reanudable; se salta carpetas modificadas hace menos de --min_age_hours (una corrida que se está grabando).
Corre con prioridad baja (nice 19).

    python scripts/xarm5/upkeep/pack_trial_images.py ~/data/policy_trials --workers 6
    python scripts/xarm5/upkeep/pack_trial_images.py ~/data/policy_trials/<corrida> --restore
"""
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from splatsim.utils.data import frame_video  # noqa: E402

IMAGE_DIRS = ("image_1", "images")


@dataclass
class Args:
    roots: tyro.conf.Positional[tuple[Path, ...]]
    workers: int = 4
    threads: int = 4                 # hilos de ffmpeg por proceso
    min_age_hours: float = 1.0
    restore: bool = False            # video -> PNG
    dry_run: bool = False


def image_dirs(root: Path):
    """Carpetas de imágenes bajo root (no entra en ellas para listar subcarpetas)."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in [d for d in dirnames if d in IMAGE_DIRS]:
            yield Path(dirpath) / d
        dirnames[:] = [d for d in dirnames if d not in IMAGE_DIRS]


def dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in os.scandir(d) if f.is_file())


def _work(d: Path, threads: int, restore: bool):
    before = dir_bytes(d)
    if restore:
        frame_video.restore(d)
        result = {}
    else:
        result = frame_video.pack(d, threads)
    return d, before, dir_bytes(d), result


def main(args: Args):
    os.nice(19)
    cutoff = time.time() - args.min_age_hours * 3600
    dirs = []
    for root in args.roots:
        for d in image_dirs(root.expanduser()):
            if d.stat().st_mtime > cutoff:
                print(f"[pack] reciente, se salta: {d}")
                continue
            has_png = any(frame_video.PNG.match(f.name) for f in os.scandir(d))
            if has_png != args.restore or (args.restore and frame_video.load_manifest(d)):
                dirs.append(d)
    print(f"[pack] {len(dirs)} carpetas {'a restaurar' if args.restore else 'con PNG'}")
    if args.dry_run or not dirs:
        return
    total_before = total_after = 0
    reasons = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_work, d, args.threads, args.restore) for d in dirs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                d, before, after, result = fut.result()
            except Exception as e:  # noqa: BLE001  una carpeta mala no detiene el resto
                print(f"[pack] ERROR: {e}")
                reasons["error"] += 1
                continue
            total_before += before
            total_after += after
            for cam, why in result.items():
                reasons[why or "ok"] += 1
                if why:
                    print(f"[pack] {d} {cam}: {why}")
            if i % 200 == 0 or i == len(dirs):
                print(f"[pack] {i}/{len(dirs)}  {total_before / 1e9:.1f} GB -> {total_after / 1e9:.1f} GB", flush=True)
    print(f"[pack] cámaras: {dict(reasons)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
