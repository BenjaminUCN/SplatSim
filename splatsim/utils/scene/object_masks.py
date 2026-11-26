"""Máscaras de un objeto suelto (un tomate) en cada foto de COLMAP, para recortar su splat (carve.py).

El splat se entrena con el fondo, como en SplatSim original: así las gaussianas del fondo se forman
donde está el fondo (cajas, mesa) y no pegadas al objeto. Sin fondo no funciona: con fotos sobre
blanco y train.py -w, una gaussiana blanca opaca y el espacio vacío dan el mismo píxel, y los puntos
iniciales de COLMAP quedaron como una capa blanca alrededor del tomate (51 % de las gaussianas); con
RGBA, train.py multiplica el render por el alfa y el fondo ni siquiera entra en la pérdida.

    <escena>/images/            fotos sin distorsión de COLMAP (intactas)
    <escena>/masks/<foto>.png   máscara de cada foto (255 = objeto), para revisarla o corregirla a mano

birefnet-general conserva el pedúnculo y la rama; isnet-general-use (el más rápido) los corta y a veces
pierde el tomate o deja pasar una caja.
"""
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = "birefnet-general"
MASK_FOLDER = "masks"

Masker = Callable[[Image.Image], Image.Image]
"""Foto RGB → máscara L del mismo tamaño (255 = objeto, valores intermedios en el borde)."""


def rembg_masker(model: str = DEFAULT_MODEL) -> Masker:
    """Máscaras con rembg, en la GPU si onnxruntime-gpu la encuentra. En una RTX 5080, birefnet-general
    tarda ~3.4 s por foto e isnet-general-use ~0.1 s, más ~1 min la primera mientras CUDA compila. El
    modelo se carga con la primera foto que lo necesite: con todas las máscaras reusadas, no se carga."""
    session = None

    def masker(image: Image.Image) -> Image.Image:
        nonlocal session
        from rembg import new_session, remove

        if session is None:
            import onnxruntime as ort

            try:
                ort.preload_dlls()      # las librerías de CUDA del paquete de torch (nvidia-*-cu12)
            except Exception:
                pass
            session = new_session(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return remove(image, session=session, only_mask=True).convert("L")

    return masker


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def mask_path(scene: Path, photo_name: str) -> Path:
    return scene / MASK_FOLDER / f"{Path(photo_name).stem}.png"


def compute_masks(scene: Path, masker: Masker | None, reuse_masks: bool = False,
                  log: Callable[[str], None] = print) -> int:
    """Una máscara por foto de images/ en masks/. Con reuse_masks, las que ya estén (p.ej. corregidas a
    mano) se dejan y solo se calculan las que falten. Devuelve cuántas calculó."""
    images, masks = scene / "images", scene / MASK_FOLDER
    photos = photos_in(images) if images.is_dir() else []
    if not photos:
        raise SystemExit(f"No hay fotos en {images}: corre antes COLMAP (convert.py o --steps colmap)")
    if not reuse_masks:
        shutil.rmtree(masks, ignore_errors=True)
    masks.mkdir(parents=True, exist_ok=True)

    computed, coverage = 0, []
    for i, photo in enumerate(photos):
        path = mask_path(scene, photo.name)
        if reuse_masks and path.is_file():
            mask = Image.open(path).convert("L")
        else:
            if masker is None:
                raise SystemExit(f"Falta {path} y no hay con qué calcularla")
            mask = masker(Image.open(photo).convert("RGB"))
            mask.save(path)
            computed += 1
        coverage.append(float((np.asarray(mask) > 127).mean()))
        if (i + 1) % 50 == 0 or i + 1 == len(photos):
            print(f"[{i + 1}/{len(photos)}] máscaras", flush=True)

    coverage = np.asarray(coverage)
    empty = int((coverage < 0.001).sum())
    log(f"máscaras: {len(photos)} fotos ({computed} nuevas); el objeto cubre {np.median(coverage):.1%} de la "
        f"foto (mediana), {coverage.min():.1%}–{coverage.max():.1%}")
    if empty:
        log(f"⚠️ {empty} foto(s) sin objeto en la máscara: revisa {masks}")
    return computed
