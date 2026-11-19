"""Fotos de un objeto suelto (un tomate) sobre fondo blanco, para entrenar su splat con train.py -w.

COLMAP corre con el fondo: un tomate liso y brillante casi no tiene puntos que emparejar, y sin fondo
COLMAP no ubica las cámaras. Solo las fotos con que se entrena pierden el fondo, en otra carpeta:
    <escena>/images/         fotos sin distorsión (las de COLMAP, intactas)
    <escena>/masks/<foto>.png máscara de cada foto (255 = objeto), para revisarla o corregirla a mano
    <escena>/images_white/   fotos compuestas sobre blanco  ← train.py -i images_white -w

Cada foto conserva nombre, extensión y tamaño: train.py la busca por el nombre que guardó COLMAP y
las cámaras de sparse/0 son las de images/. Con fondo blanco en las fotos y en el render (-w), el
fondo entra en la pérdida y el entrenamiento lo deja vacío. Una foto RGBA haría lo contrario:
train.py multiplica el render por el alfa, así que el fondo quedaría fuera de la pérdida y las
gaussianas flotantes no se castigarían.
"""
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = "isnet-general-use"
WHITE_FOLDER, MASK_FOLDER = "images_white", "masks"

Masker = Callable[[Image.Image], Image.Image]
"""Foto RGB → máscara L del mismo tamaño (255 = objeto, valores intermedios en el borde)."""


def composite_on_white(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """rgb (H, W, 3) uint8 sobre blanco, con mask (H, W) uint8 como alfa suave."""
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"La máscara mide {mask.shape} y la foto {rgb.shape[:2]}")
    # en enteros: rgb·a + 255·(255 - a) ≤ 255·255 cabe en uint16, y en una foto 4K es ~10× más rápido que float
    alpha = mask.astype(np.uint16)[..., None]
    out = rgb.astype(np.uint16) * alpha + 255 * (255 - alpha) + 127
    return (out // 255).astype(np.uint8)


def rembg_masker(model: str = DEFAULT_MODEL) -> Masker:
    """Máscaras con rembg, en la GPU si onnxruntime-gpu la encuentra (la primera foto tarda ~1 min
    en una RTX 5080 mientras CUDA compila; después ~0.1 s por foto; en la CPU, ~0.7 s). El modelo se
    carga con la primera foto que lo necesite: con todas las máscaras reusadas, no se carga."""
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


def whiten_folder(scene: Path, masker: Masker | None, reuse_masks: bool = False,
                  log: Callable[[str], None] = print) -> int:
    """images/ → masks/ y images_white/. Con reuse_masks, las máscaras que ya estén en masks/ (p.ej.
    corregidas a mano) se usan tal cual y solo se calculan las que falten; si no, se calculan todas.
    Devuelve cuántas máscaras calculó."""
    images, masks, white = scene / "images", scene / MASK_FOLDER, scene / WHITE_FOLDER
    photos = photos_in(images) if images.is_dir() else []
    if not photos:
        raise SystemExit(f"No hay fotos en {images}: corre antes COLMAP (convert.py o --steps colmap)")
    if not reuse_masks:
        shutil.rmtree(masks, ignore_errors=True)
    shutil.rmtree(white, ignore_errors=True)
    masks.mkdir(parents=True, exist_ok=True)
    white.mkdir(parents=True)

    computed, coverage = 0, []
    for i, photo in enumerate(photos):
        image = Image.open(photo).convert("RGB")
        mask_path = masks / f"{photo.stem}.png"
        if reuse_masks and mask_path.is_file():
            mask = Image.open(mask_path).convert("L")
        else:
            if masker is None:
                raise SystemExit(f"Falta {mask_path} y no hay con qué calcularla")
            mask = masker(image)
            mask.save(mask_path)
            computed += 1
        if mask.size != image.size:
            raise SystemExit(f"{mask_path} mide {mask.size} y la foto {image.size}")
        mask_array = np.asarray(mask)
        coverage.append(float((mask_array > 127).mean()))
        out = Image.fromarray(composite_on_white(np.asarray(image), mask_array))
        jpeg = photo.suffix.lower() in {".jpg", ".jpeg"}
        out.save(white / photo.name, **({"quality": 95, "subsampling": 0} if jpeg else {}))
        if (i + 1) % 50 == 0 or i + 1 == len(photos):
            print(f"[{i + 1}/{len(photos)}] máscaras", flush=True)

    coverage = np.asarray(coverage)
    empty = int((coverage < 0.001).sum())
    log(f"fondo blanco: {len(photos)} fotos ({computed} máscaras nuevas); el objeto cubre "
        f"{np.median(coverage):.1%} de la foto (mediana), {coverage.min():.1%}–{coverage.max():.1%}")
    if empty:
        log(f"⚠️ {empty} foto(s) sin objeto en la máscara: revisa {masks}")
    return computed
