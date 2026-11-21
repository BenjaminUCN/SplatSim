"""Recorta el splat de un objeto suelto con sus máscaras: borra las gaussianas que caen fuera de la
silueta del objeto en demasiadas de las fotos donde se ven.

Un punto del objeto cae dentro de su silueta en todas las fotos; uno del fondo cae fuera en casi
todas, aunque en alguna quede detrás del objeto. Se vota en vez de exigir todas para tolerar
máscaras con errores (las que en alguna foto cortan el pedúnculo), y la máscara se agranda unos
píxeles para no comerse el borde. Las cámaras son las de sparse/0 (PINHOLE, sin distorsión) y las
máscaras, las de images/ (object_masks.py); las gaussianas están en el mismo marco que COLMAP.
"""
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import maximum_filter

from splatsim.utils.scene.object_masks import mask_path

COLMAP_LOADER = (Path(__file__).resolve().parents[3]
                 / "submodules/gaussian-splatting-wrapper/gaussian_splatting/scene/colmap_loader.py")


@dataclass
class View:
    name: str
    R: np.ndarray        # mundo → cámara
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _colmap_loader():
    """colmap_loader.py del submódulo, sin importar su paquete scene (que carga torch)."""
    spec = importlib.util.spec_from_file_location("colmap_loader", COLMAP_LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_views(sparse: Path) -> list[View]:
    loader = _colmap_loader()
    if (sparse / "images.bin").is_file():
        extrinsics = loader.read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = loader.read_intrinsics_binary(str(sparse / "cameras.bin"))
    else:
        extrinsics = loader.read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = loader.read_intrinsics_text(str(sparse / "cameras.txt"))
    views = []
    for image in extrinsics.values():
        camera = intrinsics[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise SystemExit(f"Cámara {camera.model} en {sparse}: se esperan las de image_undistorter (PINHOLE)")
        views.append(View(image.name, loader.qvec2rotmat(image.qvec), np.asarray(image.tvec, dtype=np.float64),
                          float(fx), float(fy), float(cx), float(cy), int(camera.width), int(camera.height)))
    return sorted(views, key=lambda v: v.name)


def load_mask(path: Path, dilate_px: int) -> np.ndarray:
    """Máscara booleana (True = objeto), agrandada dilate_px píxeles."""
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if dilate_px > 0:
        mask = maximum_filter(mask, size=2 * dilate_px + 1)
    return mask


def count_votes(xyz: np.ndarray, views: list[View], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(en cuántas fotos cae dentro del cuadro, en cuántas de ellas fuera de la máscara) por gaussiana."""
    seen = np.zeros(len(xyz), np.int32)
    outside = np.zeros(len(xyz), np.int32)
    for view, mask in zip(views, masks):
        cam = xyz @ view.R.T + view.t
        z = cam[:, 2]
        front = z > 1e-6
        safe_z = np.where(front, z, 1.0)
        # la máscara puede no tener el tamaño de la cámara: se escala
        sx, sy = mask.shape[1] / view.width, mask.shape[0] / view.height
        u = (view.fx * cam[:, 0] / safe_z + view.cx) * sx
        v = (view.fy * cam[:, 1] / safe_z + view.cy) * sy
        in_frame = front & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        idx = np.flatnonzero(in_frame)
        inside = mask[v[idx].astype(np.int64), u[idx].astype(np.int64)]
        seen[idx] += 1
        outside[idx[~inside]] += 1
    return seen, outside


def keep_mask(seen: np.ndarray, outside: np.ndarray, max_outside: float, min_views: int) -> np.ndarray:
    """Se queda si se ve en al menos min_views fotos y cae fuera de la máscara en a lo más max_outside de ellas."""
    return (seen >= min_views) & (outside <= max_outside * seen)


def carve_scene(scene: Path, ply: Path, out: Path, max_outside: float = 0.2, min_views: int = 3,
                dilate_px: int = 10, log: Callable[[str], None] = print) -> tuple[int, int]:
    """Escribe en out el splat de ply sin las gaussianas fuera del objeto. Devuelve (quedan, había)."""
    views = read_views(scene / "sparse" / "0")
    with_mask = [v for v in views if mask_path(scene, v.name).is_file()]
    if len(with_mask) < len(views):
        log(f"⚠️ {len(views) - len(with_mask)} de {len(views)} fotos sin máscara: no votan")
    if len(with_mask) < min_views:
        raise SystemExit(f"Solo {len(with_mask)} fotos con máscara en {scene / 'masks'}: corre antes el paso mask")
    masks = [load_mask(mask_path(scene, v.name), dilate_px) for v in with_mask]

    vertex = PlyData.read(str(ply))["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    seen, outside = count_votes(xyz, with_mask, masks)
    keep = keep_mask(seen, outside, max_outside, min_views)
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex[keep], "vertex")]).write(str(out))
    log(f"carve: quedan {keep.sum()} de {len(keep)} gaussianas ({keep.mean():.1%}); se borraron "
        f"{int(((seen >= min_views) & ~keep).sum())} fuera del objeto en más del {max_outside:.0%} de sus fotos "
        f"y {int((seen < min_views).sum())} vistas en menos de {min_views} → {out}")
    return int(keep.sum()), len(keep)
