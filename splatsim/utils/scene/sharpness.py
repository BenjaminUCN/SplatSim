"""La foto más nítida de cada grupo de fotos consecutivas de un video, antes de COLMAP.

Nitidez = varianza del Laplaciano de la foto en gris: los bordes marcados la suben y el desenfoque
(de movimiento o de foco) la baja. Depende también del contenido (un primer plano de un tomate liso da
poco aunque esté enfocado), por eso no se compara con un umbral global sino entre fotos vecinas, que
muestran casi lo mismo: de cada `window` consecutivas queda la más nítida. Se extrae el video a
fps × window para terminar con las mismas fotos por segundo.

En el video del tomate (132 fotos a 2 fps), 13 de las 16 fotos que COLMAP no pudo ubicar estaban bajo
el 40 % de la nitidez mediana.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MAX_WIDTH = 1280      # se mide a este ancho: más rápido en 4K, y el ruido del sensor pesa menos
WEAK_WINDOW = 0.3     # un grupo cuya mejor foto queda bajo esto × la mediana elegida: tramo movido


@dataclass
class Frame:
    path: Path
    sharpness: float
    window: int
    kept: bool = False


def sharpness(image: Image.Image) -> float:
    """Varianza del Laplaciano de 4 vecinos (como cv2.Laplacian con ksize=1) en gris."""
    gray = image.convert("L")
    if gray.width > MAX_WIDTH:
        gray = gray.resize((MAX_WIDTH, round(gray.height * MAX_WIDTH / gray.width)), Image.BOX)
    g = np.asarray(gray, dtype=np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def pick_sharpest(photos: list[Path], window: int) -> list[Frame]:
    """Todas las fotos (en orden) con su nitidez; kept en la más nítida de cada `window` consecutivas."""
    frames = [Frame(p, sharpness(Image.open(p)), i // window) for i, p in enumerate(photos)]
    for start in range(0, len(frames), window):
        max(frames[start:start + window], key=lambda f: f.sharpness).kept = True
    return frames


def weak_windows(frames: list[Frame]) -> list[Frame]:
    """Fotos elegidas que igual quedan muy bajo la mediana: todo su grupo estaba movido."""
    kept = [f for f in frames if f.kept]
    median = float(np.median([f.sharpness for f in kept])) if kept else 0.0
    return [f for f in kept if f.sharpness < WEAK_WINDOW * median]


def write_report(frames: list[Frame], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "window", "sharpness", "kept"])
        for frame in frames:
            writer.writerow([frame.path.name, frame.window, f"{frame.sharpness:.2f}", int(frame.kept)])
