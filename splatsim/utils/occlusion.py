"""Cuánto del tomate tapa la escena, por cámara.

Se renderiza la escena completa dos veces, con el tomate pintado de negro y de blanco: lo único que
cambia entre ambas es lo que el tomate aporta a cada píxel después de la mezcla alfa, así que la
diferencia es exactamente la parte visible, y las hojas translúcidas tapan en su justa medida, sin
umbrales. Dividida por la silueta del tomate solo, da la fracción visible.

No incluye el recorte por el borde de la imagen: si el tomate sale en parte del cuadro, el porcentaje
es sobre la parte que queda dentro (cut_by_frame lo avisa). La escena incluye al brazo: si pasa
delante, también cuenta.
"""
import copy
from typing import Callable

import torch

SH_C0 = 0.28209479177387814
BLACK_DC, WHITE_DC = -0.5 / SH_C0, 0.5 / SH_C0   # f_dc que dan color 0 y 1
MIN_SILHOUETTE_PX = 10.0                         # menos que esto en cuadro: fuera de vista
LEVEL_EDGES = (25.0, 50.0, 75.0)
LEVELS = ("0-25", "25-50", "50-75", "75-100")
OUT_OF_VIEW = "out_of_view"


def occlusion_level(percent: float | None) -> str:
    if percent is None:
        return OUT_OF_VIEW
    for edge, level in zip(LEVEL_EDGES, LEVELS):
        if percent < edge:
            return level
    return LEVELS[-1]


def _paint(gaussian, target: slice, dc: float) -> None:
    gaussian._features_dc[target] = dc
    gaussian._features_rest[target] = 0.0     # sin brillo: el color es el mismo desde todo ángulo


def measure_camera(render: Callable, gaussian, target: slice, camera, background) -> dict:
    """render(camera, gaussian, background) -> imagen (3, H, W). target: las gaussianas del tomate
    dentro de gaussian, que queda como estaba."""
    with torch.no_grad():
        dc, rest = gaussian._features_dc, gaussian._features_rest
        try:
            gaussian._features_dc, gaussian._features_rest = dc.clone(), rest.clone()
            _paint(gaussian, target, BLACK_DC)
            black = render(camera, gaussian, background)
            _paint(gaussian, target, WHITE_DC)
            white = render(camera, gaussian, background)

            alone = copy.copy(gaussian)            # solo el tomate, blanco sobre negro: su alfa
            for field in ("_xyz", "_rotation", "_opacity", "_scaling", "_features_dc", "_features_rest"):
                setattr(alone, field, getattr(gaussian, field)[target])
            silhouette = render(camera, alone, torch.zeros_like(background)).mean(0).clamp(0, 1)
        finally:
            gaussian._features_dc, gaussian._features_rest = dc, rest
        visible = (white - black).mean(0).clamp(0, 1)
        silhouette_px = float(silhouette.sum())
        visible_px = float(visible.sum())
        border = torch.cat([silhouette[0], silhouette[-1], silhouette[:, 0], silhouette[:, -1]])
    percent = None
    if silhouette_px >= MIN_SILHOUETTE_PX:
        percent = round(100.0 * min(max(1.0 - visible_px / silhouette_px, 0.0), 1.0), 1)
    return {"occlusion": percent, "level": occlusion_level(percent),
            "silhouette_px": round(silhouette_px, 1), "visible_px": round(visible_px, 1),
            "cut_by_frame": bool(border.max() > 0.5)}


def fit_message(fit: dict, camera: str) -> str:
    """El resultado de fit_lateral_offset para el panel."""
    if fit["status"] == OUT_OF_VIEW:
        return f"The tomato isn't in view of {camera}."
    return {
        "ok": "{:.0f}% in {}.",
        "too_small": "At most {:.0f}% in {}: move it closer to the camera or use a bigger leaf.",
        "already_covered": "The rest of the scene already covers {:.0f}% in {}.",
    }[fit["status"]].format(fit["occlusion"], camera)


def fit_lateral_offset(measure: Callable[[float], float | None], target: float,
                       max_offset: float = 0.3, iterations: int = 10) -> dict:
    """Corrimiento lateral con que un oclusor tapa target %: 0 = centrado en el tomate (lo máximo
    que puede tapar), max_offset = fuera de él (lo que tapa el resto de la escena).

    measure(offset) -> % o None si el tomate no se ve. status: ok, too_small (ni centrado llega),
    already_covered (sin la hoja ya tapa más) u out_of_view. La forma de la hoja no es regular, así
    que devuelve la mejor medida vista, que no tiene por qué ser exacta."""
    centered = measure(0.0)
    if centered is None:
        return {"status": OUT_OF_VIEW, "offset": 0.0, "occlusion": None}
    if centered <= target:
        return {"status": "ok" if centered == target else "too_small", "offset": 0.0, "occlusion": centered}
    away = measure(max_offset)
    if away is not None and away >= target:
        return {"status": "ok" if away == target else "already_covered", "offset": max_offset, "occlusion": away}
    best = (abs(centered - target), 0.0, centered)
    low, high = 0.0, max_offset
    for _ in range(iterations):
        middle = (low + high) / 2
        value = measure(middle)
        if value is None:     # no debería pasar: el corrimiento no mueve al tomate
            break
        best = min(best, (abs(value - target), middle, value))
        if value > target:
            low = middle
        else:
            high = middle
    return {"status": "ok", "offset": best[1], "occlusion": best[2]}
