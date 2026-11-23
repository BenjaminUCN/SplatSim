"""Marco del cuerpo de un tomate real a partir de su splat recortado (splat_from_video.py --carve).

El splat de COLMAP está en un marco arbitrario y sin escala. El tomate sintético (plastic_apple) tiene
el origen en la base del fruto, z hacia arriba a lo largo del pedúnculo y unidades en metros; aquí se
calcula la transformación que deja al real igual:

1. Fruto: las gaussianas rojas, con una esfera ajustada (centro y radio en unidades de COLMAP).
2. Objeto: la componente conexa (vóxeles de r/4) que contiene al fruto; lo demás son restos que el
   recorte dejó sueltos.
3. Eje z: del centro del fruto al centroide de las gaussianas no rojas a 1–2 radios de él (el cáliz y
   el pedúnculo antes de que se doble hacia la rama). x apunta hacia donde se inclina la rama, para orientarla con --yaw.
4. Escala: el diámetro del fruto, perpendicular al eje, pasa a ser el pedido (por defecto el del tomate
   sintético, para que calce con su malla de colisión en PyBullet).
5. Pedúnculo: bottom en lo alto del fruto, top donde el tallo se aparta del eje, cut entre medio.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

SH_C0 = 0.28209479177387814
MAX_GAP = 0.014          # m de tallo sin gaussianas (mal reconstruido) que se salta
MAX_SEPAL_GAP = 0.03     # m sobre el fruto en que los sépalos pueden tapar el tallo


@dataclass
class Fit:
    matrix: np.ndarray       # 4x4, .ply -> cuerpo (m)
    keep: np.ndarray         # gaussianas del objeto (bool por vértice)
    peduncle: dict           # bottom, top, cut (m, marco del cuerpo) y radius
    aabb: list               # [[min], [max]] en el cuerpo
    radius_colmap: float     # radio de la esfera del fruto, en unidades de COLMAP


def colors(vertices) -> np.ndarray:
    dc = np.stack([vertices[f"f_dc_{i}"] for i in range(3)], axis=1).astype(float)
    return np.clip(0.5 + SH_C0 * dc, 0, 1)


def opacities(vertices) -> np.ndarray:
    return 1 / (1 + np.exp(-vertices["opacity"].astype(float)))


def is_red(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, 0] > 1.3 * rgb[:, 1]) & (rgb[:, 0] > 0.3)


def fit_sphere(points: np.ndarray, rounds: int = 8) -> tuple[np.ndarray, float]:
    """Esfera por mínimos cuadrados, descartando en cada ronda lo que queda a más de 2 medianas."""
    center = np.median(points, axis=0)
    for _ in range(rounds):
        d = np.linalg.norm(points - center, axis=1)
        p = points[d < 2 * np.median(d)]
        A = np.c_[2 * p, np.ones(len(p))]
        sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
        center = sol[:3]
    return center, float(np.sqrt(sol[3] + center @ center))


def main_component(xyz: np.ndarray, seed: np.ndarray, voxel: float) -> np.ndarray:
    """Las gaussianas en la componente conexa de vóxeles que contiene más puntos de seed."""
    vox = np.floor((xyz - xyz.min(0)) / voxel).astype(int)
    grid = np.zeros(vox.max(0) + 1, bool)
    grid[tuple(vox.T)] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    point_labels = labels[tuple(vox.T)]
    main = np.bincount(point_labels[seed]).argmax()
    return point_labels == main


def perpendicular(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - np.outer(v @ axis, axis)


def fit_tomato(vertices, diameter: float, yaw_deg: float = 0.0, min_opacity: float = 0.2,
               max_bend: float = 0.004, cut_fraction: float = 0.5, min_cut: float = 0.01) -> Fit:
    xyz = np.stack([vertices[k] for k in "xyz"], axis=1).astype(float)
    rgb, opacity = colors(vertices), opacities(vertices)
    solid = opacity > min_opacity
    red = is_red(rgb) & solid
    if red.sum() < 50:
        raise SystemExit(f"Solo {red.sum()} gaussianas rojas: ¿es un tomate?")
    center, r = fit_sphere(xyz[red])
    keep = main_component(xyz, red & (np.linalg.norm(xyz - center, axis=1) < 1.5 * r), r / 4)
    X = xyz - center
    d = np.linalg.norm(X, axis=1)

    stem = keep & solid & ~is_red(rgb)
    near = stem & (d >= r) & (d < 2 * r)
    if near.sum() < 10:
        raise SystemExit(f"Solo {near.sum()} gaussianas de tallo junto al fruto: no se ve el pedúnculo")
    axis = X[near].mean(0)
    axis /= np.linalg.norm(axis)
    origin_line = np.zeros(3)                     # el eje pasa por el centro del fruto

    # x hacia donde se inclina la rama (el tallo lejos del fruto), y luego el giro pedido
    far = stem & (d >= 2.5 * r)
    lean = perpendicular((X[far] if far.any() else X[stem]).mean(0, keepdims=True) - origin_line, axis)[0]
    if np.linalg.norm(lean) < 1e-9:
        lean = perpendicular(np.array([[1.0, 0, 0]]), axis)[0]
    ex = lean / np.linalg.norm(lean)
    ey = np.cross(axis, ex)
    yaw = np.radians(yaw_deg)
    ex, ey = np.cos(yaw) * ex + np.sin(yaw) * ey, -np.sin(yaw) * ex + np.cos(yaw) * ey
    R = np.stack([ex, ey, axis])

    fruit = keep & red & (d < 1.25 * r)       # sin el tallo rojizo cerca del nudo
    h_fruit = (X[fruit] - origin_line) @ axis
    radial = np.linalg.norm(perpendicular(X[fruit] - origin_line, axis), axis=1)
    scale = diameter / 2 / np.percentile(radial, 98)
    origin = center + origin_line + np.percentile(h_fruit, 0.5) * axis     # base del fruto
    matrix = np.eye(4)
    matrix[:3, :3] = scale * R
    matrix[:3, 3] = -scale * R @ origin

    body = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    peduncle = _peduncle(body[fruit], body[stem], max_bend, cut_fraction, min_cut)
    low = np.percentile(body[keep], 0.1, axis=0) - 0.005
    high = np.percentile(body[keep], 99.9, axis=0) + 0.005
    aabb = [[round(float(v), 4) for v in low], [round(float(v), 4) for v in high]]
    return Fit(matrix, keep, peduncle, aabb, r)


def _peduncle(fruit: np.ndarray, stem: np.ndarray, max_bend: float, cut_fraction: float, min_cut: float) -> dict:
    """bottom/top/cut sobre el eje z (m). Subiendo en tramos de 2 mm, el tallo sigue mientras haya
    gaussianas a menos de max_bend del eje (los sépalos, también verdes, se abren más), con huecos de
    hasta MAX_GAP; top es el último tramo con alguna. Justo sobre el fruto los sépalos suelen tapar el
    tallo, así que el primer tramo visible puede estar hasta MAX_SEPAL_GAP más arriba, y cut va en la
    fracción cut_fraction del tramo visible, no de todo el tallo. radius: la distancia mediana al eje
    de esas gaussianas, entre 1.5 y 4 mm."""
    radial = np.linalg.norm(fruit[:, :2], axis=1)
    bottom = float(np.percentile(fruit[radial < 0.3 * np.percentile(radial, 98), 2], 99))
    step = 0.002
    z, top, first, misses = bottom, bottom, None, 0
    radii = []
    while misses * step < (MAX_GAP if first is not None else MAX_SEPAL_GAP) and z < bottom + 0.15:
        band = stem[(stem[:, 2] >= z) & (stem[:, 2] < z + step)]
        off = np.linalg.norm(band[:, :2], axis=1)
        close = off[off < max_bend]
        if len(close):
            first = z if first is None else first
            top, misses = z + step, 0
            radii.append(np.median(close))
        else:
            misses += 1
        z += step
    if first is None or top - first < min_cut:
        visible = 0 if first is None else 1000 * (top - first)
        raise SystemExit(f"Tramo recto visible del pedúnculo de solo {visible:.0f} mm")
    cut = max(bottom + min_cut, first + cut_fraction * (top - first))
    radius = float(np.clip(np.median(radii) if radii else 0.0025, 0.0015, 0.004))
    point = lambda z: [0.0, 0.0, round(float(z), 4)]   # noqa: E731
    return {"bottom": point(bottom), "top": point(top), "cut": point(cut), "radius": round(radius, 4)}
