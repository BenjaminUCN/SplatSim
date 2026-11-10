"""Pedúnculo provisional del tomate: un cilindro verde de gaussianas y la geometría del corte.

El tallo se describe en el marco del cuerpo del tomate en PyBullet (bottom, top y cut, en metros), y
ese marco es el que usa transform_object después de aplicar la matriz del objeto. Un escaneo real
puede usar lo mismo: otro .ply y sus puntos bottom, top y cut en objects.yaml.

Un corte necesita que el tallo atraviese el plano de la hoja, cerca del filo y de través. El plano de
la hoja es el perpendicular al eje del pivote.
"""
import numpy as np

STEM_RGB = (0.27, 0.46, 0.13)
LENGTH = 0.05           # m de tallo visible sobre el tomate
RADIUS = 0.0025         # m
EMBED = 0.008           # m de tallo dentro del tomate, para que no quede una rendija
SPACING = 0.001         # m entre gaussianas
BLADE_REACH = 0.015     # m: a lo más esto del centro del filo, medido en el plano de la hoja
MAX_TILT_DEG = 45.0     # entre el tallo y el eje del pivote: más que esto, la hoja lo roza de lado
CLOSED_FRACTION = 0.8   # gripper (0 abierto, 1 cerrado) desde el que la hoja ya pasó por el tallo
RELEASE_FRACTION = 0.6  # cut-and-hold: el tallo cortado sigue sujeto hasta abrir por debajo de esto

INTACT, HELD, DROPPED = "intact", "held", "dropped"


def stem_from_tomato(xyz: np.ndarray, length: float = LENGTH) -> dict:
    """Puntos del tallo a partir de las gaussianas del tomate, en el marco del cuerpo: sale hacia
    arriba desde lo alto del tomate, sobre su centro."""
    center = np.median(xyz[:, :2], axis=0)
    top_of_tomato = np.percentile(xyz[:, 2], 99)
    point = lambda z: [round(float(v), 4) for v in (*center, z)]  # noqa: E731
    return {"bottom": point(top_of_tomato - EMBED), "top": point(top_of_tomato + length),
            "cut": point(top_of_tomato + length / 2), "radius": RADIUS}


def cylinder_points(bottom, top, radius: float, spacing: float, rng) -> np.ndarray:
    """Puntos al azar sobre el manto y las tapas del cilindro, uno por spacing² de área."""
    bottom, top = np.asarray(bottom, float), np.asarray(top, float)
    axis = top - bottom
    length = np.linalg.norm(axis)
    axis = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    side_n = max(int(2 * np.pi * radius * length / spacing**2), 1)
    angle, along = rng.uniform(0, 2 * np.pi, side_n), rng.uniform(0, length, side_n)
    side = bottom + along[:, None] * axis + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)

    cap_n = max(int(np.pi * radius**2 / spacing**2), 1)
    angle, r = rng.uniform(0, 2 * np.pi, cap_n), radius * np.sqrt(rng.uniform(0, 1, cap_n))
    cap = top + r[:, None] * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
    return np.vstack([side, cap])


def next_stem_state(state: str, gripper: float, was_closed: bool, check) -> str:
    """Cut-and-hold: al terminar de cerrar sobre el tallo lo corta y lo deja sujeto; al abrir lo
    suficiente lo suelta. check() -> resultado de cut_check; solo se evalúa al terminar de cerrar."""
    if state == INTACT and gripper >= CLOSED_FRACTION and not was_closed:
        return HELD if check() == "ok" else INTACT
    if state == HELD and gripper < RELEASE_FRACTION:
        return DROPPED
    return state


def to_world(point, position, rotation) -> np.ndarray:
    """Punto del marco del tomate en el mundo. rotation: matriz (3, 3)."""
    return np.asarray(rotation) @ np.asarray(point, float) + np.asarray(position, float)


def cut_check(bottom, top, blade, pivot_axis, reach: float = BLADE_REACH,
              max_tilt_deg: float = MAX_TILT_DEG) -> str:
    """Si la hoja cortaría el tallo, ahora: "ok" o por qué no (tilted, misses_plane, too_far)."""
    bottom, top, blade = (np.asarray(v, float) for v in (bottom, top, blade))
    axis = np.asarray(pivot_axis, float) / np.linalg.norm(pivot_axis)
    stem = top - bottom
    along = float(np.dot(stem, axis))
    tilt = np.degrees(np.arccos(np.clip(abs(along) / np.linalg.norm(stem), 0.0, 1.0)))
    if tilt > max_tilt_deg:
        return "tilted"
    t = float(np.dot(blade - bottom, axis)) / along
    if not 0.0 <= t <= 1.0:
        return "misses_plane"
    if np.linalg.norm(bottom + t * stem - blade) > reach:
        return "too_far"
    return "ok"
