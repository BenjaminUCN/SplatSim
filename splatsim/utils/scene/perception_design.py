"""Diseño del experimento de percepción (capstone): pose inicial del brazo (A1–A5) x categoría del tomate (B1–B4).

La cámara base es fija (eye-to-hand). Cada muestra es un episodio del planificador que parte de una
pose A y termina en el tomate, como los grupos de 5 episodios por posición de tomate de gello_3plants
(cada grupo parte de las 5 poses). Por cada posición de tomate hay 4 categorías x 5 poses = 20
episodios. Aquí están las definiciones y las reglas con que se acepta o marca el primer instante de
cada episodio; la generación vive en scripts/xarm5/plan_perception_episodes.py.

Sin dependencias pesadas: se prueba sin PyBullet ni GPU.
"""
from dataclasses import dataclass

import numpy as np

ARM_CONFIGS = {
    "A1": ("OUT_LEFT", "fuera del campo de visión, hacia la izquierda"),
    "A2": ("OUT_RIGHT", "fuera del campo de visión, hacia la derecha"),
    "A3": ("BASE", "pose de reset: centrada, por encima de la imagen"),
    "A4": ("IN_VIEW_FAR", "visible en la imagen, lejos del tomate"),
    "A5": ("IN_VIEW_NEAR", "visible en la imagen, cerca del tomate"),
}
# categoría: (visible, unido a la planta, descripción)
TOMATO_CATEGORIES = {
    "B1": (True, True, "en el encuadre, en su posición (caso real)"),
    "B2": (False, True, "en su posición, tapado por una hoja"),
    "B3": (False, False, "separado de su posición y fuera del encuadre"),
    "B4": (True, False, "separado de su posición, en el encuadre"),
}


@dataclass(frozen=True)
class Thresholds:
    """Por defecto, calibrados con las poses iniciales de gello_3plants: el brazo visible ocupa
    7–26 % de la imagen; el punto de agarre queda a 0.22–0.42 m del tomate en la pose cercana y a
    0.34–0.61 m en la lejana (p10–p90)."""
    arm_out_max: float = 0.0
    """fracción de la imagen que puede ocupar el brazo en A1–A3 (0 = nada)"""
    arm_in_min: float = 0.03
    """fracción mínima de la imagen que ocupa el brazo en A4/A5 para contar como visible"""
    arm_in_max: float = 0.35
    """fracción máxima de la imagen que ocupa el brazo en A4/A5 (más: pegado a la cámara)"""
    base_max_off_center: float = 0.25
    """A3: |u - 0.5| de la proyección del efector (centrado; v < 0: por encima de la imagen)"""
    far_min: float = 0.35
    """m: A4, distancia mínima del punto de agarre al tomate"""
    near_max: float = 0.33
    """m: A5, distancia máxima del punto de agarre al tomate"""
    visible_max_occlusion: float = 50.0
    """%: B1/B4, oclusión máxima del tomate por la escena (sin el brazo) para contar como visible"""
    hidden_min_occlusion: float = 90.0
    """%: B2, oclusión mínima para contar como oculto (o fuera de cuadro)"""


def check_arm(config: str, m: dict, t: Thresholds) -> list[str]:
    """Motivos por los que la pose inicial no cumple su condición ([] = cumple). m: arm_fraction
    (fracción de la imagen), ee_uv (proyección del efector en fracción de la imagen, None si está
    detrás de la cámara) y grasp_distance (m, del punto de agarre de la pinza al tomate)."""
    fraction = m["arm_fraction"]
    if config in ("A1", "A2", "A3"):
        reasons = [] if fraction <= t.arm_out_max else [f"brazo en cuadro ({fraction:.2%} de la imagen)"]
        if config == "A3":
            uv = m["ee_uv"]
            if uv is None or uv[1] >= 0:
                reasons.append("el efector no está por encima de la imagen")
            elif abs(uv[0] - 0.5) > t.base_max_off_center:
                reasons.append(f"el efector no está centrado (u={uv[0]:.2f})")
        return reasons
    if fraction < t.arm_in_min:
        return [f"brazo casi fuera de cuadro ({fraction:.2%} < {t.arm_in_min:.2%})"]
    if fraction > t.arm_in_max:
        return [f"brazo pegado a la cámara ({fraction:.0%} de la imagen > {t.arm_in_max:.0%})"]
    if config == "A4" and m["grasp_distance"] < t.far_min:
        return [f"pinza a {m['grasp_distance']:.2f} m del tomate (< {t.far_min} m)"]
    if config == "A5" and m["grasp_distance"] > t.near_max:
        return [f"pinza a {m['grasp_distance']:.2f} m del tomate (> {t.near_max} m)"]
    return []


def check_tomato(category: str, m: dict, t: Thresholds) -> list[str]:
    """Motivos por los que el tomate no cumple su categoría ([] = cumple). m: occlusion (% tapado
    por la escena sin el brazo, None = fuera de cuadro) y cut_by_frame."""
    occlusion, cut = m["occlusion"], m["cut_by_frame"]
    if category in ("B1", "B4"):
        if occlusion is None:
            return ["tomate fuera de cuadro"]
        reasons = []
        if cut:
            reasons.append("el borde de la imagen corta al tomate")
        if occlusion > t.visible_max_occlusion:
            reasons.append(f"tomate {occlusion:.0f}% tapado (> {t.visible_max_occlusion:.0f}%)")
        return reasons
    if category == "B2":
        if occlusion is not None and occlusion < t.hidden_min_occlusion:
            return [f"tomate solo {occlusion:.0f}% tapado (< {t.hidden_min_occlusion:.0f}%)"]
        return []
    if category == "B3":
        return [] if occlusion is None else ["tomate en cuadro"]
    raise ValueError(f"categoría desconocida {category!r}")


def farthest_points(points, count: int, rng=None, taken=()) -> list[int]:
    """Índices de hasta count puntos bien repartidos, en orden: cada uno es el más lejano a los ya
    elegidos y a taken (puntos elegidos antes, que no se devuelven). Sin taken, el primero es al azar."""
    points = np.asarray(points, float)
    count = min(count, len(points))
    if not count:
        return []
    taken = np.asarray(taken, float).reshape(-1, 3)
    if len(taken):
        distance = np.min(np.linalg.norm(points[:, None] - taken[None], axis=2), axis=1)
        chosen = []
    else:
        rng = rng or np.random.default_rng()
        chosen = [int(rng.integers(len(points)))]
        distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return chosen
