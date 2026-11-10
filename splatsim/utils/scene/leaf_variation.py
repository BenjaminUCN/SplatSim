"""Una hoja distinta por episodio, ajustada a un porcentaje de oclusión objetivo.

Para cada episodio se sortea el objetivo, la hoja, la cámara, la dirección del corrimiento y el giro
de la hoja. El sorteo depende solo de la semilla y del nombre del episodio, así que no cambia al
repartir los episodios entre procesos ni al continuar un render a medias.

La hoja se ajusta en el primer instante del episodio y queda fija en el mundo durante todo él. No
tiene cuerpo en PyBullet: si el brazo pasara por donde está, la atravesaría, así que se busca una
distancia lejos de su recorrido. arm_distance anota lo que quedó.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from splatsim.utils.scene.leaf_segmentation import place_between
from splatsim.utils.render.occlusion import fit_lateral_offset

DEFAULT_TARGETS = (0.0, 25.0, 50.0, 75.0)
# Del tomate (0) a la cámara (1). Si la hoja no alcanza el objetivo, se acerca a la cámara y tapa más.
FRACTIONS = (0.25, 0.4, 0.55)
# Centro de la hoja a menos de esto del recorrido del brazo: el brazo la atravesaría.
MIN_ARM_DISTANCE = 0.1


@dataclass(frozen=True)
class EpisodeDraw:
    target: float
    leaf: Path
    camera_pick: float     # en [0, 1): elige entre las cámaras donde se ve el tomate entero
    direction_deg: float   # hacia dónde se corre la hoja, dentro del plano de la imagen
    roll_deg: float        # giro de la hoja alrededor de la línea de vista


def draw_episode(episode: str, leaves: list[Path], targets: tuple[float, ...], seed: int = 0) -> EpisodeDraw:
    rng = np.random.default_rng([seed, int(episode)])
    return EpisodeDraw(target=float(targets[rng.integers(len(targets))]),
                       leaf=leaves[rng.integers(len(leaves))],
                       camera_pick=float(rng.random()),
                       direction_deg=float(rng.uniform(0, 360)),
                       roll_deg=float(rng.uniform(0, 360)))


def lateral_direction(view_axis, right, degrees: float) -> np.ndarray:
    """Vector unitario perpendicular a view_axis, a degrees de right (ya no hace falta que right lo sea)."""
    axis = np.asarray(view_axis, float) / np.linalg.norm(view_axis)
    x = np.asarray(right, float) - np.dot(right, axis) * axis
    x /= np.linalg.norm(x)
    angle = math.radians(degrees)
    return math.cos(angle) * x + math.sin(angle) * np.cross(axis, x)


def roll_quaternion(view_axis, degrees: float) -> tuple[float, float, float, float]:
    """wxyz del giro de degrees alrededor de view_axis."""
    axis = np.asarray(view_axis, float) / np.linalg.norm(view_axis)
    half = math.radians(degrees) / 2
    return (math.cos(half), *(float(v) * math.sin(half) for v in axis))


def fit_with_fractions(measure: Callable[[float, float], float | None], target: float,
                       fractions: tuple[float, ...] = FRACTIONS,
                       clearance: Callable[[float, float], float] | None = None) -> dict:
    """measure(fraction, offset) -> %. clearance(fraction, offset) -> distancia (m) de la hoja al
    recorrido del brazo. Prueba cada distancia hasta que la hoja alcanza el objetivo lejos del brazo.
    Devuelve el mejor ajuste con su fraction y arm_distance: primero los que no estorban al brazo,
    luego el más cercano al objetivo; si todos estorban, el más lejano al brazo."""
    def rank(fit):
        crossing = fit["arm_distance"] < MIN_ARM_DISTANCE
        return (crossing, -fit["arm_distance"] if crossing else abs(fit["occlusion"] - target))

    best = None
    for fraction in fractions:
        fit = fit_lateral_offset(lambda offset: measure(fraction, offset), target)
        fit["fraction"] = fraction
        if fit["occlusion"] is None or fit["status"] == "already_covered":
            return fit
        fit["arm_distance"] = clearance(fraction, fit["offset"]) if clearance else math.inf
        if best is None or rank(fit) < rank(best):
            best = fit
        if fit["status"] == "ok" and fit["arm_distance"] >= MIN_ARM_DISTANCE:
            break
    return best


def place_episode_leaf(server, draw: EpisodeDraw, cameras: list[str],
                       path: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Con la escena ya en el primer instante del episodio, deja puesta la hoja del sorteo.
    path: puntos del recorrido del brazo (arm_path).

    Devuelve las entradas de la hoja (formato de configs/occluders.yaml, [] si no lleva) y el
    resultado. status: ok, too_small (se usa lo más cerca que llegó), no_leaf (objetivo 0),
    already_covered (la escena ya tapa el objetivo: sin hoja) o no_camera (en ninguna cámara se ve
    el tomate entero: sin hoja)."""
    server.set_episode_occluders([])
    seen = server.measure_occlusion(cameras)
    usable = [c for c in cameras if c in seen and seen[c]["occlusion"] is not None and not seen[c]["cut_by_frame"]]
    info = {"target": draw.target, "leaf": None, "camera": None, "baseline": None, "occlusion": None}
    if draw.target <= 0:
        return [], {**info, "status": "no_leaf"}
    if not usable:
        return [], {**info, "status": "no_camera"}

    camera = usable[int(draw.camera_pick * len(usable))]
    baseline = seen[camera]["occlusion"]
    info.update(camera=camera, baseline=baseline, occlusion=baseline)
    if baseline >= draw.target:
        return [], {**info, "status": "already_covered"}

    client = server.pybullet_client
    view = server.cameras[camera]
    tomato = np.array(client.getBasePositionAndOrientation(server.apple.body_id)[0])
    view_axis = np.asarray(view.xyz, float) - tomato
    right = np.reshape(client.getMatrixFromQuaternion(client.getQuaternionFromEuler(view.rpy)), (3, 3))[:, 0]
    direction = lateral_direction(view_axis, right, draw.direction_deg)
    wxyz = roll_quaternion(view_axis, draw.roll_deg)
    position_at = lambda fraction, offset: place_between(view.xyz, tomato, fraction) + offset * direction  # noqa: E731

    leaf = {"name": draw.leaf.stem, "ply": str(draw.leaf), "position": position_at(FRACTIONS[0], 0.0).tolist(),
            "wxyz": list(wxyz), "enabled": True}
    name = server.set_episode_occluders([leaf])[0]

    def occlusion_at(fraction, offset):
        server.occluders.set_pose(name, position_at(fraction, offset), wxyz)
        return server.measure_occlusion([camera])[camera]["occlusion"]

    clearance = None
    if path is not None:
        clearance = lambda fraction, offset: path_distance(path, position_at(fraction, offset))  # noqa: E731
    fit = fit_with_fractions(occlusion_at, draw.target, clearance=clearance)
    if fit["occlusion"] is None or fit["status"] == "already_covered":
        server.set_episode_occluders([])
        return [], {**info, "status": fit["status"]}
    server.occluders.set_pose(name, position_at(fit["fraction"], fit["offset"]), wxyz)
    distance = None if path is None else round(fit["arm_distance"], 3)
    return [server.occluders.items[name].entry()], {
        **info, "status": fit["status"], "leaf": draw.leaf.stem, "occlusion": fit["occlusion"],
        "fraction": fit["fraction"], "offset": round(fit["offset"], 4), "arm_distance": distance}


def arm_points(server, per_link: int = 5) -> np.ndarray:
    """Puntos sobre el brazo en su pose actual: los orígenes de sus eslabones y entre ellos."""
    client = server.pybullet_client
    robot = server.dummy_robot
    origins = np.array([client.getLinkState(robot, i)[0] for i in range(client.getNumJoints(robot))])
    weights = np.linspace(0, 1, per_link, endpoint=False)[:, None, None]
    between = (1 - weights) * origins[:-1] + weights * origins[1:]
    return np.vstack([between.reshape(-1, 3), origins[-1:]])


def arm_path(server, frames: list[dict], step: int = 3) -> np.ndarray:
    """Dónde pasa el brazo en todo el episodio. Deja la escena en el último frame recorrido."""
    points = []
    for frame in frames[::step]:
        server.restore_state(frame)
        points.append(arm_points(server))
    return np.vstack(points)


def path_distance(path: np.ndarray, position) -> float:
    return float(np.min(np.linalg.norm(path - np.asarray(position), axis=1)))
