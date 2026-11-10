"""Episodios del planificador para el experimento de percepción: 5 poses iniciales x 4 categorías del tomate.

    python scripts/xarm5/plan_perception_episodes.py --positions 10 [--out ~/data/bc_data/perception]

Como gello_3plants, donde cada posición de tomate tiene un grupo de 5 episodios que parten de poses
distintas (reset, fuera de cuadro a cada lado y dos con el brazo en cuadro): cada posición de tomate
(un "set") da 4 categorías x 5 poses = 20 episodios, que parten de la pose A y terminan agarrando el
tomate con el planificador de plan_trajectories.py (misma trayectoria, mismo servidor, solo estado).
Las definiciones y reglas están en splatsim/utils/scene/perception_design.py.

Poses iniciales (la cámara --camera, base_rgb, es fija):
- A1 OUT_LEFT / A2 OUT_RIGHT: el brazo bajo (joint2–4 = --side-joints ± --side-jitter, la mediana de
  las demos) con joint1 girado hasta que no queda nada de él en cuadro, más --side-extra.
- A3 BASE: la pose de reset (el brazo, por encima de la imagen).
- A4 IN_VIEW_FAR / A5 IN_VIEW_NEAR: el punto de agarre a --far-distance / --near-distance m del
  tomate, en un cono de ±--start-cone-deg alrededor de la dirección tomate → base, con el brazo en
  cuadro. Se sortean por set y categoría.

Tomato: B1 en el hemisferio de configs/apple_grid.yaml, en cuadro y sin que la planta lo tape, con
las posiciones repartidas entre sets; B2 la misma, tapada por una hoja de --leaf-dir (se anota en el
meta.json y render_recordings.py la pone al renderizar); B4 corrida --float-offset m hacia la cámara;
B3 la de B4 corrida al lado hasta salir del cuadro. Una posición se acepta solo si todas sus
condiciones se pueden cumplir.

Antes de ejecutar cada episodio se comprueba su primer instante: el brazo con una máscara de PyBullet
desde la cámara y el tomate con la oclusión del splat (con el brazo fuera de cuadro). Necesita la GPU
para el splat. Los que no cumplen (o no llegan al tomate) se guardan con valid=false y los motivos, o
se descartan con --invalid skip.

Salida, como plan_trajectories.py: un episodio por carpeta (000/, 001/, ...; 20 seguidos por set),
con meta.json (set, tomato_category, arm_config, apple_position, valid, reasons, ...);
perception_episodes.csv con una fila por condición; positions.yaml y sets/<set>.json. Los sets
terminados se saltan al relanzar, y con más --positions se agregan sets lejos de los que hay. Se
renderiza después con render_recordings.py.
"""
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_trajectories import ARM_JOINTS, Kinematics, grasp_pose, plan_episode  # noqa: E402
from plan_trajectories import Args as PlanArgs  # noqa: E402
from render_recordings import make_server  # noqa: E402
from splatsim.robots.station.recorder import save_frame  # noqa: E402
from splatsim.robots.xarm.config import APPLE_HEMISPHERE, BASE_CAM_FOV, IMAGE_H, IMAGE_W  # noqa: E402
from splatsim.utils.render.camera_view import CameraView, mask_stats  # noqa: E402
from splatsim.utils.scene.apple_grid import load_random_shape, sample_hemisphere  # noqa: E402
from splatsim.utils.scene.leaf_variation import EpisodeDraw, arm_points, place_episode_leaf  # noqa: E402
from splatsim.utils.scene.perception_design import (  # noqa: E402
    ARM_CONFIGS,
    TOMATO_CATEGORIES,
    Thresholds,
    check_arm,
    check_tomato,
    farthest_points,
)
from splatsim.utils.trials.grasp_point import grasp_error, load_grasp_point  # noqa: E402

POSITIONS_FILE = "positions.yaml"
EPISODES_CSV = "perception_episodes.csv"
TOMATO_RADIUS = 0.04       # m: para descartar rápido posiciones que no pueden quedar fuera de cuadro
SEARCH_SCALE = 0.25        # resolución de la máscara del brazo al buscar poses (la comprobación es completa)
PATH_EVERY = 10            # frames: cada cuántos se toma la pose del brazo para alejar la hoja de B2


@dataclass
class Args:
    positions: int = 10
    """sets (posiciones de tomate) que debe tener la carpeta; relanzar con más agrega los que faltan"""
    out: str = "~/data/bc_data/perception"
    seed: int = 0
    robot_name: str = "xarm5_lite_plant_v3"
    camera: str = "base_rgb"
    """cámara fija con la que se comprueba la visibilidad"""
    categories: tuple[str, ...] = tuple(TOMATO_CATEGORIES)
    arm_configs: tuple[str, ...] = tuple(ARM_CONFIGS)
    leaf_dir: str | None = None
    """hojas para B2 (por defecto ~/data/occluders/<robot_name>)"""
    hidden_target: float = 98.0
    """%: oclusión que se le pide a la hoja en B2 (se acepta desde thresholds.hidden_min_occlusion)"""
    candidates: int = 600
    """posiciones del hemisferio que se prueban antes de repartir los sets"""
    frame_margin: float = 40.0
    """px: el tomate de B1 queda al menos así de lejos del borde de la imagen"""
    float_offset: tuple[float, ...] = (0.15, 0.1, 0.2)
    """m hacia la cámara para B4 (y B3), en este orden hasta que cumpla"""
    near_distance: tuple[float, float] = (0.2, 0.3)
    """m del punto de agarre al tomate en A5 (demos: 0.22–0.42, mediana 0.31)"""
    far_distance: tuple[float, float] = (0.4, 0.6)
    """m del punto de agarre al tomate en A4 (demos: 0.34–0.61, mediana 0.43)"""
    start_cone_deg: float = 35.0
    """A4/A5: ± grados de azimut alrededor de la dirección tomate → base (las demos reparten el brazo
    por toda la imagen)"""
    start_elevation_deg: tuple[float, float] = (-10.0, 35.0)
    """A4/A5: elevación del punto de partida vista desde el tomate"""
    start_pitch_deg: tuple[float, float] = (-40.0, 10.0)
    """A4/A5: inclinación de la herramienta en la pose inicial"""
    side_joints: tuple[float, float, float] = (0.63, -1.3, -0.6)
    """A1/A2: joint2–4 (mediana de las poses fuera de cuadro de gello_3plants)"""
    side_jitter: float = 0.15
    """rad ± en joint2–4 de A1/A2"""
    side_extra: tuple[float, float] = (0.1, 0.35)
    """rad que se gira joint1 más allá del primer ángulo sin brazo en cuadro (demos: |joint1| 0.8–1.3)"""
    start_attempts: int = 80
    plan_attempts: int = 10
    """planes (inclinación, velocidades, ...) que se prueban por episodio hasta que llega al tomate"""
    max_final_error: float = 0.03
    """m: error de agarre final sobre el que el episodio no llegó (como en plan_trajectories.py)"""
    invalid: Literal["mark", "skip"] = "mark"
    """mark: guarda los episodios que no cumplen con valid=false; skip: no los guarda"""
    thresholds: Thresholds = field(default_factory=Thresholds)
    # del planificador (plan_trajectories.py)
    control_hz: int = 30
    pitch_deg: tuple[float, float] = (-50.0, -5.0)
    pregrasp: tuple[float, float] = (0.06, 0.12)
    joint_speed: tuple[float, float] = (0.25, 0.5)
    approach_speed: tuple[float, float] = (0.02, 0.05)
    start_pause: tuple[float, float] = (0.2, 1.0)
    settle: tuple[float, float] = (0.0, 0.5)
    hold: tuple[float, float] = (2.0, 5.0)
    perturb: float = 0.0
    gripper: float = 0.0

    def plan_args(self) -> PlanArgs:
        keys = ("robot_name", "control_hz", "pitch_deg", "pregrasp", "joint_speed", "approach_speed",
                "start_pause", "settle", "hold", "perturb", "gripper", "seed", "max_final_error")
        return PlanArgs(**{k: getattr(self, k) for k in keys})


# ── brazo ────────────────────────────────────────────────────────────────────
class Arm:
    """Pone el brazo del servidor en una pose (sin física) y mide cómo se ve desde la cámara."""

    def __init__(self, server, kin: Kinematics, view: CameraView, point: np.ndarray, q_reset):
        self.s, self.kin, self.view, self.point = server, kin, view, point
        self.q_reset = np.asarray(q_reset, float)
        self.joints = kin.movable[:ARM_JOINTS]       # mismo URDF: mismos índices en el servidor

    def set(self, q) -> None:
        c, robot = self.s.pybullet_client, self.s.dummy_robot
        with self.s.lock:
            for i, v in zip(self.joints, q):
                c.resetJointState(robot, i, float(v))
                c.setJointMotorControl2(robot, i, c.POSITION_CONTROL, targetPosition=float(v), force=500)
            for i in range(c.getNumJoints(robot)):   # sin FK, getLinkState devolvería la pose anterior
                c.getLinkState(robot, i, computeForwardKinematics=True)

    def stats(self, q, scale: float = 1.0) -> dict:
        self.set(q)
        return mask_stats(self.view.segmentation(self.s.pybullet_client, scale) == self.s.dummy_robot)

    def measures(self, q, tomato) -> dict:
        """Lo que check_arm necesita de la pose q, con la máscara a resolución completa."""
        pos, rot = self.kin.fk(q)
        st = self.stats(q)
        return {"arm_fraction": st["fraction"], "arm": st, "ee_uv": self.view.uv_fraction(pos),
                "grasp_distance": round(float(np.linalg.norm(pos + rot @ self.point - np.asarray(tomato))), 4)}

    def can_grasp(self, tomato, pitches_deg) -> bool:
        """Hay agarre con alguna inclinación del planificador (plan_episode prueba más)."""
        return any(self.kin.ik(*grasp_pose(tomato, np.radians(p), self.point), self.q_reset) is not None
                   for p in np.linspace(*pitches_deg, 4))

    def in_view_start(self, tomato, distance, args: Args, rng) -> tuple | None:
        """A4/A5: punto de agarre a `distance` m del tomate, en el cono hacia la base, con el brazo en cuadro."""
        tomato = np.asarray(tomato, float)
        back = np.arctan2(-tomato[1], -tomato[0])          # azimut de la dirección tomate → base
        for _ in range(args.start_attempts):
            az = back + np.radians(rng.uniform(-args.start_cone_deg, args.start_cone_deg))
            el = np.radians(rng.uniform(*args.start_elevation_deg))
            start = tomato + rng.uniform(*distance) * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                                                                np.sin(el)])
            pitch = np.radians(rng.uniform(*args.start_pitch_deg))
            q = self.kin.ik(*grasp_pose(start, pitch, self.point), self.q_reset)
            if q is None or not (args.thresholds.arm_in_min
                                 <= self.stats(q, SEARCH_SCALE)["fraction"] <= args.thresholds.arm_in_max):
                continue
            return q, {"pitch_deg": round(float(np.degrees(pitch)), 1), "azimuth_offset_deg":
                       round(float(np.degrees((az - back + np.pi) % (2 * np.pi) - np.pi)), 1)}
        return None

    def left_sign(self, side_joints) -> float:
        """Signo de joint1 que lleva el brazo bajo hacia la izquierda de la imagen."""
        for sign in (1.0, -1.0):
            st = self.stats([sign * 0.3, *side_joints, 0.0], SEARCH_SCALE)
            if st["centroid"] is not None:
                return sign if st["centroid"][0] < 0.5 else -sign
        return 1.0

    def side_start(self, sign: float, args: Args, rng) -> tuple | None:
        """A1/A2: brazo bajo girado en joint1 (hacia sign) hasta que no queda nada de él en cuadro."""
        lo, hi = self.kin.arm_lower, self.kin.arm_upper
        for _ in range(10):
            mid = np.asarray(args.side_joints) + rng.uniform(-args.side_jitter, args.side_jitter, 3)
            for delta in np.arange(0.05, np.pi, 0.05):
                q = np.array([sign * delta, *mid, 0.0])
                if np.any(q < lo) or np.any(q > hi):
                    break
                if self.stats(q, SEARCH_SCALE)["pixels"]:
                    continue
                q[0] = float(np.clip(sign * (delta + rng.uniform(*args.side_extra)), lo[0], hi[0]))
                if self.stats(q)["pixels"]:
                    break                              # más allá vuelve a entrar: otro sorteo
                return q, {"joint1": round(float(q[0]), 3)}
        return None



# ── posiciones del tomate ────────────────────────────────────────────────────
def tomato_out_of_frame(view: CameraView, pos) -> bool:
    """Ningún punto del contorno del tomate (visto desde la cámara) cae en el cuadro."""
    rot = view.rotation
    pts = [np.asarray(pos) + TOMATO_RADIUS * s * rot[:, k] for k in (0, 1) for s in (-1, 1)] + [pos]
    return not view.in_frame(pts).any()


class Probe:
    """Comprobaciones de una posición de tomate antes de elegirla, las mismas que se hacen al planear
    sus episodios: así casi nada se marca después."""

    def __init__(self, server, arm: Arm, args: Args):
        self.s, self.arm, self.args = server, arm, args
        self.rng = np.random.default_rng([args.seed, 7])

    def reachable(self, pos) -> bool:
        """Hay agarre y poses A4/A5 con el brazo en cuadro."""
        a = self.args
        return (self.arm.can_grasp(pos, a.pitch_deg)
                and self.arm.in_view_start(pos, a.near_distance, a, self.rng) is not None
                and self.arm.in_view_start(pos, a.far_distance, a, self.rng) is not None)

    def scene(self, pos, leaf: list | None = None) -> dict:
        """Oclusión del tomate en pos por la escena (y la hoja), con el brazo en reset, fuera de cuadro."""
        self.s.set_episode_occluders(leaf or [])
        self.s.apple.move_to(list(pos))
        self.arm.set(self.arm.q_reset)
        return self.s.measure_occlusion([self.args.camera])[self.args.camera]

    def visible(self, pos) -> bool:
        return not check_tomato("B1", self.scene(pos), self.args.thresholds)

    def out_of_frame(self, pos) -> bool:
        """Ni un píxel del tomate en cuadro (el contorno geométrico no basta: el splat es más grande)."""
        return tomato_out_of_frame(self.arm.view, pos) and self.scene(pos)["silhouette_px"] < 1.0


def derive_positions(view: CameraView, anchor, probe: Probe, offsets) -> dict | None:
    """B4 y B3 a partir del ancla (B1/B2), o None si ninguna cumple."""
    anchor = np.asarray(anchor, float)
    to_cam = np.asarray(view.xyz) - anchor
    to_cam /= np.linalg.norm(to_cam)
    right = view.rotation[:, 0]                     # hacia la derecha de la imagen
    for offset in offsets:
        floating = anchor + offset * to_cam
        if not (probe.visible(floating) and probe.reachable(floating)):
            continue
        u = view.project([floating])[0][0, 0]
        sides = (1, -1) if u >= view.width / 2 else (-1, 1)   # el borde más cercano primero
        for shift in np.arange(0.04, 0.8, 0.02):
            for side in sides:
                outside = floating + side * shift * right
                if tomato_out_of_frame(view, outside) and probe.out_of_frame(outside) and probe.reachable(outside):
                    return {"B3": outside.tolist(), "B4": floating.tolist(), "float_offset": offset}
    return None


def choose_positions(args: Args, view: CameraView, probe: Probe, count: int, taken: list, rng) -> list[dict]:
    """count sets nuevos: B1 en cuadro, visible y al alcance, repartidos y lejos de los B1 de taken."""
    _, hemisphere = load_random_shape(APPLE_HEMISPHERE)
    pool = []
    for _ in range(args.candidates):
        pos = np.array(sample_hemisphere(hemisphere, rng=rng))
        if view.in_frame([pos], args.frame_margin)[0] and probe.arm.can_grasp(pos, args.pitch_deg):
            pool.append(pos)
    if not pool:
        raise SystemExit("Ninguna posición del hemisferio queda en cuadro y al alcance: revisa el hemisferio "
                         "(configs/apple_grid.yaml) o la pose de la cámara")
    print(f"{len(pool)} de {args.candidates} posiciones del hemisferio en cuadro y al alcance", flush=True)
    sets, rejected = [], 0
    for i in farthest_points(pool, len(pool), rng, taken=taken):
        derived = probe.visible(pool[i]) and probe.reachable(pool[i]) and \
            derive_positions(view, pool[i], probe, args.float_offset)
        if not derived:
            rejected += 1
            continue
        sets.append({"B1": pool[i].tolist(), "B2": pool[i].tolist(), **derived})
        if len(sets) == count:
            break
    print(f"{len(sets)} sets elegidos ({rejected} posiciones descartadas: tapadas por la planta, sin poses "
          "A4/A5 o sin B3/B4)", flush=True)
    return sets


def load_or_choose_positions(args: Args, out: Path, view: CameraView, probe: Probe) -> list[dict]:
    """Las de positions.yaml y, si faltan, nuevas lejos de ellas."""
    path = out / POSITIONS_FILE
    sets = (yaml.safe_load(path.read_text()) or {}).get("sets", []) if path.exists() else []
    if len(sets) < args.positions:
        rng = np.random.default_rng([args.seed, len(sets)])
        sets += choose_positions(args, view, probe, args.positions - len(sets), [s["B1"] for s in sets], rng)
        path.write_text(yaml.safe_dump({"camera": args.camera, "sets": sets}, sort_keys=False))
    if len(sets) < args.positions:
        print(f"Solo {len(sets)} posiciones cumplen todo (pedidas {args.positions})", flush=True)
    return sets[:args.positions]


# ── episodios ────────────────────────────────────────────────────────────────
def execute_from(server, arm: Arm, q_start, plan: dict, tomato, folder: Path) -> np.ndarray:
    """Como plan_trajectories.execute, pero el brazo parte de q_start en vez de la pose de reset."""
    server.reset_robot()                 # cierra el gripper y deja el próximo comando partiendo de donde esté
    server.apple.move_to(list(tomato))
    if server.peduncle is not None:
        server.peduncle.reset()
    arm.set(q_start)
    states = []
    for k, (q, gripper) in enumerate(zip(plan["q"], plan["gripper"]), 1):
        server.command_joint_state(np.array([*q, gripper]))
        obs = server.state_observation()
        save_frame(str(folder), k, obs)
        states.append(obs["state"])
    return np.array(states)


def plan_path_points(server, arm: Arm, plans: list[dict]) -> np.ndarray | None:
    """Puntos por donde pasa el brazo en los episodios (para alejar de ahí la hoja de B2)."""
    points = []
    for plan in plans:
        for q in plan["q"][::PATH_EVERY]:
            arm.set(q)
            points.append(arm_points(server))
    return np.vstack(points) if points else None


def place_leaf(server, args: Args, leaves: list[Path], tomato, path, rng) -> tuple[list, dict]:
    """La hoja que más tapa al tomate en --camera (para en la primera que llega al mínimo)."""
    server.apple.move_to(list(tomato))
    best = ([], {"status": "no_leaf", "occlusion": None})
    for leaf in rng.permutation(leaves):
        draw = EpisodeDraw(target=args.hidden_target, leaf=Path(leaf), camera_pick=0.0,
                           direction_deg=float(rng.uniform(0, 360)), roll_deg=float(rng.uniform(0, 360)))
        entries, fit = place_episode_leaf(server, draw, [args.camera], path)
        if fit.get("occlusion") is not None and (best[1]["occlusion"] is None
                                                 or fit["occlusion"] > best[1]["occlusion"]):
            best = (entries, fit)
        if fit.get("occlusion") is not None and fit["occlusion"] >= args.thresholds.hidden_min_occlusion:
            break
    server.set_episode_occluders(best[0])
    return best


CSV_FIELDS = ["episode", "set", "tomato_category", "arm_config", "arm_config_name", "tomato_x", "tomato_y",
              "tomato_z", "valid", "reasons", "arm_fraction", "ee_u", "ee_v", "grasp_distance",
              "tomato_occlusion_scene", "tomato_occlusion_start", "leaf", "frames", "final_grasp_error"]


class Episodes:
    """Numeración de los episodios de la carpeta, como plan_trajectories.py: después del mayor."""

    def __init__(self, out: Path):
        self.out = out
        self.next = max((int(e.name) for e in out.iterdir() if e.is_dir() and e.name.isdigit()), default=-1) + 1

    def take(self) -> str:
        name = f"{self.next:03d}"
        self.next += 1
        return name

    def drop_set(self, k: int) -> int:
        """Borra los episodios de un set que quedó a medias (sin sets/<k>.json). Devuelve cuántos."""
        dropped = 0
        for e in self.out.iterdir():
            meta = e / "meta.json"
            if e.is_dir() and e.name.isdigit() and meta.exists() and json.loads(meta.read_text()).get("set") == k:
                shutil.rmtree(e)
                dropped += 1
        return dropped


def run_set(server, arm: Arm, probe: Probe, args: Args, k: int, positions: dict, leaves: list[Path],
            episodes: Episodes, writer, left: float) -> tuple[int, int]:
    """Los 20 episodios de un set. Devuelve (guardados, válidos)."""
    rng = np.random.default_rng([args.seed, 1000 + k])
    plan_args, point, t = args.plan_args(), arm.point, args.thresholds
    sides = {"A1": arm.side_start(left, args, rng), "A2": arm.side_start(-left, args, rng)}
    set_info = {"set": k, "camera": args.camera, "positions": positions, "categories": {}}
    saved = valid_count = 0
    for b in args.categories:
        tomato = np.asarray(positions[b], float)
        starts = {**sides, "A3": (arm.q_reset, {}),
                  "A4": arm.in_view_start(tomato, args.far_distance, args, rng),
                  "A5": arm.in_view_start(tomato, args.near_distance, args, rng)}
        plans = {}
        for a in args.arm_configs:
            if starts[a] is not None:
                for _ in range(args.plan_attempts):
                    plan = plan_episode(arm.kin, starts[a][0], tomato, point, plan_args, rng)
                    if plan is not None:
                        plans.setdefault(a, []).append(plan)
                        break
        leaf_entries, leaf_fit = [], None
        if b == "B2":
            path = plan_path_points(server, arm, [p[0] for p in plans.values()])
            arm.set(arm.q_reset)
            leaf_entries, leaf_fit = place_leaf(server, args, leaves, tomato, path, rng)
        scene = probe.scene(tomato, leaf_entries)
        tomato_reasons = check_tomato(b, scene, t)
        set_info["categories"][b] = {"scene_occlusion": scene, "leaf": leaf_entries, "leaf_fit": leaf_fit}

        for a in args.arm_configs:
            if starts[a] is None:
                reasons, m, with_arm = [f"sin pose inicial {a} para este tomate"], None, None
            else:
                m = arm.measures(starts[a][0], tomato)
                with_arm = server.measure_occlusion([args.camera])[args.camera]   # el brazo quedó en la pose
                reasons = tomato_reasons + check_arm(a, m, t)
            row = {"set": f"{k:03d}", "tomato_category": b, "arm_config": a, "arm_config_name": ARM_CONFIGS[a][0],
                   **{f"tomato_{x}": round(float(v), 4) for x, v in zip("xyz", tomato)},
                   "tomato_occlusion_scene": "" if scene["occlusion"] is None else scene["occlusion"],
                   "leaf": leaf_entries[0]["name"] if leaf_entries else ""}
            if m is not None:
                row.update(arm_fraction=m["arm_fraction"], grasp_distance=m["grasp_distance"],
                           ee_u=m["ee_uv"][0] if m["ee_uv"] else "", ee_v=m["ee_uv"][1] if m["ee_uv"] else "",
                           tomato_occlusion_start="" if with_arm["occlusion"] is None else with_arm["occlusion"])
            if reasons and args.invalid == "skip":
                writer.writerow({**row, "valid": False, "reasons": "; ".join(reasons)})
                print(f"  set {k:03d} {b}_{a}: descartado ({'; '.join(reasons)})", flush=True)
                continue

            name, folder, states, plan, final_err = None, None, None, None, None
            for attempt in range(args.plan_attempts if a in plans else 0):
                plan = plans[a][0] if attempt == 0 else plan_episode(arm.kin, starts[a][0], tomato, point,
                                                                     plan_args, rng)
                if plan is None:
                    continue
                folder = episodes.out / f".{k:03d}_{b}_{a}.tmp"
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir()
                states = execute_from(server, arm, starts[a][0], plan, tomato, folder)
                final_err = float(grasp_error(states, tomato, point)[plan["target_end"] - 1])
                if final_err <= args.max_final_error:
                    break
            if plan is None or states is None:
                reasons.append("el planificador no encontró trayectoria hasta el tomate")
            elif final_err > args.max_final_error:
                reasons.append(f"no llegó al tomate (error final {final_err * 100:.1f} cm)")
            valid = not reasons
            if folder is not None and (valid or args.invalid == "mark"):
                name = episodes.take()
                meta = {"source": "planner_perception", "set": k, "tomato_category": b, "arm_config": a,
                        "arm_config_name": ARM_CONFIGS[a][0], "apple_position": tomato.tolist(),
                        "tomato_visible": TOMATO_CATEGORIES[b][0], "tomato_attached": TOMATO_CATEGORIES[b][1],
                        "valid": valid, "reasons": reasons, "camera": args.camera,
                        "start": {**{kk: v for kk, v in (m or {}).items()}, "q": np.round(starts[a][0], 5).tolist(),
                                  **starts[a][1], "tomato_with_arm": with_arm},
                        "tomato_scene": scene, "occluders": leaf_entries, "leaf_fit": leaf_fit,
                        "pitch_deg": round(plan["pitch_deg"], 2), "pregrasp": round(plan["pregrasp"], 4),
                        "phases": plan["phases"], "final_grasp_error": round(final_err, 4),
                        "seed": args.seed, "target": "tomato"}
                (folder / "meta.json").write_text(json.dumps(meta, indent=2))
                os.rename(folder, episodes.out / name)
                saved += 1
                valid_count += valid
                row.update(frames=len(states), final_grasp_error=round(final_err, 4))
            elif folder is not None:
                shutil.rmtree(folder)
            writer.writerow({**row, "episode": name or "", "valid": valid, "reasons": "; ".join(reasons)})
            print(f"  set {k:03d} {b}_{a} → {name or '-'}: {'ok' if valid else 'MARCADO: ' + '; '.join(reasons)}",
                  flush=True)
    sets_dir = episodes.out / "sets"
    sets_dir.mkdir(exist_ok=True)
    (sets_dir / f"{k:03d}.json").write_text(json.dumps(set_info, indent=2))
    return saved, valid_count


def main(args: Args):
    unknown = (set(args.categories) - set(TOMATO_CATEGORIES)) | (set(args.arm_configs) - set(ARM_CONFIGS))
    if unknown:
        raise SystemExit(f"Desconocidas: {sorted(unknown)}")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    leaves = []
    if "B2" in args.categories:
        leaf_dir = Path(args.leaf_dir or f"~/data/occluders/{args.robot_name}").expanduser()
        leaves = sorted(leaf_dir.glob("*.ply"))
        if not leaves:
            raise SystemExit(f"B2 necesita hojas y no hay en {leaf_dir}: recorta alguna con segment_leaf.py "
                             "o pasa --leaf-dir (o quita B2 de --categories)")

    server = make_server(args.robot_name, [args.camera], control_hz=args.control_hz)
    cam = server.cameras[args.camera]
    view = CameraView(tuple(cam.xyz), tuple(cam.rpy), BASE_CAM_FOV, IMAGE_W, IMAGE_H)
    server.reset_robot()
    q_reset = np.asarray(server.state_observation()["joint_positions"][:ARM_JOINTS], float)
    arm = Arm(server, Kinematics(server.urdf_path), view, load_grasp_point(), q_reset)
    probe = Probe(server, arm, args)
    left = arm.left_sign(args.side_joints)
    sets = load_or_choose_positions(args, out, view, probe)

    info_path = out / "perception_info.yaml"
    runs = (yaml.safe_load(info_path.read_text()) or {}).get("runs", []) if info_path.exists() else []
    runs.append({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "camera_pose": {"xyz": list(cam.xyz), "rpy": list(cam.rpy)}, "q_reset": q_reset.round(5).tolist(),
                 "left_joint1_sign": left, "grasp_point": arm.point.tolist(), **json.loads(json.dumps(asdict(args)))})
    info_path.write_text(yaml.safe_dump({"runs": runs}, sort_keys=False, allow_unicode=True))

    episodes = Episodes(out)
    csv_path = out / EPISODES_CSV
    new_csv = not csv_path.exists()
    total = valid = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()
        for k, positions in enumerate(sets):
            if (out / "sets" / f"{k:03d}.json").exists():
                continue
            dropped = episodes.drop_set(k)
            print(f"[{k + 1}/{len(sets)}] set {k:03d}: tomato {np.round(positions['B1'], 3).tolist()}"
                  + (f" (se borran {dropped} episodios de un intento anterior)" if dropped else ""), flush=True)
            s, v = run_set(server, arm, probe, args, k, positions, leaves, episodes, writer, left)
            f.flush()
            total, valid = total + s, valid + v
    print(f"Listo: {total} episodios nuevos ({valid} válidos, {total - valid} marcados) en {out}. "
          f"Renderiza con: python scripts/xarm5/render_recordings.py {out} --fps 10", flush=True)


if __name__ == "__main__":
    # como plan_trajectories.py: el servidor deja hilos que no terminan solos
    try:
        main(tyro.cli(Args))
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"código {e.code}", flush=True)
            os._exit(1 if isinstance(e.code, str) else int(e.code))
    os._exit(0)
