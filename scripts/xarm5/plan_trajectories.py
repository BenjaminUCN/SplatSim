"""Genera demostraciones sintéticas: el brazo va hasta el tomate con un planificador, sin GELLO.

    python scripts/xarm5/plan_trajectories.py --episodes 100 [--out ~/data/bc_data/planned] [--seed 0]

Cada episodio pone el tomate en una posición aleatoria (la forma guardada en
configs/apple_grid.yaml, hemisferio o caja; lejos de las posiciones de la grilla, que quedan para
los ensayos, y a no más de --max-azimuth del frente) o en la grilla (--positions grid), y planifica como terminan las demos humanas de
gello_3plants: el eje de la herramienta apunta al tomate en el plano del brazo (joint5 ≈ 0) con
una inclinación hacia abajo al azar (--pitch-deg), y el tomate queda en el punto de agarre
(configs/grasp_point.yaml). Fases: pausa inicial, ir en espacio articular (mínimo jerk) a una
pose previa a unos cm hacia atrás sobre el eje de la herramienta, avanzar en línea recta hasta el
agarre y sostener. Velocidades, distancia previa y pausas se sortean por episodio; --perturb
agrega una desviación suave a la ida que se corrige antes de llegar (muestra recuperaciones).

La trayectoria se ejecuta en el servidor del xArm5 (misma física y control articular que en la
teleoperación, sim_clock=fixed a --control-hz) sin cámaras ni GPU, y se graba solo el estado, un
frame por comando, igual que con GELLO en realtime: se renderiza después con
render_recordings.py (o la pestaña Data de train_server.py). Un episodio cuyo error de agarre
final supera --max-final-error se descarta y se prueba otra posición. Los episodios se agregan
después del mayor número que ya tenga la carpeta; cada uno guarda su plan en meta.json y la
carpeta, los parámetros en plan_info.yaml.
"""
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pybullet as p
import tyro
import yaml
from pybullet_utils.bullet_client import BulletClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from render_recordings import make_server  # noqa: E402
from splatsim.robots.station.recorder import save_frame  # noqa: E402
from splatsim.robots.xarm.config import APPLE_HEMISPHERE, APPLE_RANGE  # noqa: E402
from splatsim.utils.apple_grid import load_grid, load_random_range, load_random_shape, sample_hemisphere  # noqa: E402
from splatsim.utils.grasp_point import grasp_error, load_grasp_point  # noqa: E402

PLAN_INFO_FILE = "plan_info.yaml"
EE_LINK = 6          # link_eef: el de obs["state"]
ARM_JOINTS = 5


@dataclass
class Args:
    episodes: int = 50
    """episodios a generar (se agregan a los que ya tenga la carpeta)"""
    out: str = "~/data/bc_data/planned"
    seed: int | None = None
    positions: Literal["random", "grid"] = "random"
    """random: la forma guardada en apple_grid.yaml (hemisferio o caja); grid: las de la grilla"""
    exclude_grid: float = 0.05
    """m: con random, se descartan posiciones a menos de esto de una de la grilla (0 = ninguna)"""
    max_azimuth: float = 30.0
    """°: con random, se descartan posiciones que se apartan más que esto del frente del robot
    (el facing del hemisferio; 180 = −x, hacia las plantas). Las demos llegan a ±23° y la grilla
    a ±20°; más allá el tomate queda fuera de la escena y de la cámara base. 180 = sin límite"""
    robot_name: str = "xarm5_lite_plant_v3"
    control_hz: int = 30
    pitch_deg: tuple[float, float] = (-50.0, -5.0)
    """inclinación del eje de la herramienta bajo la horizontal (demos: mediana −20°)"""
    pregrasp: tuple[float, float] = (0.06, 0.12)
    """m hacia atrás sobre el eje de la herramienta donde termina la ida"""
    joint_speed: tuple[float, float] = (0.25, 0.5)
    """rad/s medios del joint que más se mueve en la ida"""
    approach_speed: tuple[float, float] = (0.02, 0.05)
    """m/s medios del avance final en línea recta"""
    start_pause: tuple[float, float] = (0.2, 1.0)
    settle: tuple[float, float] = (0.0, 0.5)
    """s quieto en la pose previa antes de avanzar"""
    hold: tuple[float, float] = (2.0, 5.0)
    """s sosteniendo en el agarre al final"""
    perturb: float = 0.0
    """rad: amplitud de la desviación suave de la ida (0 = sin desviación)"""
    gripper: float = 0.0
    """comando del gripper durante todo el episodio (0 = cerrado, como en las demos)"""
    max_final_error: float = 0.02
    """m: error de agarre final sobre el que se descarta el episodio"""
    max_attempts: int = 30
    """posiciones a probar por episodio antes de rendirse"""


# ── cinemática ───────────────────────────────────────────────────────────────
class Kinematics:
    """IK y FK del brazo en un PyBullet propio (DIRECT), sin tocar la física del servidor."""

    def __init__(self, urdf_path: str):
        self.c = BulletClient(connection_mode=p.DIRECT)
        self.body = self.c.loadURDF(urdf_path, useFixedBase=True)
        self.movable = [i for i in range(self.c.getNumJoints(self.body))
                        if self.c.getJointInfo(self.body, i)[2] != p.JOINT_FIXED]
        info = [self.c.getJointInfo(self.body, i) for i in self.movable]
        self.lower = [j[8] for j in info]
        self.upper = [j[9] for j in info]
        self.arm_lower = np.array(self.lower[:ARM_JOINTS])
        self.arm_upper = np.array(self.upper[:ARM_JOINTS])

    def set(self, q):
        for i, v in zip(self.movable, q):
            self.c.resetJointState(self.body, i, v)

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Posición y matriz de rotación de link_eef."""
        self.set(list(q) + [0.0] * (len(self.movable) - ARM_JOINTS))
        s = self.c.getLinkState(self.body, EE_LINK, computeForwardKinematics=True)
        return np.array(s[4]), np.array(self.c.getMatrixFromQuaternion(s[5])).reshape(3, 3)

    def ik(self, pos, rot, seed) -> np.ndarray | None:
        """Joints del brazo que llevan link_eef a (pos, rot) partiendo de seed, o None si no
        llega a menos de 3 mm y 3° o se sale de los límites."""
        quat = _quat_from_matrix(rot)
        q = np.array(seed, dtype=float)
        for _ in range(8):     # IK de PyBullet desde la solución anterior hasta que converja
            self.set(list(q) + [0.0] * (len(self.movable) - ARM_JOINTS))
            sol = self.c.calculateInverseKinematics(
                self.body, EE_LINK, pos, quat, lowerLimits=self.lower, upperLimits=self.upper,
                jointRanges=[u - lo for lo, u in zip(self.lower, self.upper)],
                restPoses=list(q) + [0.0] * (len(self.movable) - ARM_JOINTS),
                maxNumIterations=200, residualThreshold=1e-6)
            q = np.array(sol[:ARM_JOINTS])
        fpos, frot = self.fk(q)
        angle = np.degrees(np.arccos(np.clip((np.trace(frot.T @ rot) - 1) / 2, -1, 1)))
        if np.linalg.norm(fpos - pos) > 0.003 or angle > 3.0:
            return None
        if np.any(q < self.arm_lower) or np.any(q > self.arm_upper):
            return None
        return q


def _quat_from_matrix(m) -> list[float]:
    from scipy.spatial.transform import Rotation
    return list(Rotation.from_matrix(m).as_quat())   # xyzw, como PyBullet


def tool_rotation(azimuth: float, pitch: float) -> np.ndarray:
    """Eje z de la herramienta con ese azimut e inclinación (rad), x en el plano vertical del
    brazo (roll de las demos, joint5 ≈ 0)."""
    z = np.array([np.cos(pitch) * np.cos(azimuth), np.cos(pitch) * np.sin(azimuth), np.sin(pitch)])
    h = np.cross([0.0, 0.0, 1.0], z)
    h /= np.linalg.norm(h)
    x = np.cross(z, h)
    return np.column_stack([x, np.cross(z, x), z])


def grasp_pose(tomato, pitch: float, point) -> tuple[np.ndarray, np.ndarray]:
    """Pose de link_eef que deja el tomate en el punto de agarre, con el eje de la herramienta
    en el plano vertical que pasa por la base (el único alcanzable con 5 joints)."""
    tomato = np.asarray(tomato, dtype=float)
    az = np.arctan2(tomato[1], tomato[0])
    for _ in range(5):         # el plano del brazo es el del efector, no el del tomate
        rot = tool_rotation(az, pitch)
        pos = tomato - rot @ point
        az = np.arctan2(pos[1], pos[0])
    return pos, rot


# ── trayectoria ──────────────────────────────────────────────────────────────
def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def plan_episode(kin: Kinematics, q0: np.ndarray, tomato, point, args: Args, rng) -> dict | None:
    """Trayectoria articular a control_hz hacia el tomate, o None si no hay solución."""
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    for _ in range(6):         # otras inclinaciones si esta no alcanza
        pitch = np.radians(u(args.pitch_deg))
        pos, rot = grasp_pose(tomato, pitch, point)
        back = u(args.pregrasp)
        pre_pos = pos - back * rot[:, 2]
        q_pre = kin.ik(pre_pos, rot, q0)
        if q_pre is None:
            continue
        # avance en línea recta: IK cada ~5 mm, cada una desde la anterior
        n = max(2, int(np.ceil(back / 0.005)) + 1)
        path, q = [], q_pre
        for s in np.linspace(0, 1, n):
            q = kin.ik(pre_pos + s * (pos - pre_pos), rot, q)
            if q is None:
                break
            path.append(q)
        else:
            break
    else:
        return None

    hz = args.control_hz
    path = np.array(path)
    reach_t = max(float(np.max(np.abs(q_pre - q0))) / u(args.joint_speed), 1.0)
    approach_t = max(back / u(args.approach_speed), 0.5)
    phases = {"start_pause": u(args.start_pause), "reach": reach_t, "settle": u(args.settle),
              "approach": approach_t, "hold": u(args.hold)}

    def steps(seconds):
        return max(1, int(round(seconds * hz)))

    s = min_jerk(np.linspace(0, 1, steps(reach_t)))
    reach = q0 + s[:, None] * (q_pre - q0)
    if args.perturb > 0:       # desviación suave que vale 0 al salir y al llegar a la pose previa
        t = np.linspace(0, 1, len(reach))
        envelope = np.sin(np.pi * t) ** 2
        for j in range(ARM_JOINTS):
            freq, phase = rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi)
            reach[:, j] += args.perturb * rng.uniform(-1, 1) * envelope * np.sin(2 * np.pi * freq * t + phase)
    # avance: mínimo jerk a lo largo de los puntos de la recta
    idx = min_jerk(np.linspace(0, 1, steps(approach_t))) * (len(path) - 1)
    approach = np.column_stack([np.interp(idx, np.arange(len(path)), path[:, j]) for j in range(ARM_JOINTS)])
    q = np.concatenate([np.repeat(q0[None], steps(phases["start_pause"]), 0), reach,
                        np.repeat(q_pre[None], steps(phases["settle"]), 0) if phases["settle"] > 0
                        else np.zeros((0, ARM_JOINTS)),
                        approach, np.repeat(path[-1][None], steps(phases["hold"]), 0)])
    return {"q": q, "pitch_deg": float(np.degrees(pitch)), "pregrasp": back,
            "phases": {k: round(v, 3) for k, v in phases.items()}}


# ── posiciones del tomate ────────────────────────────────────────────────────
class Positions:
    def __init__(self, args: Args, rng):
        self.args, self.rng = args, rng
        self.grid = np.array(load_grid() or np.zeros((0, 3)))
        self.shape, self.hemisphere = load_random_shape(APPLE_HEMISPHERE)
        self.range = load_random_range(APPLE_RANGE)
        self.facing = float(self.hemisphere.get("facing", 180.0))   # frente del robot (°)
        self._next = 0
        if args.positions == "grid" and not len(self.grid):
            raise SystemExit("La grilla (configs/apple_grid.yaml) está vacía")

    def sample(self) -> list[float]:
        if self.args.positions == "grid":
            pos = self.grid[self._next % len(self.grid)]
            self._next += 1
            return [float(v) for v in pos]
        for _ in range(1000):
            if self.shape == "hemisphere":
                pos = np.array(sample_hemisphere(self.hemisphere, rng=self.rng))
            else:
                pos = np.array([self.rng.uniform(*self.range[a]) for a in "xyz"])
            off = (np.degrees(np.arctan2(pos[1], pos[0])) - self.facing + 180) % 360 - 180
            if abs(off) > self.args.max_azimuth:
                continue
            if not len(self.grid) or self.args.exclude_grid <= 0 or \
                    np.min(np.linalg.norm(self.grid - pos, axis=1)) >= self.args.exclude_grid:
                return [float(v) for v in pos]
        raise SystemExit("No se encontró una posición en el rango: sube --max-azimuth o baja --exclude-grid")


# ── ejecución ────────────────────────────────────────────────────────────────
def execute(server, plan: dict, tomato, gripper: float, folder: Path) -> np.ndarray:
    """Corre la trayectoria en el servidor (un comando por frame) y graba el estado; devuelve
    los state de cada frame."""
    server.reset_robot()
    server.apple.move_to(tomato)
    states = []
    for k, q in enumerate(plan["q"], 1):
        server.command_joint_state(np.array([*q, gripper]))
        obs = server.state_observation()
        save_frame(str(folder), k, obs)
        states.append(obs["state"])
    return np.array(states)


def main(args: Args):
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    point = load_grasp_point()
    positions = Positions(args, rng)
    existing = [int(e.name) for e in out.iterdir() if e.is_dir() and e.name.isdigit()]
    first = max(existing, default=-1) + 1
    info = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "grasp_point": point.tolist(),
            **asdict(args)}
    runs = []
    if (out / PLAN_INFO_FILE).exists():
        runs = (yaml.safe_load((out / PLAN_INFO_FILE).read_text()) or {}).get("runs", [])
    runs.append({**info, "first_episode": first})
    with open(out / PLAN_INFO_FILE, "w") as f:
        yaml.safe_dump({"runs": runs}, f, sort_keys=False, allow_unicode=True)

    server = make_server(args.robot_name, [], control_hz=args.control_hz)
    server.reset_robot()
    q0 = np.asarray(server.state_observation()["joint_positions"][:ARM_JOINTS])
    kin = Kinematics(server.urdf_path)
    print(f"{args.episodes} episodios → {out} (desde {first:03d}), posiciones {args.positions}"
          + (f" ({positions.shape})" if args.positions == "random" else ""), flush=True)

    rejected = 0
    for n in range(args.episodes):
        name = f"{first + n:03d}"
        start = time.time()
        for attempt in range(args.max_attempts):
            tomato = positions.sample()
            plan = plan_episode(kin, q0, tomato, point, args, rng)
            if plan is None:
                rejected += 1
                continue
            tmp = out / f".{name}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            states = execute(server, plan, tomato, args.gripper, tmp)
            err = grasp_error(states, tomato, point)
            if err[-1] > args.max_final_error:
                shutil.rmtree(tmp)
                rejected += 1
                continue
            with open(tmp / "meta.json", "w") as f:
                json.dump({"source": "planner", "apple_position": tomato, "attempt": attempt,
                           "pitch_deg": round(plan["pitch_deg"], 2), "pregrasp": round(plan["pregrasp"], 4),
                           "phases": plan["phases"], "final_grasp_error": round(float(err[-1]), 4),
                           "seed": args.seed}, f, indent=2)
            os.rename(tmp, out / name)
            print(f"[{n + 1}/{args.episodes}] {name}: {len(states)} frames "
                  f"({len(states) / args.control_hz:.1f} s), inclinación {plan['pitch_deg']:.0f}°, "
                  f"error final {err[-1] * 100:.1f} cm, {time.time() - start:.1f} s", flush=True)
            break
        else:
            raise SystemExit(f"{name}: ninguna de {args.max_attempts} posiciones dio una trayectoria; "
                             "revisa el rango del tomate o --pitch-deg")
    print(f"Listo: {args.episodes} episodios en {out} ({rejected} posiciones descartadas)", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
    os._exit(0)  # el servidor deja hilos (teclado, PyBullet) que no terminan solos
