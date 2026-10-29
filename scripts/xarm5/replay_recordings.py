"""Reproduce episodios grabados en el robot (real o sim), en uno o varios modos de control, para
comparar cómo sigue cada modo (position / interp / velocity) la misma trayectoria.

Primero lanzar el servidor (real o sim) y luego, p.ej.:
    python scripts/xarm5/replay_recordings.py ~/data/bc_data/real_fewshot --episodes 0 2 --hz 10

Por cada episodio y cada modo: cambia el modo del servidor, Reset Robot, lleva el brazo despacio a
la primera pose del episodio, y manda la trayectoria a --hz (remuestreada a instantes k/hz desde los
tiempos grabados), grabándola como un episodio normal. Se reproduce lo que pidió el cliente al
grabar (joint_command, p.ej. el GELLO); las grabaciones anteriores a ese campo solo tienen la
posición alcanzada (joint_positions), que se usa en su lugar.

En el robot real, antes de cada reproducción el panel pregunta al operador (caja Operator: Start /
Skip / Stop), para reponer el tomate y tener la parada de emergencia a mano.

Salida en <folder>_replay/<fecha>/: una carpeta por modo con los episodios grabados (meta.json con
el episodio de origen) y summary.csv con el error de seguimiento de las 5 articulaciones respecto
de la trayectoria de origen: RMS, máximo, final y el retraso que mejor las alinea.
"""
import csv
import datetime
import glob
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import tyro
from gello.zmq_core.robot_node import ZMQClientRobot

from splatsim.robots.joint_controller import MODES
from splatsim.utils.remote_call import ask_operator, call

ARM = slice(0, 5)        # articulaciones del brazo (la 6ª es el gripper)
SETTLE_SECONDS = 0.5     # s quieto en la primera pose antes de empezar
MAX_LAG = 0.5            # s: retraso máximo buscado al alinear la reproducción con el origen


@dataclass
class Args:
    folder: tyro.conf.Positional[str]      # carpeta de grabaciones (000/, 001/, ...) o un episodio
    episodes: tuple[int, ...] | None = None  # números de episodio (por defecto todos)
    modes: tuple[str, ...] = tuple(MODES)
    hz: int | None = None                  # frecuencia de comandos; None -> control_hz del servidor
    record: bool = True                    # grabar cada reproducción
    output_dir: str | None = None          # None -> <folder>_replay/<fecha>
    ask: bool | None = None                # --ask True/False: preguntar al operador antes de cada una; None -> solo real
    reset: bool = True                     # Reset Robot antes de cada reproducción
    approach_speed: float = 0.3            # rad/s hasta la primera pose
    max_approach: float = 1.0              # rad: más lejos que esto no se mueve (¿falta Reset Robot?)
    hold_seconds: float = 1.0              # s sosteniendo la última pose al terminar (sigue grabando)
    robot_port: int = 6001
    hostname: str = "127.0.0.1"


def find_episodes(folder: str, numbers) -> list[str]:
    folder = os.path.expanduser(folder)
    if glob.glob(os.path.join(folder, "*.pkl")):
        return [folder]                    # un episodio
    found = {int(d): os.path.join(folder, d) for d in os.listdir(folder)
             if d.isdigit() and os.path.isdir(os.path.join(folder, d))}
    if numbers is None:
        return [found[n] for n in sorted(found)]
    missing = [n for n in numbers if n not in found]
    if missing:
        raise SystemExit(f"{folder} no tiene los episodios {missing}")
    return [found[n] for n in numbers]


def load_reference(episode: str, hz: float):
    """(tiempos, joints 6) del episodio desde 0 s, y si es lo comandado (True) o lo alcanzado."""
    times, joints, commanded = [], [], True
    for k, path in enumerate(sorted(glob.glob(os.path.join(episode, "*.pkl")))):
        with open(path, "rb") as f:
            obs = pickle.load(f)
        if "joint_command" not in obs:
            commanded = False
        q = obs.get("joint_command", obs["joint_positions"])
        t = obs.get("time", obs.get("sim_time"))
        times.append(k / hz if t is None else float(t))
        joints.append(np.asarray(q, dtype=float))
    if not joints:
        raise SystemExit(f"{episode} no tiene frames")
    times = np.asarray(times) - times[0]
    return times, np.array(joints), commanded


def nearest_turn(joints, current):
    """Desplaza cada articulación del brazo en vueltas enteras (2π) para que la trayectoria empiece
    lo más cerca posible de current: misma pose física, sin dar una vuelta entera (p.ej. la muñeca
    grabada en -2π con el brazo en 0). Devuelve los joints y las vueltas por articulación."""
    turns = np.round((joints[0, ARM] - current[ARM]) / (2 * np.pi))
    joints = joints.copy()
    joints[:, ARM] -= 2 * np.pi * turns
    return joints, turns.astype(int)


def resample(times, joints, t):
    """Joints interpolados linealmente en los instantes t (constantes fuera del rango)."""
    return np.stack([np.interp(t, times, joints[:, j]) for j in range(joints.shape[1])], axis=1)


def tracking(times, joints, sample_t, achieved) -> dict:
    """Error del brazo (rad, norma infinito por instante) contra el origen, y el retraso que mejor
    los alinea."""
    def err(lag):
        return np.abs(achieved[:, ARM] - resample(times, joints, sample_t - lag)[:, ARM]).max(axis=1)
    e = err(0.0)
    lags = np.arange(0.0, MAX_LAG + 1e-9, 0.01)
    lag = lags[int(np.argmin([np.sqrt(np.mean(err(l) ** 2)) for l in lags]))]
    return {"rms_err": float(np.sqrt(np.mean(e ** 2))), "max_err": float(e.max()),
            "final_err": float(e[-1]), "lag_s": float(lag)}


class Pacer:
    """Un comando por periodo de reloj; en el sim con sim_clock=fixed cada comando ya vale 1/hz
    simulado y no se espera."""
    def __init__(self, hz: float, wall: bool):
        self.period, self.wall = 1 / hz, wall
        self.start = time.monotonic()
        self.k = 0

    def wait(self) -> float:
        """Espera el próximo instante y devuelve su tiempo desde el inicio."""
        t = self.k * self.period
        if self.wall:
            time.sleep(max(0.0, self.start + t - time.monotonic()))
        self.k += 1
        return t


def approach(robot, target, speed: float, hz: float, wall: bool) -> None:
    """Lleva el brazo de donde está a target con un perfil de mínimo jerk."""
    q0 = np.asarray(robot.get_joint_state(), dtype=float)
    duration = max(np.abs(target[ARM] - q0[ARM]).max() / speed, 1.0)
    pacer = Pacer(hz, wall)
    while (t := pacer.wait()) < duration + SETTLE_SECONDS:
        tau = min(t / duration, 1.0)
        s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
        robot.command_joint_state(q0 + s * (target - q0))


def replay(robot, times, joints, args, hz, wall, record: bool, state_only: bool):
    """Manda la trayectoria a hz; devuelve los instantes y joints alcanzados (tras cada comando)."""
    commands = resample(times, joints, np.arange(0.0, times[-1] + 1e-9, 1 / hz))
    hold = int(round(args.hold_seconds * hz))
    sample_t, achieved = [], []
    pacer = Pacer(hz, wall)
    for q in [*commands, *[commands[-1]] * hold]:
        t = pacer.wait()
        robot.command_joint_state(q)
        # con imágenes, el servidor graba la observación que sigue a cada comando
        if record and not state_only:
            q_now = robot.get_observations()["joint_positions"]
        else:
            q_now = robot.get_joint_state()
        sample_t.append(t + 1 / hz)        # ~lo alcanzado al final del periodo del comando
        achieved.append(np.asarray(q_now, dtype=float))
    return np.array(sample_t), np.array(achieved)


def main(args: Args) -> None:
    bad = [m for m in args.modes if m not in MODES]
    if bad:
        raise SystemExit(f"modos desconocidos {bad}; usa {list(MODES)}")
    episodes = find_episodes(args.folder, args.episodes)
    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    obs = robot.get_observations()
    real = "sim_time" not in obs
    hz = args.hz or obs.get("control_hz")
    if hz is None:
        raise SystemExit("--hz: el servidor del sim no informa su frecuencia; usa la de su --hz")
    # el sim con sim_clock=fixed avanza 1/hz por comando: no hace falta esperar al reloj
    wall = real or obs.get("sim_clock") == "realtime"
    state_only = obs.get("sim_clock") == "realtime"
    ask = real if args.ask is None else args.ask
    original_mode = obs.get("control_mode")

    source = os.path.expanduser(args.folder).rstrip("/")
    root = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(
        f"{source}_replay", datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)
    print(f"[Replay] {len(episodes)} episodios × {len(args.modes)} modos a {hz} Hz "
          f"({'robot real' if real else 'sim'}); resultados en {root}")

    total = len(episodes) * len(args.modes)
    fields = ["source_episode", "mode", "folder", "commanded", "frames", "duration",
              "rms_err", "max_err", "final_err", "lag_s"]
    n = 0
    try:
        with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for episode in episodes:
                times, joints, commanded = load_reference(episode, hz)
                if not commanded:
                    print(f"[Replay] {episode}: sin joint_command (grabación antigua), se "
                          "reproduce la posición alcanzada")
                for mode in args.modes:
                    n += 1
                    what = f"episode {os.path.basename(episode)} in {mode}"
                    print(f"[Replay] {n}/{total} — {what} ({times[-1]:.1f} s)")
                    if ask:
                        answer = ask_operator(robot, f"Replay {n}/{total}: {what}. Set the scene, "
                                              "then Start", ["Start", "Skip", "Stop replays"])
                        if answer == "Skip":
                            continue
                        if answer == "Stop replays":
                            return
                    call(robot, "set_control_mode", mode=mode)
                    if args.reset:
                        call(robot, "reset_robot")
                    current = np.asarray(robot.get_joint_state(), dtype=float)
                    joints, turns = nearest_turn(joints, current)
                    if turns.any():
                        print(f"[Replay] vueltas de 2π restadas por articulación: {turns.tolist()} "
                              "(misma pose física)")
                    gap = np.abs(joints[0, ARM] - current[ARM]).max()
                    if gap > args.max_approach:
                        print(f"[Replay] la primera pose está a {gap:.2f} rad (> --max_approach "
                              f"{args.max_approach}); se salta. ¿Falta Reset Robot o es de otro robot?")
                        continue
                    approach(robot, joints[0], args.approach_speed, hz, wall)
                    folder = None
                    if args.record:
                        folder = call(robot, "start_recording", max_seconds=times[-1] + args.hold_seconds + 5,
                                      root=os.path.join(root, mode),
                                      metadata={"source": episode, "control_mode": mode, "hz": hz,
                                                "commanded": commanded})
                    try:
                        sample_t, achieved = replay(robot, times, joints, args, hz, wall,
                                                    args.record, state_only)
                    finally:
                        call(robot, "stop_recording")
                    result = tracking(times, joints, sample_t, achieved)
                    writer.writerow({"source_episode": episode, "mode": mode, "folder": folder or "",
                                     "commanded": commanded, "frames": len(sample_t),
                                     "duration": round(float(sample_t[-1]), 3),
                                     **{k: round(v, 4) for k, v in result.items()}})
                    f.flush()
                    print(f"[Replay] {mode}: error RMS {result['rms_err']:.3f} rad, máx "
                          f"{result['max_err']:.3f} rad, final {result['final_err']:.3f} rad, "
                          f"retraso {result['lag_s']:.2f} s")
    except KeyboardInterrupt:
        print("[Replay] Detenido")
    finally:
        try:
            call(robot, "stop_recording")
            if original_mode is not None:
                call(robot, "set_control_mode", mode=original_mode)
        except Exception as e:             # el servidor pudo cerrarse antes
            print(f"[Replay] No se pudo restaurar el modo {original_mode}: {e}")
        print(f"[Replay] Resultados en {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
