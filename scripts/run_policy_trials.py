"""Corre ensayos automáticos de la política con el tomate en una grilla de posiciones.

Primero lanzar el servidor (con el mismo --hz con que se entrenó la política):
    python scripts/launch_nodes.py --robot sim_xarm5_splat_camera_2 --robot_name xarm5_lite_plant_v3 --hz 10
Luego:
    python scripts/run_policy_trials.py --trials-per-position 3 --trial-seconds 15

Por cada posición de APPLE_GRID y cada ensayo: resetea el brazo, ubica el tomate,
resetea la política, graba y deja actuar a la política hasta que el servidor corta
la grabación (trial_seconds de tiempo simulado). La política se carga una sola vez.

Salida en output_dir/<fecha>/: una carpeta por ensayo (000/, 001/, ... con meta.json)
y trials.csv con un resumen (distancia final y mínima del efector al tomate).
"""
import csv
import datetime
import os
import pickle
from dataclasses import dataclass

import numpy as np
import tyro
from gello.zmq_core.robot_node import ZMQClientRobot

# Posiciones del tomate (x, y, z) en el mundo de PyBullet. Editar antes de correr.
# Referencia: los sliders de --move_apple usan x∈[-0.9,-0.3], y∈[-0.8,0.8], z∈[-0.5,0.6].
APPLE_GRID = [
    (-0.50, -0.15, 0.05), (-0.50, 0.00, 0.05), (-0.50, 0.15, 0.05),
    (-0.50, -0.15, 0.15), (-0.50, 0.00, 0.15), (-0.50, 0.15, 0.15),
    (-0.50, -0.15, 0.25), (-0.50, 0.00, 0.25), (-0.50, 0.15, 0.25),
]
APPLE_NAME = "plastic_apple"
APPLE_QUAT = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Args:
    trials_per_position: int = 3
    trial_seconds: float = 15.0   # duración de cada ensayo, en tiempo simulado
    positions: tuple[int, ...] | None = None  # índices de APPLE_GRID a usar (por defecto todos)
    output_dir: str = "~/data/policy_trials"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"


def call(robot: ZMQClientRobot, method: str, **args):
    """Llama un método del servidor que no está en ZMQClientRobot (REMOTE_METHODS)."""
    robot._socket.send(pickle.dumps({"method": method, "args": args}))
    result = pickle.loads(robot._socket.recv())
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{method}: {result['error']}")
    return result


def reset_policy(agent) -> None:
    agent.policy.reset()             # vacía las colas de observaciones y acciones
    agent.make_new_prediction = True


def run_trial(robot, agent, apple_pos, trial_seconds, metadata, root) -> dict:
    call(robot, "stop_recording")
    call(robot, "reset_robot")
    robot.set_object_pose(APPLE_NAME, np.array(apple_pos), np.array(APPLE_QUAT), use_gravity=False)
    reset_policy(agent)

    obs = robot.get_observations()
    folder = call(robot, "start_recording", max_seconds=trial_seconds,
                  metadata=metadata, root=root)

    distances = []
    while True:
        robot.command_joint_state(agent.act(obs))
        obs = robot.get_observations()
        ee = np.asarray(obs["state"][:3])
        apple = np.asarray(obs[f"{APPLE_NAME}_position"])
        distances.append(float(np.linalg.norm(ee - apple)))
        if not obs["recording"]:   # el servidor cortó al llegar a trial_seconds
            break

    return {"folder": folder, "frames": len(distances),
            "final_dist": distances[-1], "min_dist": min(distances)}


def main(args: Args) -> None:
    from splatsim.agents.lerobot_diffusion_agent import LeRobotDiffusionAgent

    root = os.path.join(os.path.expanduser(args.output_dir),
                        datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)

    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    agent = LeRobotDiffusionAgent(device="cuda")

    positions = args.positions if args.positions is not None else range(len(APPLE_GRID))
    total = len(positions) * args.trials_per_position
    fields = ["pos_idx", "trial", "x", "y", "z", "folder", "frames", "final_dist", "min_dist"]

    with open(os.path.join(root, "trials.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = 0
        try:
            for pos_idx in positions:
                apple_pos = APPLE_GRID[pos_idx]
                for trial in range(args.trials_per_position):
                    n += 1
                    print(f"\n[Trials] {n}/{total} — posición {pos_idx} {apple_pos}, ensayo {trial}")
                    metadata = {"pos_idx": pos_idx, "trial": trial, "apple_position": apple_pos,
                                "trial_seconds": args.trial_seconds}
                    result = run_trial(robot, agent, apple_pos, args.trial_seconds, metadata, root)
                    row = {"pos_idx": pos_idx, "trial": trial,
                           "x": apple_pos[0], "y": apple_pos[1], "z": apple_pos[2], **result}
                    writer.writerow(row)
                    f.flush()
                    print(f"[Trials] {result['frames']} frames, dist final "
                          f"{result['final_dist']:.3f} m, mín {result['min_dist']:.3f} m")
        except KeyboardInterrupt:
            print("\n[Trials] Interrumpido.")
        finally:
            # socket nuevo: si Ctrl+C cortó un request a medias, el anterior queda inutilizable
            cleanup = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            call(cleanup, "stop_recording")
            call(cleanup, "reset_robot")

    print(f"\n[Trials] Resultados en {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
