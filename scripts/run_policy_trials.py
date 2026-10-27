"""Corre ensayos automáticos de la política con el tomate en una grilla de posiciones.

Primero lanzar el servidor (con el mismo --hz con que se entrenó la política):
    python scripts/launch_nodes.py --robot sim_xarm5_splat_camera_2 --robot_name xarm5_lite_plant_v3 --hz 10
Luego:
    python scripts/run_policy_trials.py --trials-per-position 3 --trial-seconds 15

Las posiciones del tomate se leen de configs/apple_grid.yaml (se agregan con el botón
"Save Apple Position to Grid" del servidor con --move_apple, o editando el archivo).

Por cada posición de la grilla y cada ensayo: resetea el brazo, ubica el tomate,
resetea la política, graba y deja actuar a la política hasta que el servidor corta
la grabación (trial_seconds de tiempo simulado). La política se carga una sola vez.

Salida en output_dir/<nombre del modelo>/<fecha>/: una carpeta por ensayo (000/, 001/, ... con meta.json)
y trials.csv con un resumen (distancia final y mínima del efector al tomate).
"""
import csv
import datetime
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
import tyro
import yaml
from gello.zmq_core.robot_node import ZMQClientRobot

APPLE_NAME = "plastic_apple"
APPLE_QUAT = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Args:
    trials_per_position: int = 3
    trial_seconds: float = 15.0   # duración de cada ensayo, en tiempo simulado
    positions: tuple[int, ...] | None = None  # índices de la grilla a usar (por defecto todos)
    grid_path: str = "configs/apple_grid.yaml"
    output_dir: str = "~/data/policy_trials"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    # modelo a evaluar (None -> el por defecto del agente); dataset: de configs/registry.yaml
    checkpoint: str | None = None
    dataset_root: str | None = None
    # subcarpeta de output_dir; None -> nombre del modelo en el registro (o su carpeta)
    run_name: str | None = None


def call(robot: ZMQClientRobot, method: str, **args):
    """Llama un método del servidor que no está en ZMQClientRobot (REMOTE_METHODS)."""
    robot._socket.send(pickle.dumps({"method": method, "args": args}))
    result = pickle.loads(robot._socket.recv())
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{method}: {result['error']}")
    return result


def run_name_for(checkpoint: str) -> str:
    """Nombre del modelo en configs/registry.yaml (name o suggested_name), o su carpeta."""
    from splatsim.utils.registry import model_for_checkpoint, model_label
    found = model_for_checkpoint(checkpoint)
    return model_label(*found) if found else os.path.basename(checkpoint.rstrip("/"))


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
    with open(args.grid_path) as f:
        apple_grid = [tuple(p) for p in (yaml.safe_load(f) or {}).get("positions") or []]
    if not apple_grid:
        raise SystemExit(f"{args.grid_path} no tiene posiciones: guárdalas con el botón "
                         "'Save Apple Position to Grid' (launch_nodes.py --move_apple).")
    print(f"[Trials] {len(apple_grid)} posiciones cargadas de {args.grid_path}")

    from splatsim.agents.lerobot_diffusion_agent import DEFAULT_CHECKPOINT, LeRobotDiffusionAgent

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    # apto para nombre de carpeta (el panel pasa la etiqueta del modelo, que puede tener espacios)
    run_name = re.sub(r"[^\w.+-]+", "_", args.run_name or run_name_for(checkpoint)).strip("_")
    root = os.path.join(os.path.expanduser(args.output_dir), run_name,
                        datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)

    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    agent = LeRobotDiffusionAgent(checkpoint=args.checkpoint, dataset_root=args.dataset_root,
                                  device="cuda")

    positions = args.positions if args.positions is not None else range(len(apple_grid))
    total = len(positions) * args.trials_per_position
    fields = ["pos_idx", "trial", "x", "y", "z", "folder", "frames", "final_dist", "min_dist"]

    with open(os.path.join(root, "trials.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = 0
        try:
            for pos_idx in positions:
                apple_pos = apple_grid[pos_idx]
                for trial in range(args.trials_per_position):
                    n += 1
                    print(f"\n[Trials] {n}/{total} — posición {pos_idx} {apple_pos}, ensayo {trial}")
                    metadata = {"pos_idx": pos_idx, "trial": trial, "apple_position": apple_pos,
                                "trial_seconds": args.trial_seconds,
                                "checkpoint": checkpoint, "run_name": run_name}
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
