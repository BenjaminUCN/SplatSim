"""Corre ensayos automáticos de la política con el tomate en una grilla de posiciones.

Primero lanzar el servidor (con el mismo --hz con que se entrenó la política):
    python scripts/launch_nodes.py --robot sim_xarm5_splat_camera_2 --robot_name xarm5_lite_plant_v3 --hz 10
Luego:
    python scripts/xarm5/run_policy_trials.py --trials-per-position 3 --trial-seconds 15

Las posiciones del tomate se leen de configs/apple_grid.yaml (se agregan con el botón
"Save Apple Position to Grid" del servidor con --move_apple, o editando el archivo).

Por cada posición de la grilla y cada ensayo: resetea el brazo, ubica el tomate,
resetea la política, graba y deja actuar a la política hasta que el servidor corta
la grabación (trial_seconds de tiempo simulado). La política se carga una sola vez.

Salida en output_dir/<nombre del modelo>/<fecha>/: una carpeta por ensayo (000/, 001/, ... con meta.json)
y trials.csv con un resumen (distancia final y mínima del efector al tomate).

Robot real (launch_nodes.py --robot xarm5_lite): no hay posición del tomate, así que los ensayos se
hacen con el operador, por el panel: antes de cada uno pide poner el tomate en la posición de la
grilla (Start / Skip / Stop trials) y al final marcar el resultado (Success / Fail / Discard), que
queda en la columna manual_success (Discard no escribe fila). La duración es de reloj.
"""
import time
import csv
import datetime
import os
import re
from dataclasses import dataclass

import numpy as np
import tyro
import yaml
from gello.zmq_core.robot_node import ZMQClientRobot

from splatsim.utils.grasp_point import grasp_error, load_grasp_point
from splatsim.utils.remote_call import ask_operator, call

APPLE_NAME = "plastic_apple"
APPLE_QUAT = (0.0, 0.0, 0.0, 1.0)
APPLE_KEY = f"{APPLE_NAME}_position"


@dataclass
class Args:
    trials_per_position: int = 3
    trial_seconds: float = 15.0   # duración de cada ensayo, en tiempo simulado
    positions: tuple[int, ...] | None = None  # índices de la grilla a usar (por defecto todos)
    grid_path: str = "configs/apple_grid.yaml"
    output_dir: str = "~/data/policy_trials"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    # modelo a evaluar, carpeta o repo del Hub (None -> el por defecto del agente);
    # dataset: carpeta o repo del Hub, None -> de configs/registry.yaml
    checkpoint: str | None = None
    dataset_root: str | None = None
    # subcarpeta de output_dir; None -> nombre del modelo en el registro (o su carpeta)
    run_name: str | None = None
    # ensayos en la posición actual del tomate, sin moverlo (en vez de recorrer la grilla): lo usa
    # "Record as trial" del panel; pos_idx es el índice de la grilla si coincide (a 1 cm), si no -1
    here: bool = False


def run_name_for(checkpoint: str) -> str:
    """Nombre del modelo en configs/registry.yaml (name o suggested_name), o su carpeta."""
    from splatsim.utils.registry import model_for_checkpoint, model_label
    found = model_for_checkpoint(checkpoint)
    return model_label(*found) if found else os.path.basename(checkpoint.rstrip("/"))


def reset_policy(agent) -> None:
    agent.policy.reset()             # vacía las colas de observaciones y acciones
    agent.make_new_prediction = True


def grid_index(apple_grid, pos, tol: float = 0.01) -> int:
    """Índice de la grilla más cercano a pos si está a menos de tol (m), si no -1."""
    if not apple_grid:
        return -1
    d = np.linalg.norm(np.asarray(apple_grid) - np.asarray(pos), axis=1)
    return int(d.argmin()) if d.min() < tol else -1


def run_trial(robot, agent, apple_pos, trial_seconds, metadata, root, grasp_point,
              reset: bool = True) -> dict:
    """apple_pos None: el tomate se deja donde está. Sin su posición en la observación (robot
    real) solo se cuentan los frames."""
    call(robot, "stop_recording")
    if reset:
        call(robot, "reset_robot")
    if apple_pos is not None:
        robot.set_object_pose(APPLE_NAME, np.array(apple_pos), np.array(APPLE_QUAT), use_gravity=False)
    reset_policy(agent)

    obs = robot.get_observations()
    folder = call(robot, "start_recording", max_seconds=trial_seconds,
                  metadata=metadata, root=root)

    # robot real (trae control_hz): un comando por periodo de reloj; en el sim cada comando ya
    # vale 1/hz de tiempo simulado y no hace falta esperar
    period = 1 / obs["control_hz"] if "control_hz" in obs else 0.0
    frames, distances, grasp_errors = 0, [], []
    while True:
        t0 = time.monotonic()
        robot.command_joint_state(agent.act(obs))
        obs = robot.get_observations()
        frames += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
        if APPLE_KEY in obs:
            ee = np.asarray(obs["state"][:3])
            apple = np.asarray(obs[APPLE_KEY])
            distances.append(float(np.linalg.norm(ee - apple)))
            # distancia del tomate al punto de agarre, en el marco del efector (configs/grasp_point.yaml)
            grasp_errors.append(float(grasp_error(obs["state"], apple, grasp_point)))
        if not obs["recording"]:   # el servidor cortó al llegar a trial_seconds
            break

    result = {"folder": folder, "frames": frames}
    if distances:
        result.update(final_dist=distances[-1], min_dist=min(distances),
                      final_grasp_err=grasp_errors[-1], min_grasp_err=min(grasp_errors))
    return result


def main(args: Args) -> None:
    with open(args.grid_path) as f:
        apple_grid = [tuple(p) for p in (yaml.safe_load(f) or {}).get("positions") or []]
    if not apple_grid and not args.here:
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

    # robot real: sin posición del tomate, el operador lo pone y marca el resultado
    manual = APPLE_KEY not in robot.get_observations()
    if manual:
        print("[Trials] Sin posición del tomate (robot real): ensayos con el operador, en el panel")
    here = None
    if args.here:
        if not manual:
            here = tuple(float(v) for v in robot.get_observations()[APPLE_KEY])
        positions = [grid_index(apple_grid, here) if here else -1]
    else:
        positions = args.positions if args.positions is not None else range(len(apple_grid))
    total = len(positions) * args.trials_per_position
    fields = ["pos_idx", "trial", "x", "y", "z", "folder", "frames", "final_dist", "min_dist",
              "final_grasp_err", "min_grasp_err", "manual_success"]
    grasp_point = load_grasp_point()

    with open(os.path.join(root, "trials.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = 0
        stop = False
        try:
            for pos_idx in positions:
                apple_pos = here if args.here else apple_grid[pos_idx]
                for trial in range(args.trials_per_position):
                    n += 1
                    where = "actual" if args.here else ""
                    print(f"\n[Trials] {n}/{total} — posición {where} {pos_idx} {apple_pos}, ensayo {trial}")
                    if manual:
                        call(robot, "stop_recording")
                        call(robot, "reset_robot")
                        place = (f"at grid position #{pos_idx} {tuple(apple_pos)}" if apple_pos
                                 else "where you want it")
                        answer = ask_operator(robot, f"Trial {n}/{total}: place the tomato {place}, "
                                              "then press Start", ["Start", "Skip", "Stop trials"])
                        if answer == "Stop trials":
                            stop = True
                            break
                        if answer == "Skip":
                            print("[Trials] Saltado por el operador")
                            continue
                    metadata = {"pos_idx": pos_idx, "trial": trial, "apple_position": apple_pos,
                                "trial_seconds": args.trial_seconds, "manual": manual,
                                "checkpoint": checkpoint, "run_name": run_name}
                    result = run_trial(robot, agent, None if (args.here or manual) else apple_pos,
                                       args.trial_seconds, metadata, root, grasp_point,
                                       reset=not manual)
                    if manual:
                        call(robot, "stop_recording")
                        answer = ask_operator(robot, f"Trial {n}/{total} finished: result?",
                                              ["Success", "Fail", "Discard"])
                        if answer == "Discard":
                            print(f"[Trials] Descartado por el operador ({result['folder']})")
                            continue
                        result["manual_success"] = int(answer == "Success")
                    row = {"pos_idx": pos_idx, "trial": trial, **result}
                    if apple_pos:
                        row.update(x=apple_pos[0], y=apple_pos[1], z=apple_pos[2])
                    writer.writerow(row)
                    f.flush()
                    if manual:
                        print(f"[Trials] {result['frames']} frames, resultado: {answer.lower()}")
                    else:
                        print(f"[Trials] {result['frames']} frames, error de agarre final "
                              f"{result['final_grasp_err']:.3f} m, mín {result['min_grasp_err']:.3f} m")
                if stop:
                    print("[Trials] Detenido por el operador")
                    break
        except KeyboardInterrupt:
            print("\n[Trials] Interrumpido.")
        finally:
            # socket nuevo: si Ctrl+C cortó un request a medias, el anterior queda inutilizable
            cleanup = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            call(cleanup, "operator_prompt")     # quita una pregunta pendiente del panel
            call(cleanup, "stop_recording")
            call(cleanup, "reset_robot")

    print(f"\n[Trials] Resultados en {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
