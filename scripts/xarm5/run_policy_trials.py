"""Corre ensayos automáticos de la política con el tomate en una grilla de posiciones.

Primero lanzar el servidor (con el mismo --hz con que se entrenó la política):
    python scripts/launch_nodes.py --robot sim_xarm5 --robot_name xarm5_lite_plant_v3 --hz 10
Luego, con un modelo de configs/registry.yaml (--list-models muestra los disponibles):
    python scripts/xarm5/run_policy_trials.py --model "gello benjamin 200 2cam" --trials-per-position 3 --trial-seconds 15

Las posiciones del tomate se leen de configs/apple_grid.yaml (se agregan con el botón
"Save Apple Position to Grid" del servidor con --move_apple, o editando el archivo).

Por cada posición de la grilla y cada ensayo: resetea el brazo, ubica el tomate,
resetea la política, graba y deja actuar a la política hasta que el servidor corta
la grabación (trial_seconds de tiempo simulado). La política se carga una sola vez.

Salida en output_dir/<nombre del modelo>/<fecha>/: una carpeta por ensayo (000/, 001/, ... con meta.json)
y trials.csv con un resumen (distancia final y mínima del efector al tomate).

Robot real (launch_nodes.py --robot xarm5): no hay posición del tomate, así que los ensayos se
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

from splatsim.utils.trials.grasp_point import SUCCESS_RADIUS, grasp_error, load_grasp_point
from splatsim.utils.remote_call import ask_operator, call
from splatsim.utils.trials.durations import other_durations

from splatsim.robots.xarm.config import APPLE_KEY, APPLE_NAME

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
    # modelo a evaluar por su nombre en configs/registry.yaml (name, carpeta o parte única de
    # ellos; --list-models los muestra), o por su carpeta / repo del Hub con --checkpoint
    # (ninguno -> el por defecto del agente); dataset: carpeta o repo del Hub, None -> del registro
    model: str | None = None
    list_models: bool = False     # muestra los modelos del registro y termina
    checkpoint: str | None = None
    dataset_root: str | None = None
    # subcarpeta de output_dir; None -> nombre del modelo en el registro (o su carpeta)
    run_name: str | None = None
    # ensayos en la posición actual del tomate, sin moverlo (en vez de recorrer la grilla): lo usa
    # "Record as trial" del panel; pos_idx es el índice de la grilla si coincide (a 1 cm), si no -1
    here: bool = False
    # oclusión con hojas: por cada posición y cada objetivo (%), una hoja de leaf_dir ajustada a ese
    # objetivo en las cámaras base (0 = sin hoja). La hoja depende solo de leaf_seed y la posición,
    # así todos los modelos ven la misma. None -> sin hojas (como antes)
    occlusion_targets: tuple[float, ...] | None = None
    leaf_dir: str = "~/data/occluders/xarm5_lite_plant_v3"
    leaf_seed: int = 0
    # IK del agente: off (solo posición, lo de siempre) o hybrid (con la orientación si alcanza)
    ik_orientation: str = "off"
    # acciones que se ejecutan por cada predicción de la política (None: las del modelo, 32 en todos los
    # entrenados hasta ahora); menos = vuelve a mirar más seguido. Hasta horizon - n_obs_steps + 1
    n_action_steps: int | None = None
    # None: una acción por paso de control, llegue o no el brazo (lo de siempre). Con un valor (rad,
    # norma de las 5 articulaciones), repite cada acción hasta que el brazo llegue a esa distancia del
    # comando (0.01 en el agente del SplatSim original) o hasta waypoint_max_steps pasos
    waypoint_tolerance: float | None = None
    waypoint_max_steps: int = 10
    # termina el ensayo success_extra_seconds después de que el error de agarre baja de
    # SUCCESS_RADIUS (4 cm): ahorra tiempo y no cambia la tasa de éxito; en esos segundos el brazo
    # sigue acercándose, así el error mínimo no queda cortado en ~4 cm. time_to_success se anota
    # igual con o sin esta opción
    stop_on_success: bool = True
    success_extra_seconds: float = 5.0   # tiempo simulado (de reloj en el robot real)
    # cámaras que la política recibe en negro (p. ej. base_rgb o wrist_rgb), para medir de cuál depende.
    # Solo cambia lo que ve la política: la grabación del ensayo guarda la imagen real
    blank_cameras: tuple[str, ...] = ()


def run_name_for(checkpoint: str) -> str:
    """Nombre del modelo en configs/registry.yaml (name o suggested_name), o su carpeta."""
    from splatsim.utils.data.registry import model_for_checkpoint, model_label
    found = model_for_checkpoint(checkpoint)
    return model_label(*found) if found else os.path.basename(checkpoint.rstrip("/"))


def policy_cameras(agent) -> list[str]:
    """Cámaras con que se entrenó la política, las únicas que se renderizan y graban en un ensayo."""
    return [key.removeprefix("observation.images.") for key in agent.policy.config.image_features]


def reset_policy(agent) -> None:
    agent.reset()                    # vacía las colas de observaciones y acciones


def leaf_columns(fit: dict | None, target: float | None) -> dict:
    """Columnas de trials.csv para la hoja del ensayo (vacías sin --occlusion-targets)."""
    if fit is None:
        return {}
    return {"occlusion_target": target, "leaf_status": fit.get("status"), "leaf": fit.get("leaf"),
            "leaf_camera": fit.get("camera"), "occlusion_fit": fit.get("occlusion")}


def blank(obs: dict, cameras) -> dict:
    """Copia de obs con las imágenes de esas cámaras en negro (mismo tipo y forma)."""
    if not cameras:
        return obs
    out = dict(obs)
    for cam in cameras:
        image = obs[cam]
        out[cam] = image * 0 if hasattr(image, "detach") else np.zeros_like(image)
    return out


def grid_index(apple_grid, pos, tol: float = 0.01) -> int:
    """Índice de la grilla más cercano a pos si está a menos de tol (m), si no -1."""
    if not apple_grid:
        return -1
    d = np.linalg.norm(np.asarray(apple_grid) - np.asarray(pos), axis=1)
    return int(d.argmin()) if d.min() < tol else -1


def occlusion_columns(measured: dict) -> dict:
    """{occlusion_<cámara>: %} (vacío si el tomate no está en cuadro)."""
    return {f"occlusion_{cam}": r["occlusion"] for cam, r in measured.items()}


def run_trial(robot, agent, apple_pos, trial_seconds, metadata, root, grasp_point,
              reset: bool = True, measure_occlusion: bool = False, leaf: dict | None = None,
              stop_on_success: bool = False, success_extra_seconds: float = 0.0,
              blank_cameras: tuple[str, ...] = ()) -> dict:
    """apple_pos None: el tomate se deja donde está. Sin su posición en la observación (robot
    real) solo se cuentan los frames. leaf: {leaf_dir, target, key, seed} para place_trial_leaf.
    time_to_success: segundos desde el primer comando hasta que el error de agarre baja de
    SUCCESS_RADIUS (tiempo simulado en el sim, de reloj en el robot real); vacío si no llega.
    stop_on_success: corta el ensayo success_extra_seconds después de ese momento.
    blank_cameras: cámaras que la política recibe en negro."""
    call(robot, "stop_recording")
    if reset:
        call(robot, "reset_robot")
    if apple_pos is not None:
        robot.set_object_pose(APPLE_NAME, np.array(apple_pos), np.array(APPLE_QUAT), use_gravity=False)
    reset_policy(agent)
    leaf_fit = call(robot, "place_trial_leaf", **leaf) if leaf is not None else None
    leaf_info = leaf_columns(leaf_fit, leaf["target"] if leaf else None)
    ik_before = dict(agent.ik_counts)
    waypoints_before = dict(agent.waypoint_counts)

    # antes de que el brazo se mueva: cuánto tapa la planta (y las hojas) con el brazo en reposo
    occlusion = occlusion_columns(call(robot, "measure_occlusion")) if measure_occlusion else {}
    metadata = {**metadata, **occlusion, **leaf_info, "leaf_fit": leaf_fit,
                "ik_orientation": agent.ik_orientation, "relative_actions": agent.relative_actions}
    obs = robot.get_observations()
    folder = call(robot, "start_recording", max_seconds=trial_seconds,
                  metadata=metadata, root=root, images_in_pkl=False, cameras=policy_cameras(agent))

    # robot real (trae control_hz): un comando por periodo de reloj; en el sim cada comando ya
    # vale 1/hz de tiempo simulado y no hace falta esperar
    period = 1 / obs["control_hz"] if "control_hz" in obs else 0.0
    frames, distances, grasp_errors = 0, [], []
    clock = (lambda o: o["sim_time"]) if "sim_time" in obs else (lambda o: time.monotonic())
    start, time_to_success, stopped = clock(obs), None, False
    while True:
        t0 = time.monotonic()
        robot.command_joint_state(agent.act(blank(obs, blank_cameras)))
        obs = robot.get_observations()
        frames += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
        if APPLE_KEY in obs:
            ee = np.asarray(obs["state"][:3])
            apple = np.asarray(obs[APPLE_KEY])
            distances.append(float(np.linalg.norm(ee - apple)))
            # distancia del tomate al punto de agarre, en el marco del efector (configs/grasp_point.yaml)
            grasp_errors.append(float(grasp_error(obs["state"], apple, grasp_point)))
            if time_to_success is None and grasp_errors[-1] <= SUCCESS_RADIUS:
                time_to_success = round(clock(obs) - start, 3)
            if (stop_on_success and time_to_success is not None
                    and clock(obs) - start >= time_to_success + success_extra_seconds - 1e-9):
                call(robot, "stop_recording")
                stopped = True
                break
        if not obs["recording"]:   # el servidor cortó al llegar a trial_seconds
            break

    result = {"folder": folder, "frames": frames, "time_to_success": time_to_success,
              "stopped_on_success": int(stopped),
              **occlusion, **leaf_info}
    if agent.ik_orientation != "off":
        result["ik_fallbacks"] = agent.ik_counts["fallback"] - ik_before["fallback"]
    if agent.waypoint_tolerance is not None:
        # acciones terminadas al llegar y por agotar waypoint_max_steps
        result["waypoints_reached"] = agent.waypoint_counts["reached"] - waypoints_before["reached"]
        result["waypoint_timeouts"] = agent.waypoint_counts["timeout"] - waypoints_before["timeout"]
    if distances:
        result.update(final_dist=distances[-1], min_dist=min(distances),
                      final_grasp_err=grasp_errors[-1], min_grasp_err=min(grasp_errors))
    return result


def main(args: Args) -> None:
    from splatsim.utils.data.registry import find_model, models_table, model_label
    if args.list_models:
        print(models_table())
        return
    if args.model:
        if args.checkpoint:
            raise SystemExit("Usa --model o --checkpoint, no ambos")
        try:
            key, model = find_model(args.model)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        args.checkpoint = model["path"]
        print(f"[Trials] Modelo {model_label(key, model)}: {args.checkpoint}")
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
    # corridas anteriores del mismo modelo con otra duración: compare_trials.py no las mezcla
    for seconds, runs in other_durations(os.path.dirname(root), args.trial_seconds, skip=root).items():
        print(f"[Trials] AVISO: {run_name} ya tiene corridas de {seconds:g} s ({', '.join(runs)}) y esta "
              f"es de {args.trial_seconds:g} s. Sus resultados no son comparables: compare_trials.py "
              "las separa por duración; usa la misma --trial_seconds para juntarlas.")

    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    agent = LeRobotDiffusionAgent(checkpoint=args.checkpoint, dataset_root=args.dataset_root,
                                  device="cuda", ik_orientation=args.ik_orientation,
                                  n_action_steps=args.n_action_steps,
                                  waypoint_tolerance=args.waypoint_tolerance,
                                  waypoint_max_steps=args.waypoint_max_steps)

    first_obs = robot.get_observations()
    missing = [c for c in policy_cameras(agent) if first_obs.get(c) is None]
    if missing:
        raise SystemExit(f"La política usa {missing}, pero el servidor no las tiene: "
                         "relanza launch_nodes.py con esas cámaras.")

    # robot real: sin posición del tomate, el operador lo pone y marca el resultado
    manual = APPLE_KEY not in first_obs
    if manual:
        print("[Trials] Sin posición del tomate (robot real): ensayos con el operador, en el panel")
    here = None
    if args.here:
        if not manual:
            here = tuple(float(v) for v in robot.get_observations()[APPLE_KEY])
        positions = [grid_index(apple_grid, here) if here else -1]
    else:
        positions = args.positions if args.positions is not None else range(len(apple_grid))
    targets = list(args.occlusion_targets) if args.occlusion_targets and not manual else [None]
    total = len(positions) * len(targets) * args.trials_per_position
    fields = ["pos_idx", "trial", "x", "y", "z", "folder", "frames", "final_dist", "min_dist",
              "final_grasp_err", "min_grasp_err", "manual_success", "grasp_point", "trial_seconds",
              "time_to_success", "stopped_on_success", "success_extra_seconds", "n_action_steps"]
    if targets != [None]:
        fields += ["occlusion_target", "leaf_status", "leaf", "leaf_camera", "occlusion_fit"]
    elif not manual:
        # la hoja de una corrida anterior con --occlusion_targets sigue en el servidor: se quita
        call(robot, "place_trial_leaf", leaf_dir=args.leaf_dir, target=0.0, key=0, seed=args.leaf_seed)
    if args.ik_orientation != "off":
        fields += ["ik_fallbacks"]
    if args.blank_cameras:
        fields += ["blank_cameras"]
    unknown = set(args.blank_cameras) - set(policy_cameras(agent))
    if unknown:
        raise SystemExit(f"--blank_cameras: la política no usa {sorted(unknown)} (usa {policy_cameras(agent)})")
    if args.waypoint_tolerance is not None:
        fields += ["waypoint_tolerance", "waypoint_max_steps", "waypoints_reached", "waypoint_timeouts"]
    if not manual:
        fields += list(occlusion_columns(call(robot, "measure_occlusion")))
    grasp_point = load_grasp_point()
    # con qué punto se midió el error de agarre: compare_trials.py recalcula si cambió
    point_text = " ".join(f"{v:.4f}" for v in grasp_point)

    with open(os.path.join(root, "trials.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = 0
        stop = False
        try:
            for pos_idx, target in ((p, t) for p in positions for t in targets):
                apple_pos = here if args.here else apple_grid[pos_idx]
                # la misma hoja (sorteada por posición) en todos los niveles: solo cambia cuánto tapa
                leaf = None if target is None else {"leaf_dir": args.leaf_dir, "target": target,
                                                    "key": pos_idx, "seed": args.leaf_seed}
                for trial in range(args.trials_per_position):
                    n += 1
                    where = "actual" if args.here else ""
                    level = "" if target is None else f", oclusión {target:g}%"
                    print(f"\n[Trials] {n}/{total} — posición {where} {pos_idx} {apple_pos}{level}, ensayo {trial}")
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
                                "success_extra_seconds": args.success_extra_seconds if args.stop_on_success else None,
                                "n_action_steps": agent.policy.config.n_action_steps,
                                "waypoint_tolerance": args.waypoint_tolerance,
                                "waypoint_max_steps": args.waypoint_max_steps if args.waypoint_tolerance else None,
                                "checkpoint": checkpoint, "run_name": run_name,
                                "blank_cameras": list(args.blank_cameras)}
                    result = run_trial(robot, agent, None if (args.here or manual) else apple_pos,
                                       args.trial_seconds, metadata, root, grasp_point,
                                       reset=not manual, measure_occlusion=not manual, leaf=leaf,
                                       stop_on_success=args.stop_on_success and not manual,
                                       success_extra_seconds=args.success_extra_seconds,
                                       blank_cameras=args.blank_cameras)
                    if manual:
                        call(robot, "stop_recording")
                        answer = ask_operator(robot, f"Trial {n}/{total} finished: result?",
                                              ["Success", "Fail", "Discard"])
                        if answer == "Discard":
                            print(f"[Trials] Descartado por el operador ({result['folder']})")
                            continue
                        result["manual_success"] = int(answer == "Success")
                    row = {"pos_idx": pos_idx, "trial": trial, **result, "grasp_point": point_text,
                           "trial_seconds": args.trial_seconds,
                           "success_extra_seconds": args.success_extra_seconds if args.stop_on_success else "",
                           "n_action_steps": agent.policy.config.n_action_steps}
                    if args.waypoint_tolerance is not None:
                        row.update(waypoint_tolerance=args.waypoint_tolerance,
                                   waypoint_max_steps=args.waypoint_max_steps)
                    if args.blank_cameras:
                        row["blank_cameras"] = " ".join(args.blank_cameras)
                    if apple_pos:
                        row.update(x=apple_pos[0], y=apple_pos[1], z=apple_pos[2])
                    writer.writerow(row)
                    f.flush()
                    if manual:
                        print(f"[Trials] {result['frames']} frames, resultado: {answer.lower()}")
                    else:
                        reached = ("" if result["time_to_success"] is None
                                   else f", éxito a los {result['time_to_success']:.1f} s")
                        print(f"[Trials] {result['frames']} frames, error de agarre final "
                              f"{result['final_grasp_err']:.3f} m, mín {result['min_grasp_err']:.3f} m{reached}")
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
