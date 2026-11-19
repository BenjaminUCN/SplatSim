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
En una escena con cortadora y tallo (p. ej. --robot_name xarm5_lite_plant_v3_hand_held_down_cam_down en el
servidor), trials.csv suma el corte: min_cut_err (distancia mínima del centro del filo al punto de corte antes
de cortar), cut_err_at_command (esa distancia cuando la política ordena cerrar por primera vez, orden de la
cortadora >= 0.5), cut_err_at_closed (cuando la cortadora llega a 0.8 cerrada, el momento en que se decide el
corte), time_to_cut, y peduncle_cut / peduncle_held al final. Éxito del corte: peduncle_held = 1.

Robot real (launch_nodes.py --robot xarm5): no hay posición del tomate, así que los ensayos se
hacen con el operador, por el panel: antes de cada uno pide poner el tomate en la posición de la
grilla (Start / Skip / Stop trials) y al final marcar el resultado (Success / Fail / Discard), que
queda en la columna manual_success (Discard no escribe fila). La duración es de reloj.
"""
import time
import csv
import json
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
from splatsim.utils.scene.peduncle import CLOSED_FRACTION

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
    # cámara en que la hoja tapa el tomate (None: las cámaras base, como antes). Con wrist_rgb, la hoja
    # se ajusta con el brazo en la pose de pregrasp del planner (efector a leaf_pregrasp m del punto de
    # agarre, inclinación leaf_pitch_deg o la más cercana que alcance), porque en la pose inicial la
    # muñeca no ve el tomate en casi ninguna posición. La hoja queda fija en el mundo durante el ensayo
    leaf_camera: str | None = None
    leaf_pregrasp: float = 0.09     # el centro del rango de pregrasp de las demos (0.06–0.12 m)
    leaf_pitch_deg: float = -27.5   # el centro del rango de inclinación de las demos (−50 a −5°)
    # IK del agente: off (solo posición, lo de siempre) o hybrid (con la orientación si alcanza)
    ik_orientation: str = "off"
    # muestreador de la difusión (None: el del modelo, DDPM) y sus pasos (None: los del modelo, 100). DDIM
    # muestrea los mismos modelos entrenados con DDPM, sin reentrenar; p. ej. --noise_scheduler DDIM
    # --num_inference_steps 16
    noise_scheduler: str | None = None
    num_inference_steps: int | None = None
    # acciones que se ejecutan por cada predicción de la política (None: las del modelo, 32 en todos los
    # entrenados hasta ahora); menos = vuelve a mirar más seguido. Hasta horizon - n_obs_steps + 1
    n_action_steps: int | None = None
    # None: una acción por paso de control, llegue o no el brazo (lo de siempre). Con un valor (rad,
    # norma de las 5 articulaciones), repite cada acción hasta que el brazo llegue a esa distancia del
    # comando (0.01 en el agente del SplatSim original) o hasta waypoint_max_steps pasos
    waypoint_tolerance: float | None = None
    waypoint_max_steps: int = 10
    # None: la pinza recibe el valor predicho (lo de siempre). Con un valor, 1 si la predicción lo alcanza y
    # 0 si no, como la orden binaria de las demos del planificador
    gripper_threshold: float | None = None
    # al ordenar cerrar la pinza, el brazo se queda quieto hasta que cierra (como en las demos del planificador)
    hold_while_closing: bool = False
    # una vez cerrada la pinza (>= 0.8), queda cerrada hasta el final del ensayo (como en las demos de cortar y sujetar)
    latch_closed: bool = False
    # termina el ensayo success_extra_seconds después de que el error de agarre baja de
    # SUCCESS_RADIUS (4 cm): ahorra tiempo y no cambia la tasa de éxito; en esos segundos el brazo
    # sigue acercándose, así el error mínimo no queda cortado en ~4 cm. time_to_success se anota
    # igual con o sin esta opción
    stop_on_success: bool = True
    success_extra_seconds: float = 5.0   # tiempo simulado (de reloj en el robot real)
    # cámaras que la política recibe en negro (p. ej. base_rgb o wrist_rgb), para medir de cuál depende.
    # Solo cambia lo que ve la política: la grabación del ensayo guarda la imagen real
    blank_cameras: tuple[str, ...] = ()
    # mide cuánto del tomate tapa la escena en esta cámara (p. ej. wrist_rgb) cada track_every pasos
    # de control durante el ensayo, no solo al inicio: guarda la serie en occlusion_track.json de cada
    # episodio y su resumen en trials.csv (track_<cámara>_mean, _max, _pre_success y _in_view)
    track_occlusion: str | None = None
    track_every: int = 5


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


def pregrasp_q(kin, tomato, point, seed, back: float, pitch_deg: float) -> list[float] | None:
    """Articulaciones del brazo en la pose de pregrasp de plan_trajectories.plan_episode: el efector
    `back` m antes de la pose de agarre, sobre el eje de la herramienta. Prueba pitch_deg y después
    las inclinaciones del rango de las demos más cercanas a ella; None si ninguna alcanza."""
    from plan_trajectories import grasp_pose
    for pitch in [pitch_deg, *sorted(np.linspace(-50.0, -5.0, 10), key=lambda p: abs(p - pitch_deg))]:
        pos, rot = grasp_pose(tomato, np.radians(pitch), point)
        q = kin.ik(pos - back * rot[:, 2], rot, seed)
        if q is not None:
            return [float(v) for v in q]
    return None


def grid_index(apple_grid, pos, tol: float = 0.01) -> int:
    """Índice de la grilla más cercano a pos si está a menos de tol (m), si no -1."""
    if not apple_grid:
        return -1
    d = np.linalg.norm(np.asarray(apple_grid) - np.asarray(pos), axis=1)
    return int(d.argmin()) if d.min() < tol else -1


def occlusion_columns(measured: dict) -> dict:
    """{occlusion_<cámara>: %} (vacío si el tomate no está en cuadro)."""
    return {f"occlusion_{cam}": r["occlusion"] for cam, r in measured.items()}


TRACK_FILE = "occlusion_track.json"
CLOSE_COMMAND = 0.5   # orden de la cortadora desde la que se cuenta como "cerrar" (las demos dan 0/1)


def track_columns(camera: str) -> list[str]:
    return [f"track_{camera}_{k}" for k in ("mean", "max", "pre_success", "in_view")]


def track_summary(camera: str, samples: list[tuple[float, float | None]],
                  time_to_success: float | None) -> dict:
    """samples: (segundos desde el inicio, % tapado o None si el tomate no está en cuadro). mean/max
    sobre las muestras en cuadro; pre_success: la media hasta time_to_success (todo el ensayo si no
    llegó); in_view: fracción de muestras con el tomate en cuadro."""
    seen = [o for _, o in samples if o is not None]
    pre = [o for t, o in samples if o is not None and (time_to_success is None or t <= time_to_success + 1e-9)]
    mean = lambda v: round(float(np.mean(v)), 1) if v else None  # noqa: E731
    cols = track_columns(camera)
    return {cols[0]: mean(seen), cols[1]: max(seen) if seen else None, cols[2]: mean(pre),
            cols[3]: round(len(seen) / len(samples), 2) if samples else None}


def run_trial(robot, agent, apple_pos, trial_seconds, metadata, root, grasp_point,
              reset: bool = True, measure_occlusion: bool = False, leaf: dict | None = None,
              stop_on_success: bool = False, success_extra_seconds: float = 0.0,
              blank_cameras: tuple[str, ...] = (), track_camera: str | None = None,
              track_every: int = 5) -> dict:
    """apple_pos None: el tomate se deja donde está. Sin su posición en la observación (robot
    real) solo se cuentan los frames. leaf: {leaf_dir, target, key, seed} para place_trial_leaf.
    time_to_success: segundos desde el primer comando hasta que el error de agarre baja de
    SUCCESS_RADIUS (tiempo simulado en el sim, de reloj en el robot real); vacío si no llega.
    stop_on_success: corta el ensayo success_extra_seconds después de ese momento.
    blank_cameras: cámaras que la política recibe en negro.
    track_camera: mide la oclusión del tomate en esa cámara cada track_every pasos (track_summary)."""
    call(robot, "stop_recording")
    if reset:
        call(robot, "reset_robot")
    if apple_pos is not None:
        robot.set_object_pose(APPLE_NAME, np.array(apple_pos), np.array(APPLE_QUAT), use_gravity=False)
    reset_policy(agent)
    leaf_fit = call(robot, "place_trial_leaf", **leaf) if leaf is not None else None
    leaf_info = leaf_columns(leaf_fit, leaf["target"] if leaf else None)
    ik_before = dict(agent.ik_counts)
    latencies = getattr(agent, "chunk_latencies_ms", [])   # los agentes de prueba no lo tienen
    chunks_before = len(latencies)
    waypoints_before = dict(agent.waypoint_counts)
    holds_before = getattr(agent, "hold_count", 0)

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
    cut_errors, time_to_cut = [], None   # escena con cortadora y tallo (peduncle_error en la observación)
    # error del filo al punto de corte cuando la política ordena cerrar (orden >= CLOSE_COMMAND por primera
    # vez) y cuando la cortadora llega a CLOSED_FRACTION, el momento en que el simulador decide el corte
    cut_err_at_command = cut_err_at_closed = None
    clock = (lambda o: o["sim_time"]) if "sim_time" in obs else (lambda o: time.monotonic())
    start, time_to_success, stopped = clock(obs), None, False

    def track():
        r = call(robot, "measure_occlusion", camera_names=[track_camera]).get(track_camera) or {}
        samples.append((round(clock(obs) - start, 3), r.get("occlusion")))
    samples = []
    if track_camera:
        track()
    while True:
        t0 = time.monotonic()
        action = agent.act(blank(obs, blank_cameras))
        robot.command_joint_state(action)
        obs = robot.get_observations()
        frames += 1
        if track_camera and frames % track_every == 0:
            track()
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
        if "peduncle_error" in obs and obs["peduncle_error"] is not None:
            if not obs["peduncle_cut"]:      # después del corte el tallo se mueve con la cortadora
                cut_errors.append(float(obs["peduncle_error"]))
            elif time_to_cut is None:
                time_to_cut = round(clock(obs) - start, 3)
            if cut_err_at_command is None and float(action[-1]) >= CLOSE_COMMAND:
                cut_err_at_command = cut_errors[-1] if cut_errors else 0.0
            if cut_err_at_closed is None and float(np.ravel(obs["gripper_position"])[0]) >= CLOSED_FRACTION:
                cut_err_at_closed = cut_errors[-1] if cut_errors else 0.0
        if not obs["recording"]:   # el servidor cortó al llegar a trial_seconds
            break

    result = {"folder": folder, "frames": frames, "time_to_success": time_to_success,
              "stopped_on_success": int(stopped),
              **occlusion, **leaf_info}
    if track_camera:
        result.update(track_summary(track_camera, samples, time_to_success))
        with open(os.path.join(folder, TRACK_FILE), "w") as f:
            json.dump({"camera": track_camera, "every": track_every,
                       "samples": [{"t": t, "occlusion": o} for t, o in samples]}, f)
    chunks = latencies[chunks_before:]
    if chunks:   # el primer bloque del primer ensayo incluye la carga de CUDA: la mediana lo absorbe
        result.update(chunk_latency_ms=round(float(np.median(chunks)), 2), chunks=len(chunks))
    if agent.ik_orientation != "off":
        result["ik_fallbacks"] = agent.ik_counts["fallback"] - ik_before["fallback"]
    if agent.waypoint_tolerance is not None:
        # acciones terminadas al llegar y por agotar waypoint_max_steps
        result["waypoints_reached"] = agent.waypoint_counts["reached"] - waypoints_before["reached"]
        result["waypoint_timeouts"] = agent.waypoint_counts["timeout"] - waypoints_before["timeout"]
    if getattr(agent, "hold_while_closing", False):
        result["closing_holds"] = agent.hold_count - holds_before
    if distances:
        result.update(final_dist=distances[-1], min_dist=min(distances),
                      final_grasp_err=grasp_errors[-1], min_grasp_err=min(grasp_errors))
    if "peduncle_error" in obs and obs["peduncle_error"] is not None:
        # éxito del corte: cortado y con el tomate sujeto al final (cut-and-hold)
        result.update(min_cut_err=min(cut_errors) if cut_errors else 0.0, time_to_cut=time_to_cut,
                      cut_err_at_command=cut_err_at_command, cut_err_at_closed=cut_err_at_closed,
                      peduncle_cut=int(bool(obs["peduncle_cut"])), peduncle_held=int(bool(obs["peduncle_held"])))
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
                                  waypoint_max_steps=args.waypoint_max_steps,
                                  noise_scheduler=args.noise_scheduler,
                                  num_inference_steps=args.num_inference_steps,
                                  gripper_threshold=args.gripper_threshold,
                                  hold_while_closing=args.hold_while_closing,
                                  latch_closed=args.latch_closed)

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
              "time_to_success", "stopped_on_success", "success_extra_seconds", "n_action_steps",
              "noise_scheduler", "num_inference_steps", "chunk_latency_ms", "chunks"]
    if first_obs.get("peduncle_error") is not None:   # escena con cortadora: métricas del corte
        fields += ["min_cut_err", "cut_err_at_command", "cut_err_at_closed", "time_to_cut", "peduncle_cut",
                   "peduncle_held"]
    if targets != [None]:
        fields += ["occlusion_target", "leaf_status", "leaf", "leaf_camera", "occlusion_fit"]
    elif not manual:
        # la hoja de una corrida anterior con --occlusion_targets sigue en el servidor: se quita
        call(robot, "place_trial_leaf", leaf_dir=args.leaf_dir, target=0.0, key=0, seed=args.leaf_seed)
    if args.ik_orientation != "off":
        fields += ["ik_fallbacks"]
    if args.gripper_threshold is not None:
        fields += ["gripper_threshold"]
    if args.hold_while_closing:
        fields += ["closing_holds"]
    if args.blank_cameras:
        fields += ["blank_cameras"]
    unknown = set(args.blank_cameras) - set(policy_cameras(agent))
    if unknown:
        raise SystemExit(f"--blank_cameras: la política no usa {sorted(unknown)} (usa {policy_cameras(agent)})")
    if args.waypoint_tolerance is not None:
        fields += ["waypoint_tolerance", "waypoint_max_steps", "waypoints_reached", "waypoint_timeouts"]
    if not manual:
        fields += list(occlusion_columns(call(robot, "measure_occlusion")))
    if args.track_occlusion:
        if manual or args.track_occlusion not in first_obs:
            raise SystemExit(f"--track_occlusion {args.track_occlusion}: solo en el sim y con una cámara del servidor")
        fields += track_columns(args.track_occlusion)
    grasp_point = load_grasp_point()
    pregrasp = {}   # pos_idx -> articulaciones de pregrasp (--leaf_camera wrist_rgb)
    if args.leaf_camera and targets != [None]:
        if args.leaf_camera not in first_obs:
            raise SystemExit(f"--leaf_camera {args.leaf_camera}: el servidor no tiene esa cámara")
        if "wrist" in args.leaf_camera:
            from plan_trajectories import ARM_JOINTS, Kinematics
            kin = Kinematics(call(robot, "robot_urdf"))
            call(robot, "reset_robot")
            sim_q = robot.get_observations()["sim_joint_positions"]
            seed = [sim_q[i] for i in kin.movable[:ARM_JOINTS]]
            for pos_idx in positions:
                apple_pos = here if args.here else apple_grid[pos_idx]
                pregrasp[pos_idx] = pregrasp_q(kin, apple_pos, grasp_point, seed,
                                               args.leaf_pregrasp, args.leaf_pitch_deg)
                if pregrasp[pos_idx] is None:
                    print(f"[Trials] AVISO: sin pregrasp para la posición {pos_idx}: la hoja se ajusta "
                          "con el brazo en la pose inicial")
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
                if leaf is not None and args.leaf_camera:
                    leaf.update(cameras=[args.leaf_camera], reference_q=pregrasp.get(pos_idx))
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
                                "gripper_threshold": args.gripper_threshold,
                                "hold_while_closing": args.hold_while_closing,
                                "latch_closed": args.latch_closed,
                                "waypoint_max_steps": args.waypoint_max_steps if args.waypoint_tolerance else None,
                                "checkpoint": checkpoint, "run_name": run_name,
                                "blank_cameras": list(args.blank_cameras),
                                "leaf_reference": "pregrasp" if pregrasp.get(pos_idx) else None,
                                "leaf_pregrasp": args.leaf_pregrasp if pregrasp else None}
                    result = run_trial(robot, agent, None if (args.here or manual) else apple_pos,
                                       args.trial_seconds, metadata, root, grasp_point,
                                       reset=not manual, measure_occlusion=not manual, leaf=leaf,
                                       stop_on_success=args.stop_on_success and not manual,
                                       success_extra_seconds=args.success_extra_seconds,
                                       blank_cameras=args.blank_cameras,
                                       track_camera=args.track_occlusion, track_every=args.track_every)
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
                           "n_action_steps": agent.policy.config.n_action_steps,
                           "noise_scheduler": agent.sampler[0], "num_inference_steps": agent.sampler[1]}
                    if args.waypoint_tolerance is not None:
                        row.update(waypoint_tolerance=args.waypoint_tolerance,
                                   waypoint_max_steps=args.waypoint_max_steps)
                    if args.gripper_threshold is not None:
                        row["gripper_threshold"] = args.gripper_threshold
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
