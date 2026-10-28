"""
Servidor PyBullet + Gaussian Splat para el xArm5 Lite.
Clase delgada: orquesta componentes que viven en sus propios archivos.
"""
import threading  # noqa: I001
import time

import numpy as np

from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase
from splatsim.utils.camera_setup import base_camera_poses
from splatsim.utils.robot_splat_render_utils import get_curr_link_states

from gaussian_renderer import render

from .apple import AppleController
from .cameras import AppleCamera, BaseCamera, WristCamera
from splatsim.robots.joint_controller import ControllerGains, JointController

from .config import (
    ENV_CONFIG,
    FLOOR_Z,
    IDLE_STEP_AFTER,
    MAX_REALTIME_STEP,
    LOW_LEVEL_HZ,
    LOOP_HZ,
    MAX_RECORD_SECONDS,
    NUM_DOFS,
    RECORD_EVERY,
    CONTROL_HZ,
    SIM_CLOCKS,
    SIM_DT,
    SIM_MAX_DELTA,
    TELEOP_MAX_RECORD_SECONDS,
)
from .debug_ui import (
    AppleWindow,
    BaseCameraTuner,
    ResetJointsButton,
    SaveCameraPosesButton,
)
from .gripper import XArm5Gripper
from .preview import PreviewStream
from splatsim.robots.station.operator import OperatorMixin, OperatorPrompt
from splatsim.robots.station.recorder import TrajectoryRecorder, to_uint8_image
from .trail import EETrail


class XArm5PybulletRobotServerCamera(OperatorMixin, PybulletRobotServerBase):
    ENV_CONFIG = ENV_CONFIG
    # métodos que el cliente puede llamar por ZMQ además de los de la base
    REMOTE_METHODS = ("reset_robot", "start_recording", "stop_recording", "get_preview",
                      "operator_prompt", "operator_answer")

    def __init__(self, traj_save_path: str | None = None, move_apple: bool = False,
                 show_apple_window: bool = True, control_hz: int = CONTROL_HZ,
                 gui: str = "viser", panel_port: int = 8080, control_mode: str = "position",
                 sim_clock: str = "fixed", **kwargs):
        """gui: "viser" (panel web en http://localhost:<panel_port>) o "pybullet"
        (sliders y botones de PyBullet; --move_apple agrega los de la manzana).
        control_mode: position / interp / velocity (ver splatsim/robots/joint_controller.py),
        la misma ley que usa el driver real del xArm5.
        sim_clock: quién avanza la física. "fixed": cada comando avanza exactamente 1/control_hz
        simulado (dt fijo por frame aunque el cliente vaya más lento; políticas y ensayos).
        "realtime": el serve_loop la avanza continuamente al ritmo del reloj y los comandos solo
        cambian el objetivo (teleoperación con GELLO); las grabaciones guardan solo el estado,
        un frame por comando, y las imágenes se renderizan después con
        scripts/xarm5/render_recordings.py."""
        if sim_clock not in SIM_CLOCKS:
            raise ValueError(f"sim_clock debe ser uno de {SIM_CLOCKS}, no {sim_clock!r}")
        self.sim_clock = sim_clock
        # traj_save_path se acepta por compatibilidad pero no se usa (igual que antes)
        self._num_dofs = NUM_DOFS
        self.control_hz = control_hz
        self.zmq_port = kwargs.get("port", 5556)  # para lanzar clientes desde el panel
        # cada comando avanza exactamente 1/control_hz s simulados
        self.steps_per_command = round((1 / control_hz) / SIM_DT)
        # lazo articular a LOW_LEVEL_HZ dentro de cada comando, como el hilo del driver real
        self.steps_per_tick = round((1 / LOW_LEVEL_HZ) / SIM_DT)
        # en sim el tope por tick es más holgado que en el xArm real (ver SIM_MAX_DELTA)
        self.joint_ctrl = JointController(control_mode, 1 / control_hz,
                                          ControllerGains(max_delta=SIM_MAX_DELTA))
        self._holding = True             # en reposo el brazo se sostiene en su posición
        print(f"[xArm5] control_mode={control_mode}")
        print(f"[xArm5] control_hz={control_hz} → {self.steps_per_command} pasos por comando"
              + (" (sim_clock=realtime: la física sigue al reloj)" if sim_clock == "realtime" else ""))
        # el hilo ZMQ, el serve_loop y el recorder comparten PyBullet y el renderer
        self.lock = threading.RLock()
        # el splat deformado (prep_image_rendering) es uno solo: un render a la vez. Se toma
        # después de self.lock (nunca al revés), así la vista previa renderiza sin frenar la física
        self.render_lock = threading.Lock()
        self.sim_steps = 0               # pasos de simulación dados vía _step()
        self._tick_left = 0              # pasos que faltan para el próximo tick del lazo articular
        self._rt_last = time.monotonic() # realtime: última vuelta del serve_loop
        self._rt_acc = 0.0               # realtime: pasos pendientes (fracción incluida)
        self._preview = None             # PreviewStream, se crea con el primer get_preview
        self.operator = OperatorPrompt()  # preguntas de un cliente que el panel muestra
        self._last_images = {}           # cámaras de la última observación (vista previa del panel)
        self._last_images_time = 0.0
        self._last_command_time = 0.0    # time.monotonic() del último comando
        # el recorder no toca la base en su constructor, así que es seguro crearlo antes
        self.recorder = TrajectoryRecorder(
            self, MAX_RECORD_SECONDS, RECORD_EVERY, state_only_max_seconds=TELEOP_MAX_RECORD_SECONDS)
        self.gripper = None  # se crea en setup_gripper(), que la base llama en su __init__
        super().__init__(**kwargs)

        self._remove_wall()
        self._lower_floor(FLOOR_Z)

        self.apple = AppleController(self, movable=move_apple and gui == "pybullet")
        base_cams = [BaseCamera(self, name, xyz, rpy)
                     for name, (xyz, rpy) in base_camera_poses().items()]
        self.cameras = {cam.name: cam for cam in base_cams}
        wrist_cam = WristCamera(self)
        self.cameras[wrist_cam.name] = wrist_cam
        apple_cam = AppleCamera(self, self.apple)
        self.cameras[apple_cam.name] = apple_cam
        unknown = set(self.camera_names) - set(self.cameras)
        if unknown:
            raise ValueError(f"Cámaras desconocidas {sorted(unknown)}; disponibles: {sorted(self.cameras)}")

        # controles: cada uno tiene update(), que el serve_loop llama con el lock tomado
        if gui == "viser":
            from .panel import SimPanel  # viser solo hace falta con este modo
            c = self.pybullet_client
            c.configureDebugVisualizer(c.COV_ENABLE_GUI, 0)  # oculta el panel lateral de PyBullet
            self.controls = [SimPanel(self, base_cams, port=panel_port)]
        elif gui == "pybullet":
            self.controls = [*(BaseCameraTuner(self, cam) for cam in base_cams),
                             SaveCameraPosesButton(self, base_cams),
                             ResetJointsButton(self)]
        else:
            raise ValueError(f"gui debe ser 'viser' o 'pybullet', no {gui!r}")
        self.apple_window = AppleWindow(show_apple_window)
        self.trail = EETrail(self)
        self.recorder.start_keyboard_listener()

    # ── escena ───────────────────────────────────────────────────────────────
    def _remove_wall(self):
        if getattr(self, "wall", None) is not None:
            try:
                self.pybullet_client.removeBody(self.wall)
                self.wall = None
            except Exception:  # noqa: BLE001, S110
                pass

    def _lower_floor(self, z):
        if getattr(self, "plane", None) is not None:
            try:
                self.pybullet_client.resetBasePositionAndOrientation(
                    self.plane, [0, 0, z],
                    self.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
            except Exception:  # noqa: BLE001, S110
                pass

    # ── métodos que llama la base: delegadores de una línea ──────────────────
    def setup_gripper(self):
        self.gripper = XArm5Gripper(self)
        self.gripper.setup()

    def move_gripper(self, open_length, velocity=2):
        self.gripper.move(open_length, velocity)

    def num_dofs(self) -> int:
        return NUM_DOFS

    def get_joint_state(self) -> np.ndarray:
        """6 DOF: 5 joints del brazo + gripper normalizado."""
        with self.lock:
            arm = [self.pybullet_client.getJointState(self.dummy_robot, i)[0]
                   for i in range(1, 6)]
            return np.array([*arm, self.gripper.get_state()])

    def get_joint_state_dummy(self) -> np.ndarray:
        return self.get_joint_state()

    def _step(self, n: int = 1) -> None:
        for _ in range(n):
            self.pybullet_client.stepSimulation()
        self.sim_steps += n

    def _advance(self, n: int) -> None:
        """Avanza n pasos con el lazo articular: un tick del controlador cada steps_per_tick."""
        while n > 0:
            if self._tick_left == 0:
                kind, values = self.joint_ctrl.step(
                    self._arm_positions(), self.sim_steps * SIM_DT, self.steps_per_tick * SIM_DT)
                self._apply_arm_command(kind, values)
                self._tick_left = self.steps_per_tick
            k = min(n, self._tick_left)
            self._step(k)
            self._tick_left -= k
            n -= k

    def _arm_positions(self) -> np.ndarray:
        return np.array([self.pybullet_client.getJointState(self.dummy_robot, i)[0]
                         for i in range(1, 6)])

    def _apply_arm_command(self, kind: str, values) -> None:
        c = self.pybullet_client
        for i in range(1, 6):
            if kind == "position":
                c.setJointMotorControl2(self.dummy_robot, i, c.POSITION_CONTROL,
                                        targetPosition=values[i - 1], force=500)
            else:
                c.setJointMotorControl2(self.dummy_robot, i, c.VELOCITY_CONTROL,
                                        targetVelocity=values[i - 1], force=500)

    def set_control_mode(self, mode: str) -> None:
        """Cambia el modo de control; el próximo comando parte de la posición actual."""
        with self.lock:
            self.joint_ctrl = JointController(mode, self.joint_ctrl.nominal_period, self.joint_ctrl.g)
            self._hold()
        print(f"[xArm5] control_mode={mode}")

    def _hold(self) -> None:
        """Sostiene el brazo donde está (p.ej. al quedar sin comandos en modo velocidad)."""
        q = self._arm_positions()
        self._apply_arm_command("position", q)
        self.joint_ctrl.reset(q, self.sim_steps * SIM_DT)
        self._holding = True

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs()
        with self.lock:
            now = self.sim_steps * SIM_DT
            if self._holding:            # retoma desde la posición actual, sin saltos
                self.joint_ctrl.reset(self._arm_positions(), now)
                self._holding = False
            elif self.sim_clock == "realtime":
                # periodo real entre objetivos para interp/velocity (el cliente puede ir bajo
                # control_hz, p.ej. si pide imágenes en cada vuelta)
                period = min(time.monotonic() - self._last_command_time, MAX_REALTIME_STEP)
                self.joint_ctrl.nominal_period = 0.8 * self.joint_ctrl.nominal_period + 0.2 * period
            self.joint_ctrl.set_target(joint_state[:5], now)
            if self.use_gripper:
                self.gripper.command(joint_state[-1])
            if self.sim_clock == "fixed":
                # cada comando avanza exactamente 1/control_hz s simulados (frames con dt fijo),
                # empezando un tick del lazo articular
                self._tick_left = 0
                self._advance(self.steps_per_command)
            # con realtime la física la avanza el serve_loop
            self._last_command_time = time.monotonic()
            self.recorder.on_command()

    def _idle(self) -> bool:
        """Sin comandos recientes: el cliente se detuvo (o no hay)."""
        after = IDLE_STEP_AFTER
        if self.sim_clock == "realtime":   # un cliente lento no cuenta como detenido
            after = max(after, 3 * self.joint_ctrl.nominal_period)
        return time.monotonic() - self._last_command_time > after

    def set_control_hz(self, hz: int) -> str | None:
        """Cambia la frecuencia de comandos (el --hz con que se lanzó); devuelve el motivo si no se
        puede. Debe dividir la frecuencia de la física para que cada comando sea un número entero
        de pasos (dt exacto por frame)."""
        steps = (1 / hz) / SIM_DT if hz > 0 else 0
        if hz <= 0 or abs(steps - round(steps)) > 1e-6:
            divisors = [d for d in range(1, 61) if round(1 / SIM_DT) % d == 0]
            return (f"{hz} Hz does not divide the physics rate ({round(1 / SIM_DT)} Hz); "
                    f"use one of {', '.join(map(str, divisors))}")
        with self.lock:
            if self.recorder.active:
                return "stop the recording first (its frames would mix two time steps)"
            self.control_hz = hz
            self.steps_per_command = round(steps)
            self.joint_ctrl.nominal_period = 1 / hz
        print(f"[xArm5] control_hz={hz} → {self.steps_per_command} pasos por comando")
        return None

    def set_sim_clock(self, clock: str) -> None:
        if clock not in SIM_CLOCKS:
            raise ValueError(f"sim_clock debe ser uno de {SIM_CLOCKS}, no {clock!r}")
        with self.lock:
            if self.recorder.active:
                self.recorder.stop()     # una grabación no mezcla frames con y sin imágenes
            self.sim_clock = clock
            self.joint_ctrl.nominal_period = 1 / self.control_hz
            self._rt_last, self._rt_acc = time.monotonic(), 0.0
        print(f"[xArm5] sim_clock={clock}")

    def set_object_pose(self, *args, **kwargs) -> None:
        with self.lock:  # también se llama desde el hilo ZMQ
            super().set_object_pose(*args, **kwargs)

    def render_image(self, camera_name, cached_link_states=None):
        if camera_name not in self.cameras:
            raise ValueError(f"Unknown camera name {camera_name}")
        camera = self.cameras[camera_name].get()
        if camera is None:
            return None
        return render(camera, self.robot_gaussian, self.pipeline, self.background)["render"]

    # ── interfaz de backend (splatsim/robots/station: panel y recorder) ──────
    def clock(self) -> float:
        """Tiempo de las grabaciones: el simulado."""
        return self.sim_steps * SIM_DT

    @property
    def record_state_only(self) -> bool:
        return self.sim_clock == "realtime"

    @property
    def control_mode(self) -> str:
        return self.joint_ctrl.mode

    @property
    def preview_cameras(self) -> list[str]:
        """Cámaras que se pueden ver en el panel (todas las del splat, no solo las grabadas)."""
        return list(self.cameras)

    def preview_images(self, names, max_age: float) -> dict:
        """uint8 HxWx3 por cámara: las de la última observación o la vista previa en curso si
        tienen menos de max_age s; las que falten se renderizan."""
        images = self.recent_images(max_age)
        missing = [n for n in names if n not in images]
        if missing:
            images.update(self.render_preview(missing))
        return {n: to_uint8_image(images[n]) for n in names if images.get(n) is not None}

    def render_cameras(self, camera_names) -> dict:
        """Renderiza las cámaras con la escena actual, sin pasar por get_observations (que
        alimenta al recorder). Solo toma self.lock para copiar el estado de PyBullet: si quien
        llama no lo tiene tomado, la física sigue mientras se renderiza."""
        if getattr(self, "gaussians_backup", None) is None:
            return {}                    # sin cámaras en camera_names no se carga el splat
        c = self.pybullet_client
        with self.lock:
            data = {}
            for name, body in zip(self.splat_object_name_list, self.urdf_object_list):
                data[f"{name}_position"], data[f"{name}_orientation"] = \
                    c.getBasePositionAndOrientation(body)
            cached = get_curr_link_states(self.dummy_robot, self.use_link_centers)
            cameras = {name: self.cameras[name].get() for name in camera_names}
        with self.render_lock:
            self.prep_image_rendering(data=data, cached_link_states=cached)
            return {name: render(cam, self.robot_gaussian, self.pipeline, self.background)["render"]
                    for name, cam in cameras.items() if cam is not None}

    render_preview = render_cameras      # la vista previa del panel

    def recent_images(self, max_age: float) -> dict:
        """Imágenes de la última observación pedida por un cliente (o de la vista previa en
        curso), si tienen menos de max_age s."""
        if time.monotonic() - self._last_images_time > max_age:
            return {}
        return dict(self._last_images)

    def _set_recent_images(self, images: dict) -> None:
        self._last_images = images
        self._last_images_time = time.monotonic()

    def get_preview(self, names, since: int = -1):
        """Vista previa para teleoperar: un hilo renderiza `names` sin parar mientras alguien la
        pida (deja de hacerlo 2 s después de la última petición). Devuelve
        {"id", "sim_time", "images": {nombre: uint8 HxWx3}}, o None si no hay un render más
        nuevo que `since`."""
        if self._preview is None:
            self._preview = PreviewStream(self)
        return self._preview.request(list(names), since)

    def state_observation(self) -> dict:
        """Observación sin imágenes (para grabar con sim_clock=realtime); incluye las posiciones
        de todas las articulaciones de PyBullet para poder renderizarla después."""
        with self.lock:
            obs = super().get_observations(render=False)
            self._add_extras(obs)
        return {k: v for k, v in obs.items() if not k.endswith("_rgb")}

    def _add_extras(self, obs: dict) -> None:
        c = self.pybullet_client
        obs["joint_positions"] = self.get_joint_state()
        obs["sim_time"] = self.sim_steps * SIM_DT
        obs["control_mode"] = self.joint_ctrl.mode
        obs["sim_clock"] = self.sim_clock
        obs["sim_joint_positions"] = np.array(
            [c.getJointState(self.dummy_robot, i)[0] for i in range(c.getNumJoints(self.dummy_robot))])

    def get_observations(self):
        with self.lock, self.render_lock:
            obs = super().get_observations()
            self._add_extras(obs)
            self._set_recent_images({k: obs[k] for k in self.camera_names if obs.get(k) is not None})
            self.recorder.on_observation(obs)
            obs["recording"] = self.recorder.active  # tras on_observation: refleja el auto-stop
            obs["reset_joint_command"] = self.reset_joint_command()
        self.apple_window.set(obs.get("apple_rgb"))
        return obs

    def restore_state(self, obs: dict) -> None:
        """Pone la escena en el estado de una observación grabada (sim_joint_positions y objetos);
        lo usa scripts/xarm5/render_recordings.py."""
        c = self.pybullet_client
        with self.lock:
            for i, q in enumerate(obs["sim_joint_positions"]):
                c.resetJointState(self.dummy_robot, i, q)
                # getLinkState sin FK (p.ej. get_current_ee_pose) devolvería la pose anterior
                c.getLinkState(self.dummy_robot, i, computeForwardKinematics=True)
            for name, body in zip(self.splat_object_name_list, self.urdf_object_list):
                if f"{name}_position" in obs:
                    c.resetBasePositionAndOrientation(
                        body, obs[f"{name}_position"], obs[f"{name}_orientation"])

    # ── control remoto (REMOTE_METHODS) ──────────────────────────────────────
    def reset_joint_command(self) -> np.ndarray:
        """Comando (5 joints + gripper cerrado) que lleva el brazo a la pose de reset."""
        arm = [self.initial_joint_state[i] * self.joint_signs[i] for i in range(5)]
        return np.array([*arm, 0.0])

    def reset_robot(self, settle_steps: int = 120) -> None:
        """Vuelve el brazo a la pose inicial al instante y cierra el gripper."""
        c = self.pybullet_client
        with self.lock:
            for i in range(1, len(self.initial_joint_state)):
                target = self.initial_joint_state[i - 1] * self.joint_signs[i - 1]
                c.resetJointState(self.dummy_robot, i, target)
                c.setJointMotorControl2(self.dummy_robot, i, c.POSITION_CONTROL,
                                        targetPosition=target, force=500)
            self.close_gripper()
            self._step(settle_steps)
            self._holding = True         # el próximo comando parte de esta pose

    def start_recording(self, max_seconds: float | None = None, metadata: dict | None = None,
                        root: str | None = None) -> str | None:
        with self.lock:
            return self.recorder.start(max_seconds, metadata, root)

    def stop_recording(self) -> None:
        with self.lock:
            if self.recorder.active:
                self.recorder.stop()

    # ── loop principal ───────────────────────────────────────────────────────
    def serve_loop(self):
        with self.lock:
            self.gripper.patch_for_splat()      # solo actúa la primera vez
            self.apple.update()
            for control in self.controls:
                control.update()
            self.recorder.tick()
            self.trail.update(active=self.recorder.active)
            now = time.monotonic()
            elapsed = min(now - self._rt_last, MAX_REALTIME_STEP)
            self._rt_last = now
            idle = self._idle()
            if self.sim_clock == "realtime" and not idle and not self._holding:
                # la física sigue al reloj; los comandos solo cambian el objetivo
                self._rt_acc += elapsed / SIM_DT
                n = int(self._rt_acc)
                self._rt_acc -= n
                self._advance(n)
            # con fixed, mientras se graba solo los comandos avanzan la física (dt fijo por
            # frame aunque el cliente se pause, p.ej. durante la inferencia de la política);
            # en reposo se sigue simulando para poder mover el robot desde la GUI
            elif idle and not (self.recorder.active and self.sim_clock == "fixed"):
                if not self._holding:
                    self._hold()
                self._step()
        time.sleep(1 / LOOP_HZ)
        self.apple_window.show()

    def serve(self) -> None:
        self.setup_scene()
        self._zmq_server_thread.start()
        print("Ready to serve.")
        while True:
            self.serve_loop()

    def setup_scene(self) -> None:
        """Pone el robot en la pose inicial y guarda los estados de referencia del render."""
        c = self.pybullet_client
        for i in range(len(self.initial_joint_state)):
            c.setJointMotorControl2(self.dummy_robot, i, c.VELOCITY_CONTROL, force=0)
        for i in range(1, len(self.initial_joint_state)):
            c.resetJointState(self.dummy_robot, i,
                              self.initial_joint_state[i - 1] * self.joint_signs[i - 1])
        self.initial_link_states = get_curr_link_states(self.dummy_robot, self.use_link_centers)

        _, ee_quat = self.get_current_ee_pose()
        self.iniital_ee_quat = ee_quat  # (sic) nombre original, por si la base lo usa

        for i in range(1, self.num_dofs()):
            c.setJointMotorControl2(self.dummy_robot, i, c.VELOCITY_CONTROL,
                                    force=250, maxVelocity=0.2)
        self.close_gripper()
        self.initial_ee_pos, self.initial_ee_quat = self.get_current_ee_pose()

        for _ in range(10000):
            for j in range(1, len(self.initial_joint_state)):
                c.resetJointState(self.dummy_robot, j,
                                  self.initial_joint_state[j - 1] * self.joint_signs[j - 1])
            c.stepSimulation()
        self.gripper.patch_for_splat()
