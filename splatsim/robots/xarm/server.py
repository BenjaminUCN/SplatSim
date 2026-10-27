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
from .config import (
    ENV_CONFIG,
    FLOOR_Z,
    IDLE_STEP_AFTER,
    LOOP_HZ,
    MAX_RECORD_SECONDS,
    NUM_DOFS,
    RECORD_EVERY,
    CONTROL_HZ,
    SIM_DT,
)
from .debug_ui import (
    AppleWindow,
    BaseCameraTuner,
    ResetJointsButton,
    SaveCameraPosesButton,
)
from .gripper import XArm5Gripper
from .recorder import TrajectoryRecorder
from .trail import EETrail


class XArm5PybulletRobotServerCamera(PybulletRobotServerBase):
    ENV_CONFIG = ENV_CONFIG
    # métodos que el cliente puede llamar por ZMQ además de los de la base
    REMOTE_METHODS = ("reset_robot", "start_recording", "stop_recording")

    def __init__(self, traj_save_path: str | None = None, move_apple: bool = False,
                 show_apple_window: bool = True, control_hz: int = CONTROL_HZ,
                 gui: str = "viser", panel_port: int = 8080, **kwargs):
        """gui: "viser" (panel web en http://localhost:<panel_port>) o "pybullet"
        (sliders y botones de PyBullet; --move_apple agrega los de la manzana)."""
        # traj_save_path se acepta por compatibilidad pero no se usa (igual que antes)
        self._num_dofs = NUM_DOFS
        self.control_hz = control_hz
        self.zmq_port = kwargs.get("port", 5556)  # para lanzar clientes desde el panel
        # cada comando avanza exactamente 1/control_hz s simulados
        self.steps_per_command = round((1 / control_hz) / SIM_DT)
        print(f"[xArm5] control_hz={control_hz} → {self.steps_per_command} pasos por comando")
        # el hilo ZMQ, el serve_loop y el recorder comparten PyBullet y el renderer
        self.lock = threading.RLock()
        self.sim_steps = 0               # pasos de simulación dados vía _step()
        self._last_images = {}           # cámaras de la última observación (vista previa del panel)
        self._last_images_time = 0.0
        self._last_command_time = 0.0    # time.monotonic() del último comando
        # el recorder no toca la base en su constructor, así que es seguro crearlo antes
        self.recorder = TrajectoryRecorder(
            self, round(MAX_RECORD_SECONDS / SIM_DT), RECORD_EVERY)
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
            from .panel import ControlPanel  # viser solo hace falta con este modo
            c = self.pybullet_client
            c.configureDebugVisualizer(c.COV_ENABLE_GUI, 0)  # oculta el panel lateral de PyBullet
            self.controls = [ControlPanel(self, base_cams, port=panel_port)]
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

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs()
        with self.lock:
            for i in range(1, 6):
                self.pybullet_client.setJointMotorControl2(
                    self.dummy_robot, i, self.pybullet_client.POSITION_CONTROL,
                    targetPosition=joint_state[i - 1], force=500)
            if self.use_gripper:
                self.gripper.command(joint_state[-1])
            # cada comando avanza exactamente 1/control_hz s simulados,
            # así los frames grabados quedan con un dt fijo
            self._step(self.steps_per_command)
            self._last_command_time = time.monotonic()
            self.recorder.on_command()

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

    def render_preview(self, camera_names) -> dict:
        """Renderiza las cámaras con la escena actual, sin pasar por get_observations
        (que alimenta al recorder). Lo usa la vista previa del panel."""
        c = self.pybullet_client
        with self.lock:
            data = {}
            for name, body in zip(self.splat_object_name_list, self.urdf_object_list):
                data[f"{name}_position"], data[f"{name}_orientation"] = \
                    c.getBasePositionAndOrientation(body)
            cached = get_curr_link_states(self.dummy_robot, self.use_link_centers)
            self.prep_image_rendering(data=data, cached_link_states=cached)
            return {name: self.render_image(name) for name in camera_names}

    def recent_images(self, max_age: float) -> dict:
        """Imágenes de la última observación pedida por un cliente, si tiene menos de max_age s."""
        if time.monotonic() - self._last_images_time > max_age:
            return {}
        return dict(self._last_images)

    def get_observations(self):
        with self.lock:
            obs = super().get_observations()
            obs["joint_positions"] = self.get_joint_state()
            obs["sim_time"] = self.sim_steps * SIM_DT
            self._last_images = {k: obs[k] for k in self.camera_names if obs.get(k) is not None}
            self._last_images_time = time.monotonic()
            self.recorder.on_observation(obs)
            obs["recording"] = self.recorder.active  # tras on_observation: refleja el auto-stop
            obs["reset_joint_command"] = self.reset_joint_command()
        self.apple_window.set(obs.get("apple_rgb"))
        return obs

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

    def start_recording(self, max_seconds: float | None = None, metadata: dict | None = None,
                        root: str | None = None) -> str | None:
        with self.lock:
            max_steps = round(max_seconds / SIM_DT) if max_seconds else None
            return self.recorder.start(max_steps, metadata, root)

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
            # mientras se graba solo los comandos avanzan la física (dt fijo por frame,
            # aunque el cliente se pause, p.ej. durante la inferencia de la política);
            # en reposo se sigue simulando para poder mover el robot desde la GUI
            idle = time.monotonic() - self._last_command_time > IDLE_STEP_AFTER
            if idle and not self.recorder.active:
                self._step()
        time.sleep(1 / LOOP_HZ)
        self.apple_window.show()

    def serve(self) -> None:
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

        self._zmq_server_thread.start()
        print("Ready to serve.")
        while True:
            self.serve_loop()
