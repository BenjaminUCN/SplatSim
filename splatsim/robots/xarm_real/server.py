"""Servidor del xArm5 real: el driver (splatsim/robots/xarm5_lite_robot_plant.py) y las cámaras
RealSense detrás del mismo protocolo ZMQ que el servidor del sim, con su recorder y su panel.

Así los mismos clientes (run_env_sim.py con GELLO o con la política, run_policy_trials.py) y el
mismo panel (splatsim/robots/station/panel.py) sirven para ambos. Diferencias con el sim:
  - las imágenes vienen de los camera nodes (scripts/launch_camera_nodes.py), uint8 HxWx3 RGB;
  - Reset Robot mueve el brazo despacio a home_joints (configs/xarm5_real.yaml), no lo teletransporta;
  - las grabaciones usan el reloj (no hay tiempo simulado): un frame por comando, con imágenes;
  - set_object_pose no hace nada (el tomate se pone a mano).
No importa nada del sim (PyBullet, splats).
"""
import threading
import time

import numpy as np

from splatsim.robots.station.operator import OperatorMixin, OperatorPrompt
from splatsim.robots.station.recorder import TrajectoryRecorder
from splatsim.robots.station.zmq_server import ZMQRobotServer, ZMQServerThread

LOOP_HZ = 100            # vueltas por segundo del lazo del servidor (panel, teclado)
RESET_TOLERANCE = 0.02   # rad: Reset Robot termina cuando todas las articulaciones están a esto
RESET_EXTRA_TIME = 3.0   # s de margen sobre la duración prevista del reset antes de rendirse
MAX_CONTROL_HZ = 30      # el lazo de control del driver corre a 30 Hz: comandar más rápido no sirve


def _min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


class XArm5RealServer(OperatorMixin):
    REMOTE_METHODS = ("reset_robot", "start_recording", "stop_recording",
                      "operator_prompt", "operator_answer")

    def __init__(self, robot, cameras: dict, port: int = 6001, host: str = "127.0.0.1",
                 control_hz: int = 30, panel_port: int | None = 8080,
                 home_joints=None, reset_speed: float = 0.3,
                 path: str = "~/data/bc_data/real", max_record_seconds: float = 120):
        """robot: el driver XArmRobot (o algo con su interfaz); cameras: nombre -> cliente de
        cámara con read() -> (imagen RGB HxWx3, profundidad), p.ej. gello ZMQClientCamera."""
        self.robot = robot
        self._cameras = cameras
        self.camera_names = list(cameras)
        self.control_hz = control_hz
        self.zmq_port = port
        self.home_joints = None if home_joints is None else np.asarray(home_joints, dtype=float)
        self.reset_speed = reset_speed
        self.path = path
        # el hilo ZMQ y el lazo del servidor (panel, teclado) comparten el driver y el recorder;
        # las cámaras tienen su propio lock (sus clientes ZMQ no se pueden usar desde dos hilos)
        self.lock = threading.RLock()
        self._camera_lock = threading.Lock()
        self._last_images, self._last_images_time = {}, 0.0
        self._t0 = time.monotonic()
        self.recorder = TrajectoryRecorder(self, max_record_seconds, every=1)
        self.operator = OperatorPrompt()   # preguntas de run_policy_trials.py (ensayos a mano)
        self._zmq_server_thread = ZMQServerThread(ZMQRobotServer(self, host=host, port=port))
        self.controls = []
        if panel_port is not None:
            from splatsim.robots.station.panel import StationPanel  # viser solo con panel
            self.controls.append(StationPanel(self, port=panel_port, label="xArm5 real"))
        self.recorder.start_keyboard_listener()

    # ── protocolo de ZMQClientRobot ──────────────────────────────────────────
    def num_dofs(self) -> int:
        return self.robot.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        return self.robot.get_joint_state()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        with self.lock:
            self.robot.command_joint_state(joint_state)
            self.recorder.on_command()

    def set_object_pose(self, *args, **kwargs) -> None:
        """No hace nada: en el robot real los objetos se ponen a mano."""

    def get_observations(self) -> dict:
        with self.lock:
            obs = self.robot.get_observations()
            obs.update(self._read_cameras(self.camera_names))
            obs["time"] = self.clock()
            obs["control_hz"] = self.control_hz   # los clientes de ensayos se pacen con esto
            self.recorder.on_observation(obs)
            obs["recording"] = self.recorder.active   # tras on_observation: refleja el auto-stop
            obs["reset_joint_command"] = self.reset_joint_command()
        return obs

    def _read_cameras(self, names) -> dict:
        images = {}
        with self._camera_lock:
            for name in names:
                image, _depth = self._cameras[name].read()
                images[name] = np.ascontiguousarray(image)
            self._last_images, self._last_images_time = dict(images), time.monotonic()
        return images

    # ── REMOTE_METHODS ───────────────────────────────────────────────────────
    def reset_joint_command(self) -> np.ndarray | None:
        """Comando (5 joints + gripper abierto) de la pose home, o None si no está configurada."""
        if self.home_joints is None:
            return None
        return np.array([*self.home_joints, 0.0])

    def reset_robot(self) -> None:
        """Lleva el brazo despacio a home_joints (perfil de mínimo jerk) y espera a que llegue."""
        if self.home_joints is None:
            print("[xArm5 real] Reset Robot: home_joints no está en configs/xarm5_real.yaml; no se mueve")
            return
        with self.lock:
            q0 = np.asarray(self.robot.get_joint_state(), dtype=float)
            arm0, gripper = q0[:5], q0[5]
            duration = max(np.abs(self.home_joints - arm0).max() / self.reset_speed, 1.0)
            print(f"[xArm5 real] Reset Robot: {duration:.1f} s hasta home")
            start = time.monotonic()
            while True:
                t = time.monotonic() - start
                target = arm0 + _min_jerk(t / duration) * (self.home_joints - arm0)
                self.robot.command_joint_state(np.array([*target, gripper]))
                err = np.abs(np.asarray(self.robot.get_joint_state())[:5] - self.home_joints).max()
                if t >= duration and err < RESET_TOLERANCE:
                    break
                if t > duration + RESET_EXTRA_TIME:
                    print(f"[xArm5 real] Reset Robot: no llegó a home (error {err:.3f} rad)")
                    break
                time.sleep(1 / self.control_hz)

    def start_recording(self, max_seconds: float | None = None, metadata: dict | None = None,
                        root: str | None = None) -> str | None:
        with self.lock:
            return self.recorder.start(max_seconds, metadata, root)

    def stop_recording(self) -> None:
        with self.lock:
            if self.recorder.active:
                self.recorder.stop()

    # ── interfaz de backend (splatsim/robots/station: panel y recorder) ──────
    def clock(self) -> float:
        """Tiempo de las grabaciones: el reloj, desde que arrancó el servidor."""
        return time.monotonic() - self._t0

    record_state_only = False      # las imágenes ya existen: siempre se graban con el frame

    @property
    def control_mode(self) -> str:
        return self.robot.ctrl.mode

    def set_control_mode(self, mode: str) -> None:
        with self.lock:
            self.robot.set_control_mode(mode)

    def set_control_hz(self, hz: int) -> str | None:
        """Cambia la frecuencia de comandos esperada (pacing de Reset Robot y de los ensayos,
        periodo nominal de interp/velocity); devuelve el motivo si no se puede."""
        if not 1 <= hz <= MAX_CONTROL_HZ:
            return f"use 1-{MAX_CONTROL_HZ} Hz (the driver's control loop runs at {MAX_CONTROL_HZ} Hz)"
        with self.lock:
            if self.recorder.active:
                return "stop the recording first"
            self.control_hz = hz
            self.robot.ctrl.nominal_period = 1 / hz
        print(f"[xArm5 real] control_hz={hz}")
        return None

    def get_current_ee_pose(self):
        quat = np.asarray(self.robot.get_observations()["ee_pos_quat"], dtype=float)
        return quat[:3], quat[3:]

    @property
    def preview_cameras(self) -> list[str]:
        return self.camera_names

    def preview_images(self, names, max_age: float) -> dict:
        """Las imágenes de la última observación si tienen menos de max_age s; si no, se leen."""
        if time.monotonic() - self._last_images_time <= max_age and set(names) <= set(self._last_images):
            return {n: self._last_images[n] for n in names}
        return self._read_cameras(names)

    # ── lazo ─────────────────────────────────────────────────────────────────
    def serve(self) -> None:
        self._zmq_server_thread.start()
        print(f"[xArm5 real] Ready to serve (ZMQ {self.zmq_port}, cámaras {self.camera_names})")
        try:
            while True:
                with self.lock:
                    self.recorder.tick()
                    for control in self.controls:
                        control.update()
                time.sleep(1 / LOOP_HZ)
        finally:
            self.robot.stop()
