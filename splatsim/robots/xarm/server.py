"""
Servidor PyBullet + Gaussian Splat para el xArm5 Lite.
Clase delgada: orquesta componentes que viven en sus propios archivos.
"""
import time  # noqa: I001

import numpy as np

from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase
from splatsim.utils.robot_splat_render_utils import get_curr_link_states

from gaussian_renderer import render

from .apple import AppleController
from .cameras import AppleCamera, BaseCamera, WristCamera
from .config import (
    ENV_CONFIG,
    FLOOR_Z,
    LOOP_HZ,
    MAX_RECORD_SECONDS,
    NUM_DOFS,
    RECORD_EVERY,
)
from .debug_ui import AppleWindow, BaseCameraTuner, ResetJointsButton
from .gripper import XArm5Gripper
from .recorder import TrajectoryRecorder
from .trail import EETrail


class XArm5PybulletRobotServerCamera(PybulletRobotServerBase):
    ENV_CONFIG = ENV_CONFIG

    def __init__(self, traj_save_path: str | None = None, move_apple: bool = False,
                 show_apple_window: bool = True, **kwargs):
        # traj_save_path se acepta por compatibilidad pero no se usa (igual que antes)
        self._num_dofs = NUM_DOFS
        # el recorder no toca la base en su constructor, así que es seguro crearlo antes
        self.recorder = TrajectoryRecorder(self, LOOP_HZ, MAX_RECORD_SECONDS, RECORD_EVERY)
        self.gripper = None  # se crea en setup_gripper(), que la base llama en su __init__
        super().__init__(**kwargs)

        self._remove_wall()
        self._lower_floor(FLOOR_Z)

        self.apple = AppleController(self, movable=move_apple)
        self.cameras = {}
        base_cam = BaseCamera(self)
        self.cameras[base_cam.name] = base_cam
        wrist_cam = WristCamera(self)
        self.cameras[wrist_cam.name] = wrist_cam
        apple_cam = AppleCamera(self, self.apple)
        self.cameras[apple_cam.name] = apple_cam

        self.base_cam_tuner = BaseCameraTuner(self, base_cam)
        self.reset_button = ResetJointsButton(self)
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
        arm = [self.pybullet_client.getJointState(self.dummy_robot, i)[0] for i in range(1, 6)]
        return np.array([*arm, self.gripper.get_state()])

    def get_joint_state_dummy(self) -> np.ndarray:
        return self.get_joint_state()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs()
        for i in range(1, 6):
            self.pybullet_client.setJointMotorControl2(
                self.dummy_robot, i, self.pybullet_client.POSITION_CONTROL,
                targetPosition=joint_state[i - 1], force=500)
        if self.use_gripper:
            self.gripper.command(joint_state[-1])
        # avanzar la sim aquí para que los links se muevan antes de get_observations()
        for _ in range(20):
            self.pybullet_client.stepSimulation()

    def render_image(self, camera_name, cached_link_states=None):
        if camera_name not in self.cameras:
            raise ValueError(f"Unknown camera name {camera_name}")
        camera = self.cameras[camera_name].get()
        if camera is None:
            return None
        return render(camera, self.robot_gaussian, self.pipeline, self.background)["render"]

    def get_observations(self):
        obs = super().get_observations()
        obs["joint_positions"] = self.get_joint_state()
        self.apple_window.set(obs.get("apple_rgb"))
        return obs

    # ── loop principal ───────────────────────────────────────────────────────
    def serve_loop(self):
        self.gripper.patch_for_splat()      # solo actúa la primera vez
        self.apple.update()
        self.base_cam_tuner.update()
        self.reset_button.update()
        self.recorder.tick()
        self.trail.update(active=self.recorder.active)
        self.pybullet_client.stepSimulation()
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
