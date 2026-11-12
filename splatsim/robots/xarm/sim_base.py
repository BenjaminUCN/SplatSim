"""Escena PyBullet + splat del xArm5, sin la clase base del SplatSim original.

XArm5PybulletRobotServerCamera heredaba de splatsim.robots.sim_robot_pybullet_base.
PybulletRobotServerBase, que trae lo del paper con el UR5 (generar demos con un planificador,
poses de agarre, una pared, una posición de "drop", la cámara COLMAP de entrenamiento). Aquí queda
solo lo que usa el xArm5: cargar el robot, los objetos y los splats, deformar el splat del robot
según PyBullet y armar las observaciones. La base del paper queda intacta para el código del UR5.
"""
import os
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import torch
import urdf_models.models_data as md
from gaussian_splatting.arguments import PipelineParams
from gaussian_splatting.gaussian_renderer import GaussianModel

from splatsim.robots.station.zmq_server import ZMQRobotServer, ZMQServerThread
from splatsim.utils.paths import OBJECTS_CONFIG, load_yaml
from splatsim.utils.robot_splat_render_utils import (
    get_curr_link_states,
    get_segmented_indices,
    get_transfomration_list,
    transform_means,
    transform_object,
)

from .config import FLOOR_Z, SCENE_OBJECTS, SIM_DT
from .splat_transform import RobotSplatTransform

JointInfo = namedtuple("JointInfo", ["id", "name", "type", "damping", "friction", "lowerLimit",
                                     "upperLimit", "maxForce", "maxVelocity", "controllable"])
SETTLE_STEPS = 100           # pasos de física tras cargar la escena (los objetos se asientan)
OBJECT_FRICTION = 1.5        # roce lateral de los objetos (el tomate)
SPLAT_BACKGROUND = [1, 1, 1]


def latest_point_cloud(model_path: str) -> Path:
    """point_cloud.ply de la última iteración del entrenamiento del splat (como Scene con
    load_iteration=-1)."""
    root = Path(model_path) / "point_cloud"
    iterations = [int(d.name.rsplit("_", 1)[1]) for d in root.iterdir() if d.name.startswith("iteration_")]
    if not iterations:
        raise FileNotFoundError(f"No hay iteraciones del splat en {root}")
    return root / f"iteration_{max(iterations)}" / "point_cloud.ply"


class XArm5SimBase:
    """Robot, objetos y splats de la escena `robot_name` de configs/object_configs/objects.yaml.

    Las subclases definen setup_gripper(), move_gripper(), get_current_gripper_state(),
    get_joint_state(), num_dofs() y render_image()."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, use_gripper: bool = True,
                 use_link_centers: bool = True, robot_name: str = "xarm5_lite_plant_v3",
                 camera_names: list[str] = ("base_rgb",),
                 object_config_path: str = OBJECTS_CONFIG):
        self.use_link_centers = use_link_centers
        self.robot_name = robot_name
        self.camera_names = list(camera_names)
        # a qué link pertenece cada gaussiana del robot (lo calcula articulated_robot_pipeline.py)
        self.robot_labels = torch.from_numpy(
            np.load(f"./data/labels_path/{robot_name}_labels.npy")).to(device="cuda").long()
        self.transformations_cache = None

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self.pybullet_client = p
        # sin pantalla (pods): la cinemática es la misma, las imágenes vienen de los splats
        p.connect(p.DIRECT if os.environ.get("SPLATSIM_HEADLESS") else p.GUI)

        self.object_config_path = object_config_path
        self.object_config = load_yaml(object_config_path)
        # la escena puede usar otra versión de un objeto (el tomate con pedúnculo) con su mismo nombre
        for name, replacement in self.object_config[robot_name].get("object_overrides", {}).items():
            self.object_config[name] = self.object_config[replacement]
        scene = self.object_config[robot_name]

        urdf_path = scene["urdf_path"][0]
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.dummy_robot = p.loadURDF(urdf_path, useFixedBase=True, basePosition=scene["base_position"][0],
                                      flags=p.URDF_USE_IMPLICIT_CYLINDER)

        self.use_gripper = use_gripper
        if use_gripper:
            self.setup_gripper()

        # joint_states trae un 0 extra al principio (la articulación del mundo)
        self.initial_joint_state = scene["joint_states"][0][1:]
        num_joints = p.getNumJoints(self.dummy_robot)
        if len(self.initial_joint_state) > num_joints:
            print(f"Warning: Provided initial joint positions ({len(self.initial_joint_state)}) exceed "
                  f"the number of joints ({num_joints}). Truncating to {num_joints} positions.")
            self.initial_joint_state = self.initial_joint_state[:num_joints]
        self.joint_signs = [1] * len(self.initial_joint_state)

        self._load_objects()
        for i in range(1, len(self.initial_joint_state)):
            p.resetJointState(self.dummy_robot, i, self.initial_joint_state[i - 1])

        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", [0, 0, FLOOR_Z])
        p.setTimeStep(SIM_DT)

        self.current_gripper_action = 0.0
        # carpeta por defecto de las grabaciones (configs/folder_configs.yaml)
        self.path = load_yaml("configs/folder_configs.yaml")["traj_folder"]

        for _ in range(SETTLE_STEPS):
            p.stepSimulation()

        self._load_splats()

    # ── carga ────────────────────────────────────────────────────────────────
    def _load_objects(self) -> None:
        """Los objetos de SCENE_OBJECTS: su URDF de urdf_models y su splat."""
        self.object_name_list = [o["object_name"] for o in SCENE_OBJECTS]
        self.splat_object_name_list = [o["splat_object_name"] for o in SCENE_OBJECTS]
        for name in [self.robot_name, *self.splat_object_name_list]:
            self.populate_transformations_cache(
                name, np.array(self.object_config[name]["transformation"]["matrix"]))

        models_lib = md.model_lib()
        self.urdf_object_list, self.urdf_object_mass_list = [], []
        for name in self.object_name_list:
            if name not in models_lib.model_name_list:
                raise FileNotFoundError(f"{name} no está en urdf_models (submodules/pybullet-URDF-models)")
            body = p.loadURDF(models_lib[name], [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))
            self.urdf_object_list.append(body)
            self.urdf_object_mass_list.append(p.getDynamicsInfo(body, -1)[0])
            p.changeDynamics(body, -1, lateralFriction=OBJECT_FRICTION)
            p.changeDynamics(body, -1, rollingFriction=0)

    def _load_splats(self) -> None:
        """Splats de los objetos y, si se renderiza alguna cámara, el de la escena con el robot."""
        self.robot_gaussian = GaussianModel(3)   # escena + objetos deformados (prep_image_rendering)
        self.object_gaussians = [GaussianModel(3) for _ in self.urdf_object_list]
        for gaussian, name in zip(self.object_gaussians, self.splat_object_name_list):
            gaussian.load_ply(self.object_config[name]["ply_path"])

        self.gaussians_backup = self.pipeline = self.background = None
        if not self.camera_names:
            return                                  # sin cámaras: solo física, sin GPU para el splat
        model_path = self.object_config[self.robot_name]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.pipeline = PipelineParams(ArgumentParser())
        self.gaussians_backup = GaussianModel(3)    # el splat de la escena, sin deformar
        self.gaussians_backup.load_ply(str(latest_point_cloud(model_path)))
        self.background = torch.tensor(SPLAT_BACKGROUND, dtype=torch.float32, device="cuda")

    def populate_transformations_cache(self, object_name, object_transformation) -> None:
        """Transformación PyBullet -> splat del objeto, su inversa y sus escalas, en la GPU."""
        if self.transformations_cache is None:
            self.transformations_cache = {}
        T = torch.tensor(object_transformation).to(device="cuda").float()
        T_inv = torch.inverse(T)
        self.transformations_cache[object_name] = {
            "transformation": T,
            "transformation_scale": torch.pow(torch.linalg.det(T[:3, :3]), 1 / 3),
            "inv_transformation": T_inv,
            "inv_transformation_scale": torch.pow(torch.linalg.det(T_inv[:3, :3]), 1 / 3),
        }

    def parse_joint_info(self) -> None:
        """self.joints con la info de cada articulación; deja libres (sin motor) las móviles."""
        self.joints = []
        for i in range(p.getNumJoints(self.dummy_robot)):
            info = p.getJointInfo(self.dummy_robot, i)
            controllable = info[2] != p.JOINT_FIXED
            if controllable:
                p.setJointMotorControl2(self.dummy_robot, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints.append(JointInfo(info[0], info[1].decode("utf-8"), info[2], info[6], info[7],
                                         info[8], info[9], info[10], info[11], controllable))

    # ── objetos ──────────────────────────────────────────────────────────────
    def set_object_pose(self, object_name: str, position, orientation, use_gravity: bool = True) -> None:
        if object_name not in self.splat_object_name_list:
            print(f"Object {object_name} not found in splat_object_name_list.")
            return
        i = self.splat_object_name_list.index(object_name)
        p.resetBasePositionAndOrientation(self.urdf_object_list[i], position, orientation)
        # sin gravedad el objeto queda fijo (masa 0)
        p.changeDynamics(self.urdf_object_list[i], -1, mass=self.urdf_object_mass_list[i] if use_gravity else 0)

    def close_gripper(self) -> None:
        self.move_gripper(0.0)
        self.current_gripper_action = 0.0

    def get_current_ee_pose(self):
        state = p.getLinkState(self.dummy_robot, 6)
        return state[0], state[1]

    # ── render ───────────────────────────────────────────────────────────────
    def prep_image_rendering(self, data, cached_link_states=None) -> None:
        """Arma self.robot_gaussian: la escena con los links del robot movidos según PyBullet y los
        objetos en sus poses de `data` (<objeto>_position / _orientation).

        Los links se mueven con RobotSplatTransform (vectorizado); SPLATSIM_LEGACY_TRANSFORM=1 usa
        las funciones por link de robot_splat_render_utils (mismo resultado, ~10 veces más lento)."""
        robot_transformation = self.object_config[self.robot_name]["transformation"]["matrix"]
        aabb = self.object_config[self.robot_name]["aabb"]["bounding_box"]
        if os.environ.get("SPLATSIM_LEGACY_TRANSFORM"):
            transformations_list = get_transfomration_list(
                self.dummy_robot, self.initial_link_states, cached_link_states=cached_link_states)
            segmented_list, xyz = get_segmented_indices(
                self.dummy_robot, self.gaussians_backup, robot_transformation, aabb, self.robot_name,
                robot_labels=self.robot_labels, transformations_cache=self.transformations_cache)
            xyz, rot, opacity, shs_featrest, shs_dc = transform_means(
                self.dummy_robot, self.gaussians_backup, xyz, segmented_list, transformations_list,
                robot_transformation, robot_name=self.robot_name, transformations_cache=self.transformations_cache)
        else:
            if getattr(self, "_robot_transform", None) is None:
                cache = self.transformations_cache[self.robot_name]
                self._robot_transform = RobotSplatTransform(
                    self.dummy_robot, self.gaussians_backup, self.robot_labels, cache["transformation"],
                    cache["transformation_scale"], cache["inv_transformation"], aabb)
            # get_transfomration_list usa los centros de los links si no le pasan cached_link_states
            links = cached_link_states or get_curr_link_states(self.dummy_robot, use_link_centers=True)
            xyz, rot, opacity, shs_featrest, shs_dc = self._robot_transform(self.initial_link_states, links)

        parts = {"xyz": [xyz], "rot": [rot], "opacity": [opacity], "scales": [self.gaussians_backup._scaling],
                 "dc": [shs_dc], "rest": [shs_featrest]}
        # el splat de cada objeto solo cambia si cambia su pose: con la misma pose exacta se reutiliza
        # (en un ensayo el tomate no tiene gravedad y casi siempre queda quieto)
        object_cache = self.__dict__.setdefault("_object_splat_cache", {})
        for gaussian, name in zip(self.object_gaussians, self.splat_object_name_list):
            pose = (tuple(data[f"{name}_position"]), tuple(data[f"{name}_orientation"]))
            cached_pose, moved = object_cache.get(name, (None, None))
            if cached_pose != pose:
                pos = torch.tensor(pose[0], device="cuda").float()
                # PyBullet da (x, y, z, w); el splat usa (w, x, y, z)
                quat = torch.roll(torch.tensor(pose[1], device="cuda").float(), 1)
                moved = transform_object(
                    gaussian, object_config=self.object_config[name], pos=pos, quat=quat,
                    robot_transformation=robot_transformation, object_name=name, robot_name=self.robot_name,
                    transformations_cache=self.transformations_cache)
                object_cache[name] = (pose, moved)
            for key, value in zip(parts, moved):
                parts[key].append(value)

        g = self.robot_gaussian
        with torch.no_grad():
            g._xyz = torch.cat(parts["xyz"], dim=0)
            g._rotation = torch.cat(parts["rot"], dim=0)
            g._opacity = torch.cat(parts["opacity"], dim=0)
            g._features_rest = torch.cat(parts["rest"], dim=0)
            g._features_dc = torch.cat(parts["dc"], dim=0)
            g._scaling = torch.cat(parts["scales"], dim=0)

    def active_camera_names(self) -> list[str]:
        """Cámaras que renderiza get_observations."""
        return self.camera_names

    def get_observations(self, render: bool = True) -> dict:
        """Estado del robot y de los objetos y, con render, las imágenes de active_camera_names()."""
        joint_positions = self.get_joint_state()
        joint_velocities = np.array([p.getJointState(self.dummy_robot, i)[1] for i in range(7)])
        ee_pos, ee_quat = self.get_current_ee_pose()
        ee_euler = p.getEulerFromQuaternion(ee_quat)
        gripper = self.get_current_gripper_state() / 0.8 if self.use_gripper else 0.0
        self.current_gripper_state = gripper

        obs = {
            "joint_positions": joint_positions[:7],
            "all_joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_positions_dummy": joint_positions,
            "ee_pos_quat": ee_quat,
            # pose del último link (no de la punta del gripper) + gripper
            "state": np.concatenate([ee_pos, ee_euler, [gripper]]),
            "action": np.concatenate([ee_pos, ee_euler, [self.current_gripper_action]]),
            "gripper_position": [gripper],   # lo lee GELLO
        }
        for name, body in zip(self.splat_object_name_list, self.urdf_object_list):
            obs[f"{name}_position"], obs[f"{name}_orientation"] = p.getBasePositionAndOrientation(body)

        cameras = self.active_camera_names()
        if render and cameras:
            cached = get_curr_link_states(self.dummy_robot, self.use_link_centers)
            self.prep_image_rendering(data=obs, cached_link_states=cached)
            for name in cameras:
                obs[name] = self.render_image(camera_name=name, cached_link_states=cached)
        for name in ("base_rgb", "wrist_rgb"):
            obs.setdefault(name, None)
        return obs

    # ── fin ──────────────────────────────────────────────────────────────────
    def stop(self) -> None:
        if self._zmq_server_thread is not None and self._zmq_server_thread.is_alive():
            self._zmq_server_thread.join()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001  al cerrar el intérprete puede faltar cualquier cosa
            pass
