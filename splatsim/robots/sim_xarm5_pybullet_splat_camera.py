"""
Servidor PyBullet + Gaussian Splat para el xArm5 Lite.
Hereda de PybulletRobotServerBase (igual que el UR5).
"""
import os
import time
import math
import pickle
import shutil
import threading
import sys
import pybullet as p
import numpy as np  
import torch
import cv2
import argparse
from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
)
from splatsim.utils.robot_splat_render_utils import get_curr_link_states

from gaussian_renderer import render

from torchvision.transforms.functional import to_pil_image
from gaussian_splatting.scene.cameras import Camera


class XArm5PybulletRobotServerCamera(PybulletRobotServerBase):
    
    """
    Servidor interactivo para el xArm5 Lite.
    Muestra el Gaussian Splat del brazo en tiempo real
    mientras el GELLO/operador envía comandos de joints.
    """

    # Configuración mínima del entorno — sin objetos extra
    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "plastic_apple",  # objeto dummy que sí existe
                "splat_object_name": "plastic_apple",
                "grasp_config": [PybulletRobotServerBase.GRASP_CONFIGS["apple"]],
            }
        ]
    }
    # Cada cuántos ticks de serve_loop se guarda un frame.
    # A 240 Hz y RECORD_EVERY=12 → ~20 fps de grabación.
    RECORD_EVERY = 12


    def __init__(self, traj_save_path: str = None, move_apple: bool = False, **kwargs):

        # El xArm5 tiene 5 DOF de brazo + 1 gripper = 6
        self._num_dofs = 6

        self._traj_save_path_override = traj_save_path

        # Estado de grabación (inicializado antes de super().__init__
        # para evitar carreras con serve_loop si se inicia muy rápido)
        self._is_recording: bool = False
        self._record_tick: int = 0          # contador de ticks del serve_loop
        self._trajectory_length: int = 0    # frames grabados en esta trayectoria
        super().__init__(**kwargs)

        self._remove_wall()
        #self._disable_robot_apple_collision()
        self._disable_apple_gravity()
        self._lower_floor(z=-0.2)
        self.setup_base_camera()
        self.setup_wrist_camera()
        self.setup_apple_camera()
        self._start_keyboard_listener()
        self._move_apple = move_apple

        self.reset_joints_btn = self.pybullet_client.addUserDebugParameter("Reset Robot Joints", 1, 0, 0)
        self.prev_reset_joints_btn_val = 0

        if self._move_apple:
            self.prev_slider_x = 0.3
            self.prev_slider_y = 0.2
            self.prev_slider_z = 0.1
            # self.apple_x = self.pybullet_client.addUserDebugParameter("apple_x", -1.0, 1.0, self.prev_slider_x)
            # self.apple_y = self.pybullet_client.addUserDebugParameter("apple_y", -1.0, 1.0, self.prev_slider_y)
            # self.apple_z = self.pybullet_client.addUserDebugParameter("apple_z", -1.0, 1.0, self.prev_slider_z)
            
            self.apple_x = self.pybullet_client.addUserDebugParameter("apple_x", 0.7, 1.0, self.prev_slider_x)
            self.apple_y = self.pybullet_client.addUserDebugParameter("apple_y", -0.4, 0.4, self.prev_slider_y)
            self.apple_z = self.pybullet_client.addUserDebugParameter("apple_z", -0.04, 0.4, self.prev_slider_z)

            self.set_random_apple_position_btn = self.pybullet_client.addUserDebugParameter("Set Random Apple Position", 1, 0, 0)
            self.prev_set_random_apple_position_btn_val = 0
          
        if self.wrist_camera_link_index is not None:
            state = self.pybullet_client.getLinkState(
                self.dummy_robot, 
                self.wrist_camera_link_index,
                computeForwardKinematics=True
            )
            print(f"[xArm5] wrist_camera posición world: {state[0]}")
            print(f"[xArm5] wrist_camera orientación world: {state[1]}")

    # ── Gestión del teclado ──────────────────────────────────────────────────

    def _start_keyboard_listener(self) -> None:
        """
        Lanza un hilo daemon que lee líneas de stdin.
        Escribe 'r' + Enter para iniciar, 's' + Enter para detener.
        No requiere dependencias externas.
        """
        def _listen():
            print("[Recording] Listo — escribe 'r' + Enter para grabar, 's' + Enter para detener.")
            for line in sys.stdin:
                ch = line.strip().lower()
                if ch == "r":
                    self._start_recording()
                elif ch == "s":
                    self._stop_recording()

        t = threading.Thread(target=_listen, daemon=True)
        t.start()

    # ── Lógica de grabación ──────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self._is_recording:
            print("[Recording] Ya se está grabando. Pulsa 's' primero para detener.")
            return

        # Contar trayectorias existentes para elegir el siguiente índice
        self.trajectory_count = len(os.listdir(self.path))
        traj_folder = os.path.join(
            self.path, str(self.trajectory_count).zfill(3)
        )
        os.makedirs(traj_folder, exist_ok=True)

        self._trajectory_length = 0
        self._record_tick = 0
        self._is_recording = True
        print(
            f"[Recording] ▶ Grabando en {traj_folder}  "
            f"(cada {self.RECORD_EVERY} ticks ≈ {240 // self.RECORD_EVERY} fps)"
        )

    def _stop_recording(self) -> None:
        if not self._is_recording:
            print("[Recording] No hay grabación activa.")
            return

        self._is_recording = False
        traj_folder = os.path.join(
            self.path, str(self.trajectory_count).zfill(3)
        )
        if self._trajectory_length == 0:
            # No se grabó ningún frame — borrar carpeta vacía
            shutil.rmtree(traj_folder, ignore_errors=True)
            print("[Recording] ■ Detenido — sin frames, carpeta eliminada.")
        else:
            # Guardar un último frame antes de cerrar
            self._save_current_frame()
            print(
                f"[Recording] ■ Guardado — {self._trajectory_length} frames "
                f"en {traj_folder}"
            )

    def _save_current_frame(self) -> None:
        """Captura observaciones y las serializa en un .pkl numerado."""
        # /0 (trajectory folder)
        #  - images_1/ (image folder)
        #      - (image_name)_(image_index).png
        #  - 00001.pkl (trajectory_length)
        #  - 00002.pkl
        
        self._trajectory_length += 1
        obs = self.get_observations()
        trajectory_folder = os.path.join(self.path, str(self.trajectory_count).zfill(3))
        frame_path = os.path.join(
            # self.path,
            # str(self.trajectory_count).zfill(3),
            trajectory_folder,
            str(self._trajectory_length).zfill(5) + ".pkl",
        )

        image_folder = os.path.join(trajectory_folder, "image_1")
        os.makedirs(image_folder, exist_ok=True)
        for image_name in [image_name for image_name in obs.keys() if image_name.endswith("_rgb") and obs[image_name] is not None]:
                frame = obs[image_name]
                frame = np.transpose(frame.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
                frame = (frame * 255).astype(np.uint8)

                image_index = len(os.listdir(image_folder)) + 1
                image_path = os.path.join(image_folder, f"{image_name}_{image_index:05d}.png")

                cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with open(frame_path, "wb") as f:
            pickle.dump(obs, f)


    def setup_apple_camera(self):
        """Guarda el índice del body de la manzana para consultarlo cada frame."""
        apple_idx = self.splat_object_name_list.index("plastic_apple")
        self.apple_body_id = self.urdf_object_list[apple_idx]
        print(f"[xArm5] apple body_id: {self.apple_body_id}")

    def get_apple_camera(self):
        """
        Cámara fija relativa a la manzana.
        Ajusta offset_xyz y offset_rpy para posicionarla donde quieras.
        """
        uid = 0
        colmap_id = 2  # distinto al de wrist/base

        # --- Offset de la cámara respecto al centro de la manzana ---
        offset_xyz = np.array([0.0, 0.0, 0.0])   # 15 cm arriba del centro
        offset_rpy = [0.0, 0.0, 0.0]            # mirando hacia abajo

        # Pose de la manzana en world frame PyBullet
        pos_apple, quat_apple = self.pybullet_client.getBasePositionAndOrientation(
            self.apple_body_id
        )

        R_apple = np.array(
            self.pybullet_client.getMatrixFromQuaternion(quat_apple)
        ).reshape(3, 3)

        # Rotación del offset
        quat_offset = self.pybullet_client.getQuaternionFromEuler(offset_rpy)
        R_offset = np.array(
            self.pybullet_client.getMatrixFromQuaternion(quat_offset)
        ).reshape(3, 3)

        # Pose final de la cámara en world frame
        R_cam_world = R_apple @ R_offset
        T_cam_world = np.array(pos_apple) + R_apple @ offset_xyz

        # Pasar al espacio del splat
        robot_transformation_inv = (
            self.transformations_cache[self.robot_name]["inv_transformation"]
            .cpu().numpy()
        )

        Trans_cam_world = np.eye(4)
        Trans_cam_world[:3, :3] = R_cam_world
        Trans_cam_world[:3, 3]  = T_cam_world

        Trans_cam_splat = robot_transformation_inv @ Trans_cam_world

        R_cw = torch.tensor(Trans_cam_splat[:3, :3], dtype=torch.float32)
        T_cw = torch.tensor(Trans_cam_splat[:3, 3],  dtype=torch.float32)

        scale = torch.pow(torch.linalg.det(R_cw), 1 / 3)
        R_cw = R_cw / scale
        T_cw = T_cw / scale

        Trans_wo_scale = torch.eye(4)
        Trans_wo_scale[:3, :3] = R_cw
        Trans_wo_scale[:3, 3]  = T_cw

        T_wc = torch.linalg.inv(Trans_wo_scale)[:3, 3]

        FoVx = 1.375955594372348
        FoVy = 1.1025297299614814
        image_width  = 1280
        image_height = 720
        image = torch.zeros((3, image_height, image_width)).float()

        resolution = (image_width, image_height)
        camera = Camera(
            resolution,
            colmap_id,
            R_cw.detach().cpu().numpy(),
            T_wc.detach().cpu().numpy(),
            FoVx,
            FoVy,
            None,
            to_pil_image(image),
            None,
            "apple_rgb",
            uid,
            scale=scale.detach().cpu().numpy(),
        )
        return camera

    def _lower_floor(self, z: float = -0.2) -> None:
        if hasattr(self, "plane") and self.plane is not None:
            try:
                self.pybullet_client.resetBasePositionAndOrientation(
                    self.plane,
                    [0, 0, z],
                    self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
                )
            except Exception:
                pass
    def update_apple_pose_from_ui(self):
        if not self._move_apple:
            return
        
        slider_x = self.pybullet_client.readUserDebugParameter(self.apple_x)
        slider_y = self.pybullet_client.readUserDebugParameter(self.apple_y)
        slider_z = self.pybullet_client.readUserDebugParameter(self.apple_z)

        slider_moved = (
            abs(slider_x - self.prev_slider_x) > 1e-6 or
            abs(slider_y - self.prev_slider_y) > 1e-6 or
            abs(slider_z - self.prev_slider_z) > 1e-6
        )

        if slider_moved:
            x, y, z = slider_x, slider_y, slider_z
            self.prev_slider_x = slider_x
            self.prev_slider_y = slider_y
            self.prev_slider_z = slider_z
            
            self.set_object_pose(
                "plastic_apple",
                [x, y, z],
                [0, 0, 0, 1],
                use_gravity=False
            )

        current_val = self.pybullet_client.readUserDebugParameter(self.set_random_apple_position_btn)
        if current_val != self.prev_set_random_apple_position_btn_val:
            self.prev_set_random_apple_position_btn_val = current_val
            print("[xArm5] Setting random apple position.")
            x = np.random.uniform(0.768, 0.85)
            y = np.random.uniform(-0.34, 0.36)
            z = np.random.uniform(-0.02, 0.337)

            self.set_object_pose(
                "plastic_apple",
                [x, y, z],
                [0, 0, 0, 1],
                use_gravity=False
            )

    def _disable_apple_gravity(self):

        apple_idx = self.splat_object_name_list.index("plastic_apple")
        apple_id = self.urdf_object_list[apple_idx]

        # Hacer la apple estática (sin gravedad)
        self.pybullet_client.changeDynamics(apple_id, -1, mass=0)

        print("[xArm5] gravedad desactivada para la apple")

    
    def setup_wrist_camera(self):
        num_joints = self.pybullet_client.getNumJoints(self.dummy_robot)
        self.wrist_camera_link_index = None

        for i in range(num_joints):
            info = self.pybullet_client.getJointInfo(self.dummy_robot, i)
            joint_name = info[1].decode("utf-8")   # nombre del joint
            link_name  = info[12].decode("utf-8")  # nombre del link hijo

            if link_name == "wrist_camera_link":
                self.wrist_camera_link_index = i
                print(f"[xArm5] wrist_camera_link encontrado: joint='{joint_name}', índice={i}")
                break

        if self.wrist_camera_link_index is None:
            print("[xArm5] WARNING: wrist_camera_link no encontrado en el URDF")

    def _remove_wall(self):
        if hasattr(self, 'wall') and self.wall is not None:
            try:
                self.pybullet_client.removeBody(self.wall)
                self.wall = None
  
            except Exception:
                pass

    def _disable_robot_apple_collision(self):
        apple_idx = self.splat_object_name_list.index("plastic_apple")
        apple_id = self.urdf_object_list[apple_idx]

        num_links = self.pybullet_client.getNumJoints(self.dummy_robot)

        for link_idx in range(-1, num_links):
            self.pybullet_client.setCollisionFilterPair(
                self.dummy_robot,
                apple_id,
                link_idx,
                -1,
                enableCollision=0
            )

    def setup_gripper(self):
        """Override: el xArm5 usa 'drive_joint' como joint padre del gripper."""
        self.__parse_joint_info__()
        self.gripper_range = [0, 0.085]

        # Nombres reales en el URDF del xArm5 lite
        mimic_parent_name = "drive_joint"
        mimic_children_names = {
            "left_finger_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_outer_knuckle_joint": 1,
            "right_finger_joint": 1,
            "right_inner_knuckle_joint": 1,
        }

        joints_0 = [joint.id for joint in self.joints if joint.name == mimic_parent_name]

        if len(joints_0) == 0:
            raise ValueError(
                f"No se encontró '{mimic_parent_name}' en los joints. "
                f"Joints disponibles: {[j.name for j in self.joints]}"
            )

        self.mimic_parent_id = joints_0[0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.pybullet_client.createConstraint(
                self.dummy_robot,
                self.mimic_parent_id,
                self.dummy_robot,
                joint_id,
                jointType=self.pybullet_client.JOINT_GEAR,
                jointAxis=[1, 0, 0],#jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            self.pybullet_client.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )

    def move_gripper(self, open_length, velocity=2):
        if not self.use_gripper:
            return
        # El xArm5 usa un rango de 0 a 0.85 en drive_joint
        # Mapear open_length [0, 0.085] → ángulo [0, 0.85]
        open_angle = np.clip(open_length * 10, 0, 0.85)
        p.setJointMotorControl2(
            self.dummy_robot,
            self.mimic_parent_id,
            self.pybullet_client.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=velocity,
        )

    def get_joint_state(self) -> np.ndarray:
        """Retorna 6 DOF: 5 joints del brazo + estado del gripper."""
        joint_states = []
        
        # Solo los 5 joints del brazo (índices 1-5 en PyBullet)
        for i in range(1, 6):
            joint_states.append(
                self.pybullet_client.getJointState(self.dummy_robot, i)[0]
            )
        
        # Estado del gripper: normalizado a [0, 1]
        if self.use_gripper:
            gripper_angle = self.pybullet_client.getJointState(
                self.dummy_robot, self.mimic_parent_id
            )[0]
            # drive_joint va de 0 a 0.85 → normalizar a [0, 1]
            gripper_normalized = np.clip(gripper_angle / 0.85, 0.0, 1.0) #1.0 - np.clip(gripper_angle / 0.85, 0.0, 1.0)
            joint_states.append(gripper_normalized)
        else:
            joint_states.append(0.0)
    
        return np.array(joint_states)  # shape (6,)
        
    def num_dofs(self) -> int:
        return 6
    
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs()
        
        for i in range(1, 6):
            self.pybullet_client.setJointMotorControl2(
                self.dummy_robot,
                i,
                self.pybullet_client.POSITION_CONTROL,
                targetPosition=joint_state[i - 1],
                force=500,
            )
        if self.use_gripper:
            self.move_gripper((joint_state[-1]) * 0.085) #self.move_gripper((1 - joint_state[-1]) * 0.085)
            self.current_gripper_action = joint_state[-1]

        # Avanzar simulación AQUÍ para que los links se muevan
        # antes de que get_observations() capture el estado
        for _ in range(20):
            self.pybullet_client.stepSimulation()

    def render_image(self, camera_name, cached_link_states=None):
        # TODO to save compute, you only need to create the splat once, then it can be rendered w/ different cameras
        if camera_name == "base_rgb":
            camera = self.get_base_camera()
        elif camera_name == "wrist_rgb":
            camera = self.get_wrist_camera()
            if camera is None:
                return None
        elif camera_name == "apple_rgb":        # ← nuevo
            camera = self.get_apple_camera()
        else:
            raise ValueError(f"Unknown camera name {camera_name}")

        rendering = render(camera, self.robot_gaussian, self.pipeline, self.background)[
            "render"
        ]

        # save the image
        return rendering

    def get_observations(self):
        #from splatsim.utils.robot_splat_render_utils import get_curr_link_states
        #curr_states = get_curr_link_states(self.dummy_robot, self.use_link_centers)
        # observa si existe movimiento del brazo
        #print(f"[DEBUG] link5 inicial: {self.initial_link_states[5]}")
        #print(f"[DEBUG] link5 actual:  {curr_states[5]}")
        
        obs = super().get_observations()
        obs["joint_positions"] = self.get_joint_state()

        self._last_apple_rgb = obs.get("apple_rgb")

        return obs

    def serve_loop(self):
        if not hasattr(self, '_splat_gripper_patched'):
            self._patch_gripper_for_splat()
            self._splat_gripper_patched = True

        self.update_apple_pose_from_ui()

        self.update_base_camera()
        
        self.check_reset_joints()

        # Guardar frame si estamos grabando y toca este tick
        if self._is_recording:
            if self._record_tick % self.RECORD_EVERY == 0:
                self._save_current_frame()
            self._record_tick += 1

        self.pybullet_client.stepSimulation()
        time.sleep(1 / 240)
        
        # Mostrar última apple_rgb ya renderizada por get_observations
        if hasattr(self, '_last_apple_rgb') and self._last_apple_rgb is not None:
            img = (self._last_apple_rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imshow("Apple Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def check_reset_joints(self):
        current_val = self.pybullet_client.readUserDebugParameter(self.reset_joints_btn)
        if current_val != self.prev_reset_joints_btn_val:
            print("[xArm5] Resetting robot joints to initial positions.")
            # Reset joint states
            # for i in range(1, len(self.initial_joint_state)):
            #     self.pybullet_client.resetJointState(
            #         self.dummy_robot,
            #         i,
            #         self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
            #     )
            # Reset needs to be done like a command_joint_state 
            for i in range(1, self.num_dofs()):
                self.pybullet_client.setJointMotorControl2(
                    self.dummy_robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
                    force=1000,
                    # force=250,
                )
            self.prev_reset_joints_btn_val = current_val
            print(f"[xArm5] reset_joints_btn value changed to {current_val}")
    
    def _patch_gripper_for_splat(self):
        """
        initial_link_states fue capturado con gripper cerrado (0),
        pero el splat fue capturado con gripper abierto (0.85).
        Solo parchamos los link states del gripper.
        """
        from splatsim.utils.robot_splat_render_utils import get_curr_link_states
        
        # Guardar estado actual del brazo
        arm_states = []
        for i in range(1, 6):
            arm_states.append(
                self.pybullet_client.getJointState(self.dummy_robot, i)[0]
            )
        
        # Mover gripper a pose del splat
        SPLAT_GRIPPER_POSE = 0.5

        self.pybullet_client.resetJointState(
            self.dummy_robot, self.mimic_parent_id, SPLAT_GRIPPER_POSE
        )
        for joint_id in self.mimic_child_multiplier:
            self.pybullet_client.resetJointState(
                self.dummy_robot, joint_id, SPLAT_GRIPPER_POSE
            )
        
        for _ in range(50):
            self.pybullet_client.stepSimulation()
        
        # Leer todos los link states con gripper abierto
        new_link_states = get_curr_link_states(self.dummy_robot, self.use_link_centers)
        
        # Solo parchamos los links del gripper (joints 8-13), no los del brazo
        gripper_joint_ids = [self.mimic_parent_id] + list(self.mimic_child_multiplier.keys())
        for joint_id in gripper_joint_ids:
            self.initial_link_states[joint_id] = new_link_states[joint_id]
        
        #print(f"[PATCH] Gripper links {gripper_joint_ids} actualizados en initial_link_states")
        
        # Restaurar brazo a su posición actual
        for i, val in enumerate(arm_states):
            self.pybullet_client.resetJointState(self.dummy_robot, i + 1, val)
        
        for _ in range(50):
            self.pybullet_client.stepSimulation()

    def get_wrist_camera(self):
        if self.wrist_camera_link_index is None:
            print("WARNING: No wrist camera index found")
            return None

        uid = 0
        colmap_id = 1

        # Obtener pose del link de la cámara en world frame
        link_state = self.pybullet_client.getLinkState(
            self.dummy_robot,
            self.wrist_camera_link_index,
            computeForwardKinematics=True,
        )
        # print("-----------------------------------------")
        # print(f"link_state[0]: {link_state[0]}")
        # print(f"link_state[1]: {link_state[1]}")
        # print(f"link_state[4]: {link_state[4]}")
        # print(f"link_state[5]: {link_state[5]}")

        robot_transformation = self.transformations_cache[self.robot_name]["transformation"]
        robot_transformation_inv = self.transformations_cache[self.robot_name]["inv_transformation"]

        # Posición y orientación en world frame de PyBullet
        T = torch.tensor(link_state[4], device=robot_transformation.device).float()
        quat = link_state[5]  # (x, y, z, w) de PyBullet

        R = (
            torch.tensor(
                self.pybullet_client.getMatrixFromQuaternion(quat),
                device=robot_transformation.device,
            )
            .reshape(3, 3)
            .float()
        )

        # Construir matriz de transformación cámara→world en espacio PyBullet
        Trans_cam_world = torch.eye(4, device=R.device)
        Trans_cam_world[:3, :3] = R
        Trans_cam_world[:3, 3] = T

        # Pasar al espacio del splat
        Trans_cam_splat = torch.matmul(robot_transformation_inv, Trans_cam_world)

        # --- CORRECCIÓN DE ORIENTACIÓN ---
        # La cámara física apunta hacia adelante del gripper.
        # Aplicamos rotación correctiva según lo observado en las imágenes.
        # Rotar 180° en X para invertir la dirección de visión si está al revés:
        #angle = -math.pi / 2  # prueba también -math.pi/2 si gira al lado equivocado
        #c, s = math.cos(angle), math.sin(angle)

        R_fix = torch.tensor([
            [1,  0,  0,  0],
            [0,  1,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1],
        ], device=R.device, dtype=torch.float32)
        Trans_cam_splat = torch.matmul(Trans_cam_splat, R_fix)

        # Parámetros intrínsecos — ajusta según tu cámara real
        #FoVx = 1.375955594372348
        #FoVy = 1.1025297299614814
        #image_width = 640
        #image_height = 480
        FoVx = 1.2112585 #69.4°
        FoVy = 0.7417649 #42.5°
        image_width = 640 #1920
        image_height = 480 #1080
        image_name = "wrist_rgb"
        image = torch.zeros((3, image_height, image_width)).float()

        # Extraer R y T sin escala
        R_cw = Trans_cam_splat[:3, :3]
        T_cw = Trans_cam_splat[:3, 3]
        scale = torch.pow(torch.linalg.det(R_cw), 1 / 3)
        R_cw = R_cw / scale
        T_cw = T_cw / scale

        Trans_cam_splat_wo_scale = torch.eye(4, device=R_cw.device)
        Trans_cam_splat_wo_scale[:3, :3] = R_cw
        Trans_cam_splat_wo_scale[:3, 3] = T_cw

        # World-to-camera
        Rt_wc = torch.linalg.inv(Trans_cam_splat_wo_scale)
        T_wc = Rt_wc[:3, 3]

        # from torchvision.transforms.functional import to_pil_image
        # from gaussian_splatting.scene.cameras import Camera

        resolution = (image_width, image_height)
        camera = Camera(
            resolution,
            colmap_id,
            R_cw.detach().cpu().numpy(),
            T_wc.detach().cpu().numpy(),
            FoVx,
            FoVy,
            None,
            to_pil_image(image),
            None,
            image_name,
            uid,
            scale=scale.detach().cpu().numpy(),
        )
        return camera
    
    def setup_base_camera(self):
        init_x, init_y, init_z   = 0.205,  0.0, 0.158
        init_r, init_pi, init_yaw =  -1.562, 0.0, -1.558

        self.base_camera_position =  (init_x, init_y, init_z)
        self.base_camera_rotation =  (0, 0, 0, 1)

        # --- Create sliders ---
        RANGE_XYZ = 1.5   # ± meters around init
        RANGE_RPY = 3.15  # ± radians

        self.sx   = self.pybullet_client.addUserDebugParameter("X",   init_x  - RANGE_XYZ, init_x  + RANGE_XYZ, init_x)
        self.sy   = self.pybullet_client.addUserDebugParameter("Y",   init_y  - RANGE_XYZ, init_y  + RANGE_XYZ, init_y)
        self.sz   = self.pybullet_client.addUserDebugParameter("Z",   init_z  - RANGE_XYZ, init_z  + RANGE_XYZ, init_z)
        self.sr   = self.pybullet_client.addUserDebugParameter("Roll",  init_r   - RANGE_RPY, init_r   + RANGE_RPY, init_r)
        self.sp   = self.pybullet_client.addUserDebugParameter("Pitch", init_pi  - RANGE_RPY, init_pi  + RANGE_RPY, init_pi)
        self.syaw = self.pybullet_client.addUserDebugParameter("Yaw",   init_yaw - RANGE_RPY, init_yaw + RANGE_RPY, init_yaw)

        # Print button to dump current values
        self.print_btn = self.pybullet_client.addUserDebugParameter(">> Print Current Values <<", 1, 0, 0)
        self.prev_btn_val = 0

        # --- Debug axes visualizer ---
        # Draws XYZ axes on the camera link so you can see orientation live
        self.line_ids = [-1, -1, -1]

        print("Sliders ready. Adjust XYZ / RPY to tune wrist_camera_vis_joint origin.")
        print(f"Initial: xyz=({init_x}, {init_y}, {init_z})  rpy=({init_r}, {init_pi}, {init_yaw})")

    def update_base_camera(self):
        # Read sliders
        x   = self.pybullet_client.readUserDebugParameter(self.sx)
        y   = self.pybullet_client.readUserDebugParameter(self.sy)
        z   = self.pybullet_client.readUserDebugParameter(self.sz)
        roll   = self.pybullet_client.readUserDebugParameter(self.sr)
        pitch = self.pybullet_client.readUserDebugParameter(self.sp)
        yaw = self.pybullet_client.readUserDebugParameter(self.syaw)

        def euler_to_matrix(roll, pitch, yaw, device="cpu"):
            roll = torch.tensor(roll, device=device)
            pitch = torch.tensor(pitch, device=device)
            yaw = torch.tensor(yaw, device=device)


            cr = torch.cos(roll)
            sr = torch.sin(roll)
            cp = torch.cos(pitch)
            sp = torch.sin(pitch)
            cy = torch.cos(yaw)
            sy = torch.sin(yaw)

            R = torch.tensor([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr]
            ], device=device)

            return R.float()

        # Build transform: eef -> camera
        self.base_camera_position = (x, y, z)
        self.base_camera_rotation = self.pybullet_client.getQuaternionFromEuler([roll, pitch, yaw])
        #self.base_camera_rotation = euler_to_matrix(roll, pitch, yaw)

        #print("------------------------")
        #print(f"base cam pos: {self.base_camera_position}")
        #print(f"base cam rot: {self.base_camera_rotation}")

        # Print button handler
        btn_val = self.pybullet_client.readUserDebugParameter(self.print_btn)
        if btn_val != self.prev_btn_val:
            self.prev_btn_val = btn_val
            #print("\n--- Copy this into your URDF ---")
            #print(f'<origin xyz="{x:.4f} {y:.4f} {z:.4f}" '
                #f'rpy="{roll:.4f} {pitch:.4f} {yaw:.4f}"/>')
            #print("--------------------------------\n")

    def get_base_camera(self):
        uid = 0
        colmap_id = 1

        robot_transformation = self.transformations_cache[self.robot_name]["transformation"]
        robot_transformation_inv = self.transformations_cache[self.robot_name]["inv_transformation"]

        #self.base_camera_position =  (-0.6823421952936142, -0.058331743496807656, 0.10904565348251515)
        #self.base_camera_rotation =  (-0.12026315918250455, 0.7541479483997733, -0.009873650746112187, 0.6455231641767785)

        # Posición y orientación en world frame de PyBullet
        T = torch.tensor(self.base_camera_position, device=robot_transformation.device).float()
        quat = self.base_camera_rotation  # (x, y, z, w) de PyBullet

        

        R = (
            torch.tensor(
                self.pybullet_client.getMatrixFromQuaternion(quat),
                device=robot_transformation.device,
            )
            .reshape(3, 3)
            .float()
        )

        # R = self.base_camera_rotation

        # Construir matriz de transformación cámara→world en espacio PyBullet
        Trans_cam_world = torch.eye(4, device=R.device)
        Trans_cam_world[:3, :3] = R
        Trans_cam_world[:3, 3] = T

        # Pasar al espacio del splat
        Trans_cam_splat = torch.matmul(robot_transformation_inv, Trans_cam_world)

        # Parámetros intrínsecos — ajusta según tu cámara real
        #FoVx = 1.375955594372348
        #FoVy = 1.1025297299614814
        #image_width = 640
        #image_height = 480
        FoVx = 1.2112585 #69.4°
        FoVy = 0.7417649 #42.5°
        image_width = 640 #1920
        image_height = 480 #1080
        image_name = "base_rgb"
        image = torch.zeros((3, image_height, image_width)).float()

        # Extraer R y T sin escala
        R_cw = Trans_cam_splat[:3, :3]
        T_cw = Trans_cam_splat[:3, 3]
        scale = torch.pow(torch.linalg.det(R_cw), 1 / 3)
        R_cw = R_cw / scale
        T_cw = T_cw / scale

        Trans_cam_splat_wo_scale = torch.eye(4, device=R_cw.device)
        Trans_cam_splat_wo_scale[:3, :3] = R_cw
        Trans_cam_splat_wo_scale[:3, 3] = T_cw

        # World-to-camera
        Rt_wc = torch.linalg.inv(Trans_cam_splat_wo_scale)
        T_wc = Rt_wc[:3, 3]

        resolution = (image_width, image_height)
        camera = Camera(
            resolution,
            colmap_id,
            R_cw.detach().cpu().numpy(),
            T_wc.detach().cpu().numpy(),
            FoVx,
            FoVy,
            None,
            to_pil_image(image),
            None,
            image_name,
            uid,
            scale=scale.detach().cpu().numpy(),
        )
        return camera

    def _debug_camera_direction(self):
        """Imprime hacia dónde apunta la cámara wrist en world frame."""
        link_state = self.pybullet_client.getLinkState(
            self.dummy_robot,
            self.wrist_camera_link_index,
            computeForwardKinematics=True,
        )
        quat = link_state[1]
        R = np.array(
            self.pybullet_client.getMatrixFromQuaternion(quat)
        ).reshape(3, 3)
        
        # En convención cámara, el eje Z apunta hacia donde mira
        forward = R[:, 2]  # tercera columna = eje Z de la cámara
        up      = R[:, 1]  # segunda columna = eje Y (arriba)
        
        print(f"[CAM] forward (hacia donde mira): {forward}")
        print(f"[CAM] up (arriba de la cámara):   {up}")

    def _debug_wrist_camera_full(self):
        """
        Imprime toda la info necesaria para calibrar la cámara wrist.
        Llamar justo antes del while True en serve().
        """
        import numpy as np

        if self.wrist_camera_link_index is None:
            print("[DEBUG CAM] wrist_camera_link_index es None!")
            return

        # Estado del link en world frame de PyBullet
        link_state = self.pybullet_client.getLinkState(
            self.dummy_robot,
            self.wrist_camera_link_index,
            computeForwardKinematics=True,
        )
        pos_world  = link_state[0]   # posición en world frame
        quat_world = link_state[1]   # orientación en world frame (x,y,z,w)

        R = np.array(
            self.pybullet_client.getMatrixFromQuaternion(quat_world)
        ).reshape(3, 3)

        # Columnas de la matriz de rotación = ejes de la cámara en world frame
        right   = R[:, 0]   # eje X cámara
        up      = R[:, 1]   # eje Y cámara  
        forward = R[:, 2]   # eje Z cámara = dirección de visión

        #print("=" * 60)
        #print(f"[DEBUG CAM] link_index      : {self.wrist_camera_link_index}")
        #print(f"[DEBUG CAM] pos world       : {np.round(pos_world, 4)}")
        #print(f"[DEBUG CAM] quat world      : {np.round(quat_world, 4)}")
        #print(f"[DEBUG CAM] eje X (right)   : {np.round(right, 3)}")
        #print(f"[DEBUG CAM] eje Y (up)      : {np.round(up, 3)}")
        #print(f"[DEBUG CAM] eje Z (forward) : {np.round(forward, 3)}")
        #print("=" * 60)

        # También imprime el link_eef para comparar
        # Buscar índice de joint_eef
        for i in range(self.pybullet_client.getNumJoints(self.dummy_robot)):
            info = self.pybullet_client.getJointInfo(self.dummy_robot, i)
            if info[12].decode("utf-8") == "link_eef":
                eef_state = self.pybullet_client.getLinkState(
                    self.dummy_robot, i, computeForwardKinematics=True
                )
                #print(f"[DEBUG EEF] pos world link_eef: {np.round(eef_state[0], 4)}")
                #print(f"[DEBUG EEF] quat world link_eef: {np.round(eef_state[1], 4)}")
                break
        
    def get_joint_state_dummy(self) -> np.ndarray:
     return self.get_joint_state()
    
    def serve(self) -> None:
        """Override para añadir debug de cámara antes del loop principal."""
        # Preparar para teleport removiendo fuerzas
        for i in range(len(self.initial_joint_state)):
            self.pybullet_client.setJointMotorControl2(
                self.dummy_robot, i, self.pybullet_client.VELOCITY_CONTROL, force=0
            )
        # Reset joint states
        for i in range(1, len(self.initial_joint_state)):
            self.pybullet_client.resetJointState(
                self.dummy_robot,
                i,
                self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
            )
        self.initial_link_states = get_curr_link_states(
            self.dummy_robot, self.use_link_centers
        )

        ee_pos, ee_quat = self.get_current_ee_pose()
        self.iniital_ee_quat = ee_quat

        for i in range(1, self.num_dofs()):
            self.pybullet_client.setJointMotorControl2(
                self.dummy_robot, i, p.VELOCITY_CONTROL,
                targetPosition=self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
                force=250, maxVelocity=0.2,
            )
        self.close_gripper()
        self.initial_ee_pos, self.initial_ee_quat = self.get_current_ee_pose()

        for i in range(10000):
            for j in range(1, len(self.initial_joint_state)):
                self.pybullet_client.resetJointState(
                    self.dummy_robot, j,
                    self.initial_joint_state[j - 1] * self.joint_signs[j - 1],
                )
            self.pybullet_client.stepSimulation()

        self._zmq_server_thread.start()
        print("Ready to serve.")

        # ══════════ DEBUG AQUÍ ══════════
        self._debug_wrist_camera_full()
        # ════════════════════════════════

        while True:
            self.serve_loop()
