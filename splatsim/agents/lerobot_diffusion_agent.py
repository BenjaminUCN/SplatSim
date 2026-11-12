from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict

import numpy as np
import pybullet as p
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame
from torchvision import transforms

from splatsim.image_classifier import SimpleCNN
from splatsim.utils.data.registry import dataset_source_for_checkpoint, hub_repo_id

_N_OBS_STEPS = 2   # temporal window size
_N_ACT_STEPS = 32  # action chunk size


DEFAULT_CHECKPOINT = "~/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial/diffusion_1021_1640"
DEFAULT_DATASET_ROOT = "~/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010"
IK_ORIENTATION_MODES = ("off", "hybrid")
IK_HYBRID_MAX_POS_ERR = 0.01   # m
EE_LINK_INDEX = 6


def wrap_angles(a):
    """Ángulos a [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def checkpoint_relative_actions(checkpoint: str) -> bool:
    """train_config.yaml del modelo dice si se entrenó con acciones relativas al estado
    (train_diffusion.py --relative_actions); sin el archivo, absolutas."""
    import yaml
    path = os.path.join(checkpoint, "train_config.yaml")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return bool((yaml.safe_load(f) or {}).get("relative_actions", False))


class LeRobotDiffusionAgent:
    """
    SplatSim agent wrapping a LeRobot DiffusionPolicy checkpoint.

    Produces action chunks of _N_ACT_STEPS steps and returns them one at a
    time. Re-runs inference only when the queue is empty.

    Args:
        checkpoint: local model folder, or Hugging Face repo id / URL.
        dataset_root: LeRobot dataset the policy was trained on: local folder, or Hugging Face
            repo id / URL (only meta/ is downloaded).
        device: "cuda" or "cpu".
        n_action_steps: acciones que se ejecutan por predicción (None: las del modelo, normalmente 32).
        ik_orientation: "off" solves IK for the predicted position only (what every trial did
            before 2026-11-10: the Euler angles passed to PyBullet were silently ignored);
            "hybrid" also asks for the predicted orientation and falls back to position-only
            when that moves the end effector more than IK_HYBRID_MAX_POS_ERR from the target
            (the xArm5 has 5 DOF, so not every pose is reachable).
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        dataset_root: str | None = None,
        device: str = "cuda",
        n_action_steps: int | None = None,
        ik_orientation: str = "off",
    ):
        """checkpoint: carpeta del modelo o repo del Hub (None -> DEFAULT_CHECKPOINT).
        dataset_root: dataset con que se entrenó, carpeta o repo del Hub; None -> se busca en
        configs/registry.yaml (scripts/xarm5/build_registry.py): su carpeta si existe en esta
        máquina, si no (modelo del Hub) su repo; si el modelo no está, DEFAULT_DATASET_ROOT."""
        if ik_orientation not in IK_ORIENTATION_MODES:
            raise ValueError(f"ik_orientation debe ser uno de {IK_ORIENTATION_MODES}")
        self.ik_orientation = ik_orientation
        self.ik_counts = {"orientation": 0, "fallback": 0}   # hybrid: con orientación / sin ella
        self.device = device
        self._requested_n_action_steps = n_action_steps   # lo aplica _load_diffusion_policy_model
        self._obs_buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=_N_OBS_STEPS)
        self._action_queue: list[np.ndarray] = []

        checkpoint = checkpoint or DEFAULT_CHECKPOINT
        checkpoint = hub_repo_id(checkpoint) or os.path.expanduser(checkpoint)
        if dataset_root is None:
            dataset_root = dataset_source_for_checkpoint(checkpoint) or DEFAULT_DATASET_ROOT
        dataset_repo = hub_repo_id(dataset_root)
        if dataset_repo:
            # sin root, LeRobotDatasetMetadata baja meta/ del Hub a ~/.cache/huggingface/lerobot
            dataset_id, dataset_root = dataset_repo, None
        else:
            dataset_root = os.path.expanduser(dataset_root)
            # con root local, el repo_id solo se usa como nombre
            dataset_id = f"local/{os.path.basename(dataset_root.rstrip('/'))}"
        print(f"[LeRobotDiffusionAgent] Dataset: {dataset_repo or dataset_root}")

        self._load_ik_dummy_robot()
        self._load_diffusion_policy_model(checkpoint, dataset_id, dataset_root)
        # acciones = pose del efector - pose del estado con que se predijo el chunk (pos y euler)
        self.relative_actions = checkpoint_relative_actions(checkpoint)
        self._chunk_reference: np.ndarray | None = None
        print(f"[LeRobotDiffusionAgent] Acciones {'relativas' if self.relative_actions else 'absolutas'}, "
              f"IK {self.ik_orientation}")

        self.make_new_prediction = True
    
    def _load_diffusion_policy_model(self, checkpoint: str, dataset_id: str, dataset_root: str | None): 
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
       
        if hub_repo_id(checkpoint):
            # se baja aquí todo el repo (~1 GB la primera vez, después queda en el caché) para
            # avisar que puede tardar; from_pretrained y los processors lo leen del caché
            from huggingface_hub import snapshot_download, try_to_load_from_cache
            if not isinstance(try_to_load_from_cache(checkpoint, "model.safetensors"), str):
                print(f"[LeRobotDiffusionAgent] Downloading {checkpoint} from the Hub "
                      "(~1 GB, only the first time)...", flush=True)
            snapshot_download(checkpoint)
        print(f"[LeRobotDiffusionAgent] Loading: {checkpoint}", flush=True)
        self.policy = DiffusionPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        n_action_steps = self._requested_n_action_steps
        if n_action_steps is not None:
            # acciones ejecutadas por predicción: el chunk entrenado es de horizon pasos desde el primer
            # paso observado, así que caben hasta horizon - n_obs_steps + 1 (modeling_diffusion.py)
            cfg = self.policy.config
            limit = cfg.horizon - cfg.n_obs_steps + 1
            if not 1 <= n_action_steps <= limit:
                raise ValueError(f"n_action_steps debe estar entre 1 y {limit} (horizon {cfg.horizon}, "
                                 f"n_obs_steps {cfg.n_obs_steps})")
            print(f"[LeRobotDiffusionAgent] n_action_steps {cfg.n_action_steps} -> {n_action_steps}")
            cfg.n_action_steps = n_action_steps
            self.policy.reset()                  # la cola de acciones se crea con maxlen n_action_steps
        self._n_act_steps = self.policy.config.n_action_steps
        
        print("[LeRobotDiffusionAgent] Policy Config:.")
        print(self.policy.config)
        print("[LeRobotDiffusionAgent] Ready.")
                
        print("[LeRobotDiffusionAgent] Loading dataset metadata.")
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
        # features del dataset sin las cámaras que la política no usa: build_inference_frame
        # exige cada imagen que aparezca aquí, así el servidor solo renderiza las necesarias
        self.input_features = {
            k: ft for k, ft in self.dataset_metadata.features.items()
            if not k.startswith("observation.images.") or k in self.policy.config.image_features
        }
        print(f"[LeRobotDiffusionAgent] Cámaras: {list(self.policy.config.image_features)}")

        print("[LeRobotDiffusionAgent] Making pre-post processors.")
        stats = self.dataset_metadata.stats
        # modelo con acciones relativas: la acción se normalizó con las estadísticas de los deltas
        from splatsim.utils.data.relative_actions import load_stats
        relative_stats = load_stats(checkpoint) if os.path.isdir(checkpoint) else None
        if relative_stats is not None:
            stats = {**stats, "action": relative_stats}
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint,
            dataset_stats=stats,
        )

    def _load_ik_dummy_robot(self):
        p.connect(p.DIRECT)
        
        urdf_path = os.path.expanduser("~/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        base_position = [0, 0, 0]

        flags = p.URDF_USE_IMPLICIT_CYLINDER
        self.dummy_robot = p.loadURDF(
            urdf_path, useFixedBase=True, basePosition=base_position, flags=flags
        )
        
        (
        self.joint_indices,
        self.lower_limits,
        self.upper_limits,
        self.joint_ranges,
        self.joint_names,
        ) = self.get_revolute_joint_limits(robot_id=self.dummy_robot)

    def _load_image_classifier(self):
        
        self.classifier_model = SimpleCNN().cuda()
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'image_classifier', 'best_model.pth')
        
        self.classifier_model.load_state_dict(torch.load(checkpoint_path))
        self.classifier_model.eval()

    def _load(self, checkpoint: str):
        """Load DiffusionPolicy from HF Hub or local path."""
        
    
    def safe_to_numpy(self, obj: Any) -> np.ndarray:

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()

        if isinstance(obj, np.ndarray):
            return obj

        if isinstance(obj, (list, tuple)):
            return np.asarray(obj)

        if np.isscalar(obj):
            return np.asarray([obj])

        raise TypeError(f"Unsupported type: {type(obj)}")
    
    def _build_obs(self, obs):
        # joint_positions = obs["joint_positions"]
        # state = self.safe_to_numpy(joint_positions).astype(np.float32).flatten()
        state = obs["state"]
        state = self.safe_to_numpy(state).astype(np.float32).flatten()
        
        # una imagen por cada cámara que usa la política (puede ser un subconjunto de las
        # del dataset); el servidor entrega tensores RGB CxHxW en [0, 1], igual que los PNG
        frame = {}
        for key in self.policy.config.image_features:
            cam = key.removeprefix("observation.images.")
            image = obs[cam]
            if hasattr(image, 'detach'):
                image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)  # como el recorder
            frame[cam] = image
        for name in self.dataset_metadata.features['observation.state']['names']:
            frame[name] = state[self.dataset_metadata.features['observation.state']['names'].index(name)]

        frame = build_inference_frame(
            observation=frame,
            ds_features=self.input_features,
            device=self.device,
        )

        frame = self.preprocess(frame)
        
        return frame


    def get_revolute_joint_limits(self, robot_id):
        """
        Extract revolute joint information from a PyBullet robot.

        Returns:
            joint_indices : list[int]
                PyBullet joint indices for revolute joints.

            lower_limits : list[float]
                Lower joint limits (radians).

            upper_limits : list[float]
                Upper joint limits (radians).

            joint_ranges : list[float]
                Joint motion ranges (upper - lower).

            joint_names : list[str]
                Human-readable joint names.
        """

        joint_indices = []
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        joint_names = []

        num_joints = p.getNumJoints(robot_id)

        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)

            joint_type = info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                joint_name = info[1].decode("utf-8")

                ll = info[8]
                ul = info[9]

                joint_indices.append(i)
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(ul - ll)
                joint_names.append(joint_name)

        return (
            joint_indices,
            lower_limits,
            upper_limits,
            joint_ranges,
            joint_names,
        )
        
    def draw_pose(self, position, quaternion, length=0.1, life_time=0):
        rot_matrix = p.getMatrixFromQuaternion(quaternion)

        x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
        z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        x_end = [position[i] + length * x_axis[i] for i in range(3)]
        y_end = [position[i] + length * y_axis[i] for i in range(3)]
        z_end = [position[i] + length * z_axis[i] for i in range(3)]

        p.addUserDebugLine(position, x_end, [1, 0, 0], 2, lifeTime=life_time)
        p.addUserDebugLine(position, y_end, [0, 1, 0], 2, lifeTime=life_time)
        p.addUserDebugLine(position, z_end, [0, 0, 1], 2, lifeTime=life_time)


    def reset(self) -> None:
        """Al empezar un ensayo: vacía las colas de la política."""
        self.policy.reset()
        self._chunk_reference = None
        self.make_new_prediction = True

    def _new_chunk_next(self) -> bool:
        """select_action va a predecir un chunk nuevo (su cola de acciones está vacía)."""
        queues = getattr(self.policy, "_queues", None)
        return not queues or len(queues.get("action", ())) == 0

    def _ik(self, position, rest, quaternion=None):
        kwargs = {"targetOrientation": quaternion} if quaternion is not None else {}
        return p.calculateInverseKinematics(
            self.dummy_robot, endEffectorLinkIndex=EE_LINK_INDEX, targetPosition=position,
            residualThreshold=0.00001, maxNumIterations=10000, restPoses=rest, **kwargs)

    def _solve_ik(self, position, quaternion, rest):
        """off: solo posición. hybrid: con orientación si el efector queda a menos de
        IK_HYBRID_MAX_POS_ERR del objetivo; si no, solo posición."""
        if self.ik_orientation == "off":
            return self._ik(position, rest)
        solution = self._ik(position, rest, quaternion)
        current = [p.getJointState(self.dummy_robot, i)[0] for i in range(1, 6)]
        for i in range(1, 6):
            p.resetJointState(self.dummy_robot, i, solution[i - 1])
        reached = np.array(p.getLinkState(self.dummy_robot, EE_LINK_INDEX, computeForwardKinematics=True)[4])
        for i in range(1, 6):
            p.resetJointState(self.dummy_robot, i, current[i - 1])
        if np.linalg.norm(reached - np.asarray(position)) <= IK_HYBRID_MAX_POS_ERR:
            self.ik_counts["orientation"] += 1
            return solution
        self.ik_counts["fallback"] += 1
        return self._ik(position, rest)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        obs["base_rgb"]        -> (H, W, 3)
        obs["joint_positions"] -> (6,)
        obs["state"] -> (7,)
        """
        
        obs_dict = obs.copy()
        
        for i in range(1, 6):
            p.resetJointState(self.dummy_robot, i, obs_dict['joint_positions'][i-1])
        
        
        if self.make_new_prediction:
            obs_frame = self._build_obs(obs)  # ya normalizado por self.preprocess

            if self.relative_actions and self._new_chunk_next():
                # el chunk que viene se entrenó relativo al estado del último paso observado
                self._chunk_reference = self.safe_to_numpy(obs["state"]).astype(np.float64).flatten()[:6]
            action = self.policy.select_action(obs_frame)
            action = self.postprocess(action)

            action_np = action.squeeze().cpu().numpy()
            if self.relative_actions:
                action_np = action_np.astype(np.float64)
                action_np[:3] += self._chunk_reference[:3]
                action_np[3:6] = wrap_angles(action_np[3:6] + self._chunk_reference[3:6])
            self.action_prediction = action_np
            print(f"New action prediction: {self.action_prediction}")
            
            self.ee_pose = self.action_prediction[:3]
            self.ee_rot = self.action_prediction[3:6]
            self.ee_gripper = self.action_prediction[6]
                
            self.make_new_prediction = False
        
        ee_pose = self.ee_pose
        ee_rot = self.ee_rot
        
        ee_quat = p.getQuaternionFromEuler(ee_rot)
        print(f"ee_pose: ({ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f})")
        print(f"ee_rot: ({ee_rot[0]:.3f}, {ee_rot[1]:.3f}, {ee_rot[2]:.3f})")
        
        joints_n = 5

        self.draw_pose(ee_pose, ee_quat, life_time=1)

        rest = obs_dict["joint_positions"][:len(self.joint_indices)]
        dummy_joint_pos = self._solve_ik(ee_pose, ee_quat, rest)

        # calculate difference between current and target joint angles
        joint_diff = np.array(dummy_joint_pos)[:joints_n] - np.array(obs_dict['joint_positions'])[:joints_n]
        if np.linalg.norm(joint_diff) < 2.0:
            self.make_new_prediction = True
        else:
            print(f"Joint difference norm: {np.linalg.norm(joint_diff):.3f}")
            print(f"Joint difference: {joint_diff}")

        joints = np.array(dummy_joint_pos)[:joints_n]
        joints = np.append(joints, self.ee_gripper)
        
        return joints
    