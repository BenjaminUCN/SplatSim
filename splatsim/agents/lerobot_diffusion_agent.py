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
from splatsim.utils.registry import dataset_for_checkpoint

# Constants from config.json
_IMAGE_KEY   = "observation.images.base_rgb"
_SIM_OBS_KEY = "base_rgb"
_IMAGE_H     = 480
_IMAGE_W     = 640
_STATE_DIM   = 6   # joints only, no gripper in state
_N_OBS_STEPS = 2   # temporal window size
_N_ACT_STEPS = 32  # action chunk size


DEFAULT_CHECKPOINT = "/home/magister/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial/diffusion_1021_1640"
DEFAULT_DATASET_ROOT = "/home/magister/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010"


class LeRobotDiffusionAgent:
    """
    SplatSim agent wrapping a LeRobot DiffusionPolicy checkpoint.

    Produces action chunks of _N_ACT_STEPS steps and returns them one at a
    time. Re-runs inference only when the queue is empty.

    Args:
        checkpoint: HuggingFace repo id or local pretrained_model folder.
        dataset_root: local LeRobot dataset the policy was trained on.
        device: "cuda" or "cpu".
        n_action_steps: overrides the default chunk size of 32.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        dataset_root: str | None = None,
        device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        """checkpoint: carpeta del modelo (None -> DEFAULT_CHECKPOINT).
        dataset_root: dataset con que se entrenó; None -> se busca en configs/registry.yaml
        (scripts/build_registry.py) y, si el modelo no está, DEFAULT_DATASET_ROOT."""
        self.device = device
        self._n_act_steps = n_action_steps or _N_ACT_STEPS
        self._obs_buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=_N_OBS_STEPS)
        self._action_queue: list[np.ndarray] = []

        checkpoint = os.path.expanduser(checkpoint or DEFAULT_CHECKPOINT)
        if dataset_root is None:
            dataset = dataset_for_checkpoint(checkpoint)
            dataset_root = dataset["path"] if dataset else DEFAULT_DATASET_ROOT
        dataset_root = os.path.expanduser(dataset_root)
        # con root local, el repo_id solo se usa como nombre
        dataset_id = f"local/{os.path.basename(dataset_root.rstrip('/'))}"
        print(f"[LeRobotDiffusionAgent] Dataset: {dataset_root}")

        self._load_ik_dummy_robot()
        self._load_diffusion_policy_model(checkpoint, dataset_id, dataset_root)
        
        self.make_new_prediction = True
    
    def _load_diffusion_policy_model(self, checkpoint: str, dataset_id: str, dataset_root: str): 
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
       
        print(f"[LeRobotDiffusionAgent] Loading: {checkpoint}")
        self.policy = DiffusionPolicy.from_pretrained(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        
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
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint,
            dataset_stats=self.dataset_metadata.stats,
        )  

    def _load_ik_dummy_robot(self):
        p.connect(p.DIRECT)
        
        urdf_path = "/home/magister/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5.urdf"
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
        
        checkpoint_path='/home/magister/fork-splatsim/image_classifier/best_model.pth'
        
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

            action = self.policy.select_action(obs_frame)
            action = self.postprocess(action)
            
            action_np = action.squeeze().cpu().numpy()
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
        ee_link_index = 6
        
        self.draw_pose(ee_pose, ee_quat, life_time=1)
        
        dummy_joint_pos = p.calculateInverseKinematics(
            self.dummy_robot,
            endEffectorLinkIndex=ee_link_index,
            targetPosition=ee_pose,
            targetOrientation=ee_rot,
            residualThreshold=0.00001,
            maxNumIterations=10000,
             
            # lowerLimits=lower_limits,
            # upperLimits=upper_limits,
            # jointRanges=joint_ranges,
            restPoses = obs_dict["joint_positions"][:len(self.joint_indices)]
        )
        
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
    