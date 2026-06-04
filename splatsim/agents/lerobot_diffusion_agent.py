from __future__ import annotations

from collections import deque
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch

import pybullet as p

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

# Constants from config.json
_IMAGE_KEY   = "observation.images.base_rgb"
_SIM_OBS_KEY = "base_rgb"
_IMAGE_H     = 480
_IMAGE_W     = 640
_STATE_DIM   = 6   # joints only, no gripper in state
_N_OBS_STEPS = 2   # temporal window size
_N_ACT_STEPS = 32  # action chunk size


class LeRobotDiffusionAgent:
    """
    SplatSim agent wrapping a LeRobot DiffusionPolicy checkpoint.

    Produces action chunks of _N_ACT_STEPS steps and returns them one at a
    time. Re-runs inference only when the queue is empty.

    Args:
        checkpoint: HuggingFace repo id or local pretrained_model folder.
        device: "cuda" or "cpu".
        n_action_steps: overrides the default chunk size of 32.
    """

    def __init__(
        self,
        # checkpoint: str = "LuEduSoHu/robot_learning_tutorial_diffusion_0528_1804",
        checkpoint: str = "LuEduSoHu/robot_learning_tutorial/diffusion_0603_1702",
        device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        self.device = device
        self._n_act_steps = n_action_steps or _N_ACT_STEPS
        self._obs_buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=_N_OBS_STEPS)
        self._action_queue: list[np.ndarray] = []
        
        # checkpoint = "/home/magister/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial/diffusion"
        checkpoint = "/home/magister/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial/diffusion_0603_1702"
        
        # dataset_id = "LuEduSoHu/25051601"
        # dataset_root = "/home/magister/lerobot/splatsim_test_data/0529_1607"
        dataset_id = "LuEduSoHu/splatsim_lerobot_splatsim_10fps_260601_1756"
        dataset_root = "/home/magister/lerobot/splatsim_test_data/0601_1755"
        
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
        
        
        base_image = obs["base_rgb"]
        base_image = np.transpose(base_image.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
        base_image = (base_image * 255).astype(np.uint8)
        
        wrist_image = obs["wrist_rgb"]
        wrist_image = np.transpose(wrist_image.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
        wrist_image = (wrist_image * 255).astype(np.uint8)
        
        frame = {
            "base_rgb": base_image,
            "wrist_rgb": wrist_image,
        }
        for name in self.dataset_metadata.features['observation.state']['names']:
            frame[name] = state[self.dataset_metadata.features['observation.state']['names'].index(name)]

        frame = build_inference_frame(
            observation=frame,
            ds_features=self.dataset_metadata.features,
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
            obs_frame = self._build_obs(obs)

            obs = self.preprocess(obs_frame)

            action = self.policy.select_action(obs)
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
        
        print(f"ee_pose: ({ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f})")
        print(f"ee_rot: ({ee_rot[0]:.3f}, {ee_rot[1]:.3f}, {ee_rot[2]:.3f})")
        
        joints_n = 5
        ee_link_index = 6
        
        self.draw_pose(ee_pose, ee_rot, life_time=1)
        
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
    