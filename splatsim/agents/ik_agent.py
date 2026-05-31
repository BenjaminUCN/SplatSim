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


class InverseKinematicAgent:
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
        device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        self.device = device
        self._n_act_steps = n_action_steps or _N_ACT_STEPS
        self._obs_buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=_N_OBS_STEPS)
        self._action_queue: list[np.ndarray] = []
        
        p.connect(p.GUI)
        
        urdf_path = "/home/magister/data/xarm5_lite_urdf/ufactory_xarm5_urdf/xarm5.urdf"
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        base_position = [0, 0, 0]

        flags = p.URDF_USE_IMPLICIT_CYLINDER
        self.dummy_robot = p.loadURDF(
            urdf_path, useFixedBase=True, basePosition=base_position, flags=flags
        )
        
        self._init_sliders()


    def _init_sliders(self):
        self.slider_ee_x = p.addUserDebugParameter("EE X", -0.9, 0.9, 0.5)
        self.slider_ee_y = p.addUserDebugParameter("EE Y", -0.9, 0.9, 0.0)
        self.slider_ee_z = p.addUserDebugParameter("EE Z", -0.9, 0.9, 0.1)
        
        self.slider_ee_roll = p.addUserDebugParameter("EE Roll", -3.14, 3.14, 0)
        self.slider_ee_pitch = p.addUserDebugParameter("EE Pitch", -3.14, 3.14, 0)
        self.slider_ee_yaw = p.addUserDebugParameter("EE Yaw", -3.14, 3.14, 0)


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
                
                print(f"Joint {i}: {joint_name}, type={joint_type}, limits=({ll}, {ul})")

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
        obs_dict = obs.copy()
        
        for i in range(1, 6):
            p.resetJointState(self.dummy_robot, i, obs_dict['joint_positions'][i-1])
        
        joints_n = 5
        
        (
        joint_indices,
        lower_limits,
        upper_limits,
        joint_ranges,
        joint_names,
        ) = self.get_revolute_joint_limits(robot_id=self.dummy_robot)
        
        ee_link = 6
        
        roll = p.readUserDebugParameter(self.slider_ee_roll)
        pitch = p.readUserDebugParameter(self.slider_ee_pitch)
        yaw = p.readUserDebugParameter(self.slider_ee_yaw)
        
        x = p.readUserDebugParameter(self.slider_ee_x)
        y = p.readUserDebugParameter(self.slider_ee_y)
        z = p.readUserDebugParameter(self.slider_ee_z)
        
        # euler = [0, 0, 0]
        # ee_pose = [0.5, 0.1, 0.1]
        euler = [roll, pitch, yaw]
        ee_pose = [x, y, z]
        ee_quat = p.getQuaternionFromEuler(euler)
        # ee_quat = [0, 0, 0, 1]
        
        state = p.getLinkState(self.dummy_robot, ee_link)
        actual_pos = state[4]
        actual_quat = state[5]
        self.draw_pose(actual_pos, actual_quat, life_time=1)
        
        self.draw_pose(ee_pose, ee_quat, life_time=1)
        
        # print(f"Number of joints: {p.getNumJoints(self.dummy_robot)}")
        
        # lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
        # upper_limits = [np.pi, 0, np.pi, np.pi, np.pi, np.pi]
        dummy_joint_pos = p.calculateInverseKinematics(
            self.dummy_robot, 
            # 6, 
            # ee_pose,
            # ee_quat,
            endEffectorLinkIndex=ee_link,
            targetPosition=ee_pose,
            targetOrientation=ee_quat,
            residualThreshold=0.00001,
            maxNumIterations=100000,
             
            # lowerLimits=lower_limits,
            # upperLimits=upper_limits,
            # jointRanges=joint_ranges,
            # restPoses = [-3.1415, -0.4886, -0.5235, -0.6108, 0.0]
            # restPoses = obs_dict["joint_positions"][:len(joint_indices)]
        )

        joints = np.array(dummy_joint_pos)[:joints_n]
        joints = np.append(joints, 1)
        
        return joints