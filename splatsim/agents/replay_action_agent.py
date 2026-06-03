from __future__ import annotations

import numpy as np
from typing import List
import pybullet as p
import pybullet_data
from splatsim.agents.agent import Agent
import pickle
import os
from tqdm import tqdm
from gello.env import RobotEnv
import cv2


from collections import deque
from typing import Any, Dict

import torch

import pybullet as p

# Constants from config.json
_IMAGE_KEY   = "observation.images.base_rgb"
_SIM_OBS_KEY = "base_rgb"
_IMAGE_H     = 480
_IMAGE_W     = 640
_STATE_DIM   = 6   # joints only, no gripper in state
_N_OBS_STEPS = 2   # temporal window size
_N_ACT_STEPS = 32  # action chunk size

class ReplayActionAgent(Agent): 
    def __init__(
        self,
        traj_folder: str,
        env: RobotEnv,
        save_images: bool = False,
        device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.last_action = np.array([0, 0, 0, 0, 0, 0, 1])  # 7-DoF

        self.traj_folder = traj_folder
        self.image_folder = None
        self.save_images = save_images
        self.traj_index = -1
        self.traj_subfolders = sorted(os.listdir(traj_folder))
        self.load_next_recorded_trajectory()
        
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
        
    ### Replay code ###

    def load_next_recorded_trajectory(self):
        self.traj_index += 1
        print(f"Loading trajectory {self.traj_index + 1}/{len(self.traj_subfolders)}: {self.traj_subfolders[self.traj_index]}")
        if self.save_images:
            self.image_folder = os.path.join(self.traj_folder, self.traj_subfolders[self.traj_index], 'images_1')
            os.makedirs(self.image_folder, exist_ok=True)

            # Delete existing images in the folder if any
            for filename in os.listdir(self.image_folder):
                file_path = os.path.join(self.image_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # take only those filenames which end with .pkl
        file_names = os.listdir(os.path.join(self.traj_folder, self.traj_subfolders[self.traj_index]))
        file_names.sort()

        self.trajectory_iter = iter(file_names)

    def next_trajectory_step(self):
        file = next(self.trajectory_iter, None)
        if file is None:
            print("No more recorded trajectories.")
            return None
            if self.traj_index + 1 < len(self.traj_subfolders):
                self.load_next_recorded_trajectory()
                return self.next_trajectory_step()
            else:
                print("No more recorded trajectories.")
                return None
        if not file.endswith('.pkl'):
            print(f"Skipping non-pkl file: {file}")
            return self.next_trajectory_step()
        file_path = os.path.join(self.traj_folder, self.traj_subfolders[self.traj_index], file)
        file = open(file_path, 'rb')
        data = pickle.load(file)

        cur_joint = data['action'][:]
        cur_joint = cur_joint.tolist()
        # Add the world joint to the recorded joint state
        # cur_joint = [0] + cur_joint 
        cur_joint = np.array(cur_joint)

        object_list = [object_position_key[:-len("_position")] for object_position_key in data.keys() if object_position_key.endswith("_position")]
        # gripper_position is for gello integration. Ignore this value
        if "gripper" in object_list:
            object_list.remove("gripper")
        for object_name in object_list:
            cur_object_position = np.array(data[object_name + '_position'])
            cur_object_rotation = np.array(data[object_name + '_orientation'])
            # cur_object_rotation = np.roll(cur_object_rotation, 1)
            # Disable gravity for objects when replaying a trajectory so that there's no jittering
            self.env._robot.set_object_pose(object_name, cur_object_position, cur_object_rotation, use_gravity=False)

        return cur_joint
    
    ### Inverse Kinematics code ###
    
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
                
                # print(f"Joint {i}: {joint_name}, type={joint_type}, limits=({ll}, {ul})")

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

    def act(self, obs):
        obs_dict = obs.copy()
        
        for i in range(1, 6):
            p.resetJointState(self.dummy_robot, i, obs_dict['joint_positions'][i-1])
        
        action = self.next_trajectory_step()
        if action is None:
            print("No more trajectory steps available.")
            return self.last_action
        
        ee_pose = action[:3]
        ee_rot = action[3:6]
        ee_gripper = action[6]
        
        print(f"ee_pose: ({ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f})")
        print(f"ee_rot: ({ee_rot[0]:.3f}, {ee_rot[1]:.3f}, {ee_rot[2]:.3f})")
        
        joints_n = 5
        
        (
        joint_indices,
        lower_limits,
        upper_limits,
        joint_ranges,
        joint_names,
        ) = self.get_revolute_joint_limits(robot_id=self.dummy_robot)
        
        ee_link = 6
        euler = [ee_rot[0], ee_rot[1], ee_rot[2]]
        
        euler = [0, 0, 0]
        # ee_pose = [0.5, 0.1, 0.1]
        ee_quat = p.getQuaternionFromEuler(euler)
        
        self.draw_pose(ee_pose, ee_quat, life_time=1)
        
        dummy_joint_pos = p.calculateInverseKinematics(
            self.dummy_robot, 
            # 6, 
            # ee_pose,
            # ee_quat,
            endEffectorLinkIndex=ee_link,
            targetPosition=ee_pose,
            targetOrientation=ee_rot,
            residualThreshold=0.00001,
            maxNumIterations=100000,
             
            # lowerLimits=lower_limits,
            # upperLimits=upper_limits,
            # jointRanges=joint_ranges,
            restPoses = obs_dict["joint_positions"][:len(joint_indices)]
        )
        
        # calculate difference between current and target joint angles
        joint_diff = np.array(dummy_joint_pos)[:6] - np.array(obs_dict['joint_positions'])[:6]
        if np.linalg.norm(joint_diff) < 2.0:
            self.make_new_prediction = True
        else:
            print(f"Joint difference norm: {np.linalg.norm(joint_diff):.3f}")
            print(f"Joint difference: {joint_diff}")

        joints = np.array(dummy_joint_pos)[:joints_n]
        joints = np.append(joints, ee_gripper)
        self.last_action = joints
        
        return joints
        
        angles = None
        if self.last_action is np.array([0, 0, 0, 0, 0, 0, 1]):
            angles = self.next_trajectory_step()
            print(f"Next action: {angles}")
        
            
        if angles is None:
            print("No more trajectory steps available.")
            return self.last_action
        else:
            if self.save_images:
                for image_name in [image_name for image_name in obs.keys() if image_name.endswith("_rgb") and obs[image_name] is not None]:
                    frame = obs[image_name]
                    frame = np.transpose(frame.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
                    frame = (frame * 255).astype(np.uint8)
                    image_index = len(os.listdir(self.image_folder))
                    image_path = os.path.join(self.image_folder, f"{image_name}_{image_index:05d}.png")
                    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.last_action = angles
            return angles
