"""Gripper del xArm5 Lite: joints espejo, apertura, estado y parche para el splat."""
import numpy as np

from splatsim.utils.robot_splat_render_utils import get_curr_link_states

from .config import (
    GRIPPER_RANGE, GRIPPER_MAX_ANGLE, SPLAT_GRIPPER_POSE,
)


class XArm5Gripper:
    PARENT = "drive_joint"
    CHILDREN = {
        "left_finger_joint": 1,
        "left_inner_knuckle_joint": 1,
        "right_outer_knuckle_joint": 1,
        "right_finger_joint": 1,
        "right_inner_knuckle_joint": 1,
    }

    def __init__(self, server):
        self.s = server
        self.parent_id = None
        self.children = {}
        self._patched = False

    # ── setup (lo llama setup_gripper de la base) ────────────────────────────
    def setup(self) -> None:
        s = self.s
        s.__parse_joint_info__()
        s.gripper_range = GRIPPER_RANGE

        parents = [j.id for j in s.joints if j.name == self.PARENT]
        if not parents:
            raise ValueError(
                f"No se encontró '{self.PARENT}'. "
                f"Joints disponibles: {[j.name for j in s.joints]}"
            )
        self.parent_id = parents[0]
        self.children = {j.id: self.CHILDREN[j.name]
                         for j in s.joints if j.name in self.CHILDREN}

        # la base podría leer estos atributos directamente
        s.mimic_parent_id = self.parent_id
        s.mimic_child_multiplier = self.children

        client = s.pybullet_client
        for joint_id, mult in self.children.items():
            c = client.createConstraint(
                s.dummy_robot, self.parent_id, s.dummy_robot, joint_id,
                jointType=client.JOINT_GEAR, jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            )
            client.changeConstraint(c, gearRatio=-mult, maxForce=100, erp=1)

    # ── control ──────────────────────────────────────────────────────────────
    def move(self, open_length: float, velocity: float = 2) -> None:
        s = self.s
        if not s.use_gripper:
            return
        # [0, 0.085] m  ->  [0, 0.85] rad
        angle = np.clip(open_length * 10, 0, GRIPPER_MAX_ANGLE)
        s.pybullet_client.setJointMotorControl2(
            s.dummy_robot, self.parent_id, s.pybullet_client.POSITION_CONTROL,
            targetPosition=angle,
            force=s.joints[self.parent_id].maxForce,
            maxVelocity=velocity,
        )

    def command(self, value: float) -> None:
        """value normalizado en [0, 1]."""
        self.move(value * GRIPPER_RANGE[1])
        self.s.current_gripper_action = value

    def get_state(self) -> float:
        """Estado normalizado en [0, 1]."""
        s = self.s
        if not s.use_gripper:
            return 0.0
        angle = s.pybullet_client.getJointState(s.dummy_robot, self.parent_id)[0]
        return float(np.clip(angle / GRIPPER_MAX_ANGLE, 0.0, 1.0))

    # ── parche del splat ─────────────────────────────────────────────────────
    def patch_for_splat(self) -> None:
        """
        initial_link_states se capturó con el gripper cerrado, pero el splat
        tiene el gripper en SPLAT_GRIPPER_POSE. Solo se parchean los links del gripper.
        Se ejecuta una sola vez.
        """
        if self._patched:
            return
        self._patched = True

        s, c = self.s, self.s.pybullet_client
        arm = [c.getJointState(s.dummy_robot, i)[0] for i in range(1, 6)]

        for jid in [self.parent_id, *self.children]:
            c.resetJointState(s.dummy_robot, jid, SPLAT_GRIPPER_POSE)
        for _ in range(50):
            c.stepSimulation()

        new_states = get_curr_link_states(s.dummy_robot, s.use_link_centers)
        for jid in [self.parent_id, *self.children]:
            s.initial_link_states[jid] = new_states[jid]

        for i, val in enumerate(arm):
            c.resetJointState(s.dummy_robot, i + 1, val)
        for _ in range(50):
            c.stepSimulation()
