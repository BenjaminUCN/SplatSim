"""Gripper del xArm5 Lite: joints espejo, apertura, estado y parche para el splat."""
import numpy as np

from splatsim.utils.robot_splat_render_utils import get_curr_link_states

from splatsim.utils.scene.hand_held import JAW_JOINT, load_design
from splatsim.utils.scene.hand_held_linkage import ROD_JOINT, Linkage

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
        self.splat_pose = SPLAT_GRIPPER_POSE

    # ── setup (lo llama el setup_gripper del servidor) ───────────────────────
    def setup(self) -> None:
        s = self.s
        s.parse_joint_info()

        parents = [j.id for j in s.joints if j.name == self.PARENT]
        if not parents:
            raise ValueError(
                f"No se encontró '{self.PARENT}'. "
                f"Joints disponibles: {[j.name for j in s.joints]}"
            )
        self.parent_id = parents[0]
        self.children = {j.id: self.CHILDREN[j.name]
                         for j in s.joints if j.name in self.CHILDREN}

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

    def follow(self) -> None:
        """Tras cada paso de la física: los joints que dependen del de mando (solo el gripper v4)."""

    def base_state(self) -> float:
        """Lo que get_observations divide por 0.8 para obs["state"] (get_current_gripper_state)."""
        return self.s.pybullet_client.getJointState(self.s.dummy_robot, self.parent_id)[0]

    def _splat_poses(self) -> dict:
        return {jid: self.splat_pose for jid in [self.parent_id, *self.children]}

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

        for jid, pose in self._splat_poses().items():
            c.resetJointState(s.dummy_robot, jid, pose)
        for _ in range(50):
            c.stepSimulation()

        new_states = get_curr_link_states(s.dummy_robot, s.use_link_centers)
        for jid in self._splat_poses():
            s.initial_link_states[jid] = new_states[jid]

        for i, val in enumerate(arm):
            c.resetJointState(s.dummy_robot, i + 1, val)
        for _ in range(50):
            c.stepSimulation()


class HandHeldGripper(XArm5Gripper):
    """Gripper hand_held_XL330: una sola articulación, el horn del XL330. Misma convención que el
    real: 0 = abierto, 1 = cerrado. El recorrido sale del límite del joint en el URDF."""
    PARENT = "gripper_drive_joint"
    CHILDREN = {}
    SPEED = 0.942     # rad/s: 9 rpm, el profile_velocity de configs/xarm5_real.yaml

    def setup(self) -> None:
        super().setup()
        s = self.s
        self.travel = s.pybullet_client.getJointInfo(s.dummy_robot, self.parent_id)[9]
        if self.travel <= 0:
            raise ValueError(f"{self.PARENT} needs a positive upper limit in the URDF")
        # fracción del recorrido en que estaba el gripper en el splat: 0 en el splat generado desde
        # el CAD; en un escaneo real, splat_gripper_value en objects.yaml
        value = s.object_config[s.robot_name].get("splat_gripper_value", 0.0)
        self.splat_pose = value * self.travel
        # la mandíbula sigue al horn por una varilla: aquí, proporcional a su ángulo
        jaws = [j for j in s.joints if j.name == JAW_JOINT]
        self.jaw_id = jaws[0].id if jaws else None
        self.jaw_travel = s.pybullet_client.getJointInfo(s.dummy_robot, self.jaw_id)[9] if jaws else 0.0
        self._patched = value == 0.0

    def move(self, open_length: float, velocity: float = SPEED) -> None:
        s = self.s
        if not s.use_gripper:
            return
        fraction = np.clip(open_length / GRIPPER_RANGE[1], 0.0, 1.0)
        s.pybullet_client.setJointMotorControl2(
            s.dummy_robot, self.parent_id, s.pybullet_client.POSITION_CONTROL,
            targetPosition=fraction * self.travel,
            force=s.joints[self.parent_id].maxForce,
            maxVelocity=self.SPEED,
        )
        if self.jaw_id is not None:
            s.pybullet_client.setJointMotorControl2(
                s.dummy_robot, self.jaw_id, s.pybullet_client.POSITION_CONTROL,
                targetPosition=fraction * self.jaw_travel,
                force=s.joints[self.jaw_id].maxForce,
                maxVelocity=self.SPEED * self.jaw_travel / self.travel,
            )

    def _splat_poses(self) -> dict:
        poses = super()._splat_poses()
        if self.jaw_id is not None:
            poses[self.jaw_id] = self.splat_pose / self.travel * self.jaw_travel
        return poses

    def get_state(self) -> float:
        s = self.s
        if not s.use_gripper:
            return 0.0
        angle = s.pybullet_client.getJointState(s.dummy_robot, self.parent_id)[0]
        return float(np.clip(angle / self.travel, 0.0, 1.0))

    def base_state(self) -> float:
        return self.get_state() * 0.8


class HandHeldLinkageGripper(HandHeldGripper):
    """hand_held_XL330 v4: el horn mueve la mandíbula por un cuatro barras (bastón), no en proporción.
    Solo el horn tiene motor; la mandíbula y el bastón se ponen desde su ángulo tras cada paso (follow)."""
    MESH_DIR = "splatsim/robot_definitions/meshes/hand_held_xl330_v4"

    def setup(self) -> None:
        super().setup()
        s, c = self.s, self.s.pybullet_client
        self.linkage = Linkage.from_manifest(load_design(self.MESH_DIR)["linkage"])
        rods = [j for j in s.joints if j.name == ROD_JOINT]
        if self.jaw_id is None or not rods:
            raise ValueError(f"{s.robot_name}: the v4 gripper needs {JAW_JOINT} and {ROD_JOINT} in the URDF")
        self.rod_id = rods[0].id
        for jid in (self.jaw_id, self.rod_id):     # sin motor: los mueve follow()
            c.setJointMotorControl2(s.dummy_robot, jid, c.VELOCITY_CONTROL, force=0)

    def move(self, open_length: float, velocity: float = HandHeldGripper.SPEED) -> None:
        s = self.s
        if not s.use_gripper:
            return
        fraction = np.clip(open_length / GRIPPER_RANGE[1], 0.0, 1.0)
        s.pybullet_client.setJointMotorControl2(
            s.dummy_robot, self.parent_id, s.pybullet_client.POSITION_CONTROL,
            targetPosition=fraction * self.travel, force=s.joints[self.parent_id].maxForce,
            maxVelocity=self.SPEED)

    def follow(self) -> None:
        s, c = self.s, self.s.pybullet_client
        jaw, rod = self.linkage.jaw_and_rod(c.getJointState(s.dummy_robot, self.parent_id)[0])
        c.resetJointState(s.dummy_robot, self.jaw_id, jaw)
        c.resetJointState(s.dummy_robot, self.rod_id, rod)

    def _splat_poses(self) -> dict:
        jaw, rod = self.linkage.jaw_and_rod(self.splat_pose)
        return {self.parent_id: self.splat_pose, self.jaw_id: jaw, self.rod_id: rod}


GRIPPERS = {"xarm": XArm5Gripper, "hand_held_xl330": HandHeldGripper,
            "hand_held_xl330_v4": HandHeldLinkageGripper}
