"""Pruebas del gripper hand_held_XL330, sin hardware, PyBullet ni GPU.

    python -m unittest tests/test_hand_held_xl330.py        (desde la raíz del repositorio)
"""
import math
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from splatsim.robots.xl330_gripper import ALERT_BIT, XL330Gripper
from splatsim.utils.scene.hand_held import (
    FLANGE_CENTER_MM, GRIPPER_CENTER_MM, JAW_JOINT, JAW_TRAVEL_RAD, MESH_DIR, PIVOT_MM, build_urdf, in_box, load_manifest, merge_splat, origin_matrix, read_ply, read_stl, sample_surface, write_ply,
)

OPEN, CLOSED = 1000, 2224


class FakeServo:
    """DynamixelSDK a nivel de registros: un XL330 con su tabla de control."""

    def __init__(self, position=OPEN):
        self.t = {11: 3, 64: 0, 70: 0, 102: 0, 108: 0, 112: 0, 116: 0, 126: 0, 128: 0, 132: position}
        self.online = True
        self.reboots = 0
        self.goals = []
        self.eeprom_writes = 0

    # PortHandler
    def openPort(self): return True
    def setBaudRate(self, b): return True
    def closePort(self): pass
    def getPortName(self): return "fake"

    # PacketHandler
    def _comm(self): return 0 if self.online else -3001
    def _err(self): return ALERT_BIT if self.t[70] else 0
    def ping(self, port, i): return 1200, self._comm(), 0

    def _read(self, port, i, addr): return self.t[addr], self._comm(), self._err()
    read1ByteTxRx = read2ByteTxRx = read4ByteTxRx = _read

    def _write(self, port, i, addr, value):
        if self.online:
            if addr == 116:
                value = value - (1 << 32) if value >= 1 << 31 else value
                self.goals.append(value)
            if addr == 11:
                self.eeprom_writes += 1
            self.t[addr] = value
        return self._comm(), self._err()
    write1ByteTxRx = write2ByteTxRx = write4ByteTxRx = _write

    def readTxRx(self, port, i, addr, length):
        assert (addr, length) == (126, 10)
        raw = (self.t[126] & 0xFFFF).to_bytes(2, "little") + (self.t[128] & 0xFFFFFFFF).to_bytes(4, "little") \
            + (self.t[132] & 0xFFFFFFFF).to_bytes(4, "little")
        return list(raw), self._comm(), self._err()

    def reboot(self, port, i):
        self.reboots += 1
        self.t[70] = 0
        self.t[64] = 0
        return self._comm(), 0


def make_gripper(servo, **kw):
    kw = {"open_position": OPEN, "closed_position": CLOSED, "stall_seconds": 2.0, **kw}
    g = XL330Gripper(handlers=(servo, servo), **kw)
    g.recover()
    return g


class TestDriver(unittest.TestCase):
    def test_connecting_does_not_move(self):
        s = FakeServo(position=1500)
        g = make_gripper(s)
        self.assertIsNone(g.problem)
        self.assertEqual(s.t[64], 1)
        self.assertEqual(s.t[11], 5)
        self.assertEqual(set(s.goals), {1500})
        self.assertAlmostEqual(g.get_state(), 500 / 1224)
        g._tick(0.0)
        self.assertEqual(set(s.goals), {1500})

    def test_mode_written_only_if_different(self):
        s = FakeServo()
        s.t[11] = 5
        make_gripper(s)
        self.assertEqual(s.eeprom_writes, 0)

    def test_uncalibrated_never_enables_torque(self):
        s = FakeServo()
        g = XL330Gripper(handlers=(s, s))
        g.recover()
        self.assertIn("not calibrated", g.problem)
        self.assertEqual(s.t[64], 0)
        g.command(1.0)
        self.assertEqual(s.goals, [])

    def test_absurd_travel_is_not_calibrated(self):
        self.assertFalse(XL330Gripper(open_position=0, closed_position=4000).calibrated)
        self.assertFalse(XL330Gripper(open_position=0, closed_position=10).calibrated)

    def test_open_is_zero_closed_is_one(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(1.0)
        g._tick(0.0)
        self.assertEqual(s.goals[-1], CLOSED)
        g.command(0.0)
        g._tick(0.1)
        self.assertEqual(s.goals[-1], OPEN)
        g.command(0.5)
        g._tick(0.2)
        self.assertEqual(s.goals[-1], OPEN + 612)

    def test_inverted_direction(self):
        s = FakeServo(position=CLOSED)
        g = make_gripper(s, open_position=CLOSED, closed_position=OPEN)
        self.assertEqual(g.get_state(), 0.0)
        g.command(1.0)
        g._tick(0.0)
        self.assertEqual(s.goals[-1], OPEN)

    def test_position_in_another_turn_after_reboot(self):
        # el calibrado cruza el cero: tras reiniciarse el servo reporta dentro de una vuelta
        s = FakeServo(position=4090)
        g = make_gripper(s, open_position=-200, closed_position=1000)
        self.assertAlmostEqual(g.get_state(), 194 / 1200)
        self.assertEqual(set(s.goals), {4090})
        g.command(0.0)
        g._tick(0.0)
        self.assertEqual(s.goals[-1], 4096 - 200)
        g.command(1.0)
        g._tick(0.1)
        self.assertEqual(s.goals[-1], 4096 + 1000)

    def test_negative_goal_is_written_as_twos_complement(self):
        s = FakeServo(position=-100)
        g = make_gripper(s, open_position=-200, closed_position=1000)
        g.command(0.0)
        g._tick(0.0)
        self.assertEqual(s.goals[-1], -200)

    def test_deadband_and_ends(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(0.5)
        g._tick(0.0)
        n = len(s.goals)
        g.command(0.51)
        g._tick(0.1)
        self.assertEqual(len(s.goals), n)
        g.command(0.99)
        g._tick(0.2)
        g.command(1.0)
        g._tick(0.3)
        self.assertEqual(s.goals[-1], CLOSED)

    def test_invalid_commands(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(float("nan"))
        self.assertIsNone(g._target)
        g.command(7)
        g._tick(0.0)
        self.assertEqual(s.goals[-1], CLOSED)

    def test_stall_stops_pushing_until_a_new_command(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(1.0)
        g._tick(0.0)
        s.t[132] = 1800
        g._tick(1.0)
        self.assertFalse(g.stalled)
        g._tick(3.1)
        self.assertTrue(g.stalled)
        self.assertEqual(s.goals[-1], 1800)
        n = len(s.goals)
        for value, t in ((1.0, 4.0), (0.95, 5.0)):      # el mismo comando o ruido no vuelven a empujar
            g.command(value)
            g._tick(t)
        self.assertEqual(len(s.goals), n)
        g.command(0.0)
        g._tick(6.0)
        self.assertFalse(g.stalled)
        self.assertEqual(s.goals[-1], OPEN)

    def test_moving_is_not_a_stall(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(1.0)
        s.t[128] = 40
        for t in range(10):
            g._tick(float(t))
        self.assertFalse(g.stalled)

    def test_hardware_error_stops_and_recover_reboots(self):
        s = FakeServo()
        g = make_gripper(s)
        s.t[70] = 1 << 5
        g._tick(0.0)
        self.assertIn("overload", g.problem)
        g.command(1.0)
        n = len(s.goals)
        with g._bus_lock:
            if g._ready:
                g._tick(0.1)
        self.assertEqual(len(s.goals), n)
        self.assertIsNone(g.recover())
        self.assertEqual(s.reboots, 1)
        self.assertEqual(s.t[64], 1)
        self.assertIsNone(g._target)      # la orden anterior no se retoma

    def test_lost_usb(self):
        s = FakeServo()
        g = make_gripper(s)
        s.online = False
        for t in range(9):
            g._tick(float(t))
        self.assertIsNone(g.problem)
        g._tick(9.0)
        self.assertIn("no response", g.problem)
        self.assertFalse(g._ready)
        self.assertIsNotNone(g.recover())  # sigue desconectado
        s.online = True
        self.assertIsNone(g.recover())

    def test_hold(self):
        s = FakeServo()
        g = make_gripper(s)
        g.command(1.0)
        g._tick(0.0)
        s.t[132] = 1400
        g.hold()
        g._tick(0.1)
        self.assertEqual(s.goals[-1], 1400)
        g._tick(0.2)
        self.assertEqual(s.goals[-1], 1400)

    def test_close_releases_torque(self):
        s = FakeServo()
        g = make_gripper(s)
        g.close()
        self.assertEqual(s.t[64], 0)

    def test_thread_end_to_end(self):
        s = FakeServo()
        g = XL330Gripper(handlers=(s, s), open_position=OPEN, closed_position=CLOSED, hz=200)
        g.start()
        g.command(1.0)
        for _ in range(200):
            if s.goals[-1] == CLOSED:
                break
            import time
            time.sleep(0.01)
        self.assertEqual(s.goals[-1], CLOSED)
        g.close()


class FakeExternal:
    def __init__(self):
        self.commands, self.holds, self.closed = [], 0, False
    def command(self, v): self.commands.append(v)
    def hold(self): self.holds += 1
    def close(self): self.closed = True
    def get_state(self): return 0.25


class TestXArmRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("pyquaternion", types.SimpleNamespace(Quaternion=object))
        from splatsim.robots.xarm5_lite_robot_plant import XArmRobot
        cls.XArmRobot = XArmRobot

    def test_no_command_at_startup_then_one_per_client_command(self):
        ext = FakeExternal()
        r = self.XArmRobot(real=False, external_gripper=ext)
        self.assertEqual(ext.commands, [])
        self.assertIsNone(r.target_command["gripper"])
        r.command_joint_state(np.array([0, 0, 0, 0, 0, 1.0]))
        self.assertEqual(ext.commands, [1.0])
        r.command_joint_state(np.zeros(5))      # sin gripper
        self.assertEqual(ext.commands, [1.0])

    def test_stopped_arm_does_not_move_the_blade(self):
        ext = FakeExternal()
        r = self.XArmRobot(real=False, external_gripper=ext)
        r._pause_arm("state 4")
        self.assertEqual(ext.holds, 1)
        r.command_joint_state(np.array([0, 0, 0, 0, 0, 1.0]))
        self.assertEqual(ext.commands, [])

    def test_state_and_stop(self):
        ext = FakeExternal()
        r = self.XArmRobot(real=False, external_gripper=ext)
        self.assertEqual(r._get_gripper_pos(), 0.25)
        r.stop()
        self.assertTrue(ext.closed)

    def test_without_external_gripper_is_as_before(self):
        r = self.XArmRobot(real=False)
        self.assertEqual(r.target_command["gripper"], 0)
        self.assertEqual(r._get_gripper_pos(), 0.0)


FIXTURE = """<robot name="xarm5">
<link name="world"/><link name="link_base"/>
<joint name="world_joint" type="fixed"><parent link="world"/><child link="link_base"/></joint>
{arm}
<link name="link_eef"/>
<joint name="joint_eef" type="fixed"><origin xyz="0 0 0.01" rpy="0 0 0"/><parent link="link5"/><child link="link_eef"/></joint>
<link name="xarm_gripper_base_link"/>
<joint name="gripper_fix" type="fixed"><origin xyz="0 0 0.002" rpy="0 0 0.3"/><parent link="link_eef"/><child link="xarm_gripper_base_link"/></joint>
<link name="left_outer_knuckle"/>
<joint name="drive_joint" type="revolute"><origin xyz="0 0.035 0.06"/><parent link="xarm_gripper_base_link"/><child link="left_outer_knuckle"/><axis xyz="1 0 0"/><limit lower="0" upper="0.85" effort="50" velocity="2"/></joint>
<link name="left_finger"/>
<joint name="left_finger_joint" type="revolute"><parent link="left_outer_knuckle"/><child link="left_finger"/><axis xyz="1 0 0"/><limit lower="0" upper="0.85" effort="50" velocity="2"/></joint>
<link name="link_tcp"/>
<joint name="joint_tcp" type="fixed"><origin xyz="0 0 0.172"/><parent link="xarm_gripper_base_link"/><child link="link_tcp"/></joint>
<link name="wrist_camera_link"/>
<joint name="wrist_fix" type="fixed"><origin xyz="0.07 0 0.03" rpy="0.1 0.2 0.3"/><parent link="xarm_gripper_base_link"/><child link="wrist_camera_link"/></joint>
<link name="camera_optical"/>
<joint name="camera_optical_fix" type="fixed"><origin xyz="0 0.01 0"/><parent link="wrist_camera_link"/><child link="camera_optical"/></joint>
<transmission name="tran"><joint name="drive_joint"/></transmission>
</robot>"""


def fixture_urdf(path):
    arm = ""
    parent = "link_base"
    for i in range(1, 6):
        arm += (f'<link name="link{i}"/><joint name="joint{i}" type="revolute"><origin xyz="0 0 0.1"/>'
                f'<parent link="{parent}"/><child link="link{i}"/><axis xyz="0 0 1"/>'
                '<limit lower="-3" upper="3" effort="50" velocity="2"/></joint>')
        parent = f"link{i}"
    with open(path, "w") as f:
        f.write(FIXTURE.format(arm=arm))


def fk(path, link):
    """Pose del link respecto de la raíz, con todas las articulaciones en 0."""
    joints = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
    T = np.eye(4)
    while link in joints:
        T = origin_matrix(joints[link].find("origin")) @ T
        link = joints[link].find("parent").get("link")
    return T


class TestUrdf(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.src = os.path.join(self.dir, "xarm5.urdf")
        self.dst = os.path.join(self.dir, "xarm5_hand_held.urdf")
        fixture_urdf(self.src)
        self.addCleanup(shutil.rmtree, self.dir)

    def build(self, **kw):
        return build_urdf(self.src, self.dst, 1.878, **kw)

    def test_old_gripper_is_gone_and_tree_is_valid(self):
        info = self.build()
        root = ET.parse(self.dst).getroot()
        links = {l.get("name") for l in root.findall("link")}
        for name in ("xarm_gripper_base_link", "left_outer_knuckle", "left_finger", "link_tcp"):
            self.assertNotIn(name, links)
        self.assertEqual(root.findall("transmission"), [])
        children = [j.find("child").get("link") for j in root.findall("joint")]
        parents = {j.find("parent").get("link") for j in root.findall("joint")}
        self.assertEqual(len(children), len(set(children)))
        self.assertEqual(set(links) - set(children), {"world"})
        self.assertLessEqual(parents, links)
        self.assertEqual(info["reparented"], ["wrist_camera_link"])

    def test_wrist_camera_keeps_its_pose(self):
        before = fk(self.src, "camera_optical")
        self.build()
        np.testing.assert_allclose(fk(self.dst, "camera_optical"), before, atol=1e-7)

    def test_tool_geometry(self):
        self.build(roll_deg=0)
        # con el marco viejo girado 0.3 rad alrededor de z, se compara en ese marco
        T_flange = fk(self.src, "xarm_gripper_base_link")
        T_base = fk(self.dst, "hand_held_base")
        rel = np.linalg.inv(T_flange) @ T_base
        def in_flange(mm):
            return (rel @ np.array([*np.array(mm) / 1000, 1]))[:3]
        np.testing.assert_allclose(in_flange((0, -51.63, -15.8)), (0, 0, 0), atol=1e-7)  # centro de la brida
        np.testing.assert_allclose(in_flange((-2, 158.27, -15.8)), (0, -0.002, 0.2099), atol=1e-7)  # punta
        np.testing.assert_allclose(in_flange((0, -51.63, 50)), (0.0658, 0, 0), atol=1e-7)  # z del ensamble hacia +x
        self.assertAlmostEqual(np.linalg.det(rel[:3, :3]), 1.0)

    def test_disc_is_bolted_to_the_plate(self):
        disc = read_stl(os.path.join(MESH_DIR, "mount_xarm5.stl")).reshape(-1, 3)
        plate = read_stl(os.path.join(MESH_DIR, "bottom_plate.stl")).reshape(-1, 3)
        # sin hueco: la cara del disco toca la de la placa, y la otra cara es la de la brida
        self.assertAlmostEqual(disc[:, 1].max(), plate[:, 1].min(), delta=0.01)
        self.assertAlmostEqual(disc[:, 1].min(), FLANGE_CENTER_MM[1], delta=0.01)
        self.assertAlmostEqual((disc[:, 0].max() + disc[:, 0].min()) / 2, FLANGE_CENTER_MM[0], delta=0.02)
        self.assertAlmostEqual((disc[:, 2].max() + disc[:, 2].min()) / 2, FLANGE_CENTER_MM[2], delta=0.02)

    def test_roll_rotates_about_the_tool_axis(self):
        self.build(roll_deg=180)
        T_flange = fk(self.src, "xarm_gripper_base_link")
        rel = np.linalg.inv(T_flange) @ fk(self.dst, "hand_held_base")
        np.testing.assert_allclose((rel @ np.array([0, -0.05163, 0.05, 1]))[:3], (-0.0658, 0, 0), atol=1e-7)

    def test_horn_joint(self):
        self.build()
        joint = [j for j in ET.parse(self.dst).getroot().findall("joint") if j.get("name") == "gripper_drive_joint"][0]
        self.assertEqual(joint.find("axis").get("xyz"), "0 0 1")
        self.assertEqual(joint.find("limit").get("upper"), "1.87800")
        np.testing.assert_allclose(origin_matrix(joint.find("origin"))[:3, 3], (0.0127, 0.01386524, 0), atol=1e-6)
        # la malla del horn queda centrada en el eje del joint
        horn = read_stl(os.path.join(self.dir, "hand_held_xl330", "horn.stl"))
        center = (horn.reshape(-1, 3).min(0) + horn.reshape(-1, 3).max(0)) / 2
        np.testing.assert_allclose(center[:2], (12.7, 13.865), atol=0.02)

    def test_jaw_opens_at_zero_and_closes_on_the_meshes(self):
        self.build()
        root = ET.parse(self.dst).getroot()
        joint = [j for j in root.findall("joint") if j.get("name") == JAW_JOINT][0]
        self.assertEqual((joint.find("parent").get("link"), joint.find("child").get("link")),
                         ("hand_held_base", "hand_held_jaw"))
        self.assertEqual(joint.find("axis").get("xyz"), "0 0 1")
        self.assertAlmostEqual(float(joint.find("limit").get("upper")), JAW_TRAVEL_RAD)
        T0 = origin_matrix(joint.find("origin"))
        np.testing.assert_allclose(T0[:2, 3], np.array(PIVOT_MM) / 1000, atol=1e-6)
        shift = np.array([-PIVOT_MM[0], -PIVOT_MM[1], 0, 0]) / 1000
        tip_mm = read_stl(os.path.join(MESH_DIR, "moving_jaw.stl")).reshape(-1, 3)
        tip_mm = tip_mm[np.argmax(tip_mm[:, 1])]
        tip_link = np.array([*tip_mm / 1000, 1]) + shift
        a = JAW_TRAVEL_RAD
        Rz = np.array([[math.cos(a), -math.sin(a), 0, 0], [math.sin(a), math.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        closed, opened = (T0 @ Rz @ tip_link)[:3] * 1000, (T0 @ tip_link)[:3] * 1000
        np.testing.assert_allclose(closed, tip_mm, atol=1e-2)       # cerrada: donde está la malla
        anvil_x = read_stl(os.path.join(MESH_DIR, "anvil.stl")).reshape(-1, 3)[:, 0].max()
        self.assertGreater(opened[0] - anvil_x, 30)                 # abierta: lejos del yunque

    def test_grasp_point_is_on_the_real_jaw_edge(self):
        jaw = read_stl(os.path.join(MESH_DIR, "moving_jaw.stl")).reshape(-1, 3)
        self.assertLess(np.linalg.norm(jaw - np.array(GRIPPER_CENTER_MM), axis=1).min(), 6)

    def test_moving_part_and_missing_gripper(self):
        extra = os.path.join(self.dir, "hoja.stl")
        shutil.copy2(os.path.join(MESH_DIR, "moving_jaw.stl"), extra)
        self.build(extra_horn_mesh=extra)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "hand_held_xl330", "hoja.stl")))
        with open(self.dst, encoding="utf-8") as f:
            self.assertIn("hand_held_xl330/hoja.stl", f.read())
        with self.assertRaises(ValueError):
            build_urdf(self.dst, self.dst + ".2", 1.878)

    def test_all_mesh_references_exist(self):
        self.build()
        for mesh in ET.parse(self.dst).getroot().iter("mesh"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, mesh.get("filename"))), mesh.get("filename"))


class TestMeshes(unittest.TestCase):
    def test_volume_matches_cad(self):
        for part in load_manifest():
            t = read_stl(os.path.join(MESH_DIR, part["file"]))
            volume = abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6
            self.assertAlmostEqual(volume / part["volume_mm3"], 1.0, delta=0.05, msg=part["name"])

    def test_sampling_stays_on_the_part(self):
        t = read_stl(os.path.join(MESH_DIR, "moving_jaw.stl"))
        pts = sample_surface(t, 1.5, np.random.default_rng(0))
        v = t.reshape(-1, 3)
        self.assertGreater(len(pts), 1000)
        self.assertTrue(np.all(pts >= v.min(0) - 1e-6) and np.all(pts <= v.max(0) + 1e-6))


def fake_splat(n=1000):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(45)],
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rng = np.random.default_rng(1)
    v = np.zeros(n, np.dtype([(k, "<f4") for k in names]))
    for k in names:
        v[k] = rng.normal(size=n)
    a = 2.0
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) * a
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rz, (0.5, -0.2, 0.1)
    return v, M


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.v, self.M = fake_splat()
        xyz = np.stack([self.v["x"], self.v["y"], self.v["z"]], 1).astype(float)
        self.world = xyz @ self.M[:3, :3].T + self.M[:3, 3]
        self.aabb = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
        inside = in_box(self.world, self.aabb)
        self.labels = np.arange(inside.sum()) % 4        # 0, 1 quedan (2 pasa a 5), 3 se borra
        self.map = {0: 0, 1: 1, 2: 5}
        self.new_world = np.array([[1.5, 0, 0], [0.2, 0.1, 0.0]])

    def merge(self):
        return merge_splat(self.v, self.labels, self.M, self.aabb, self.map, self.new_world,
                           np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]), [7, 8], 0.001)

    def test_counts_and_labels_align_with_the_renderer(self):
        merged, labels, aabb, dropped = self.merge()
        self.assertEqual(dropped, int((self.labels == 3).sum()))
        self.assertEqual(len(merged), len(self.v) - dropped + 2)
        xyz = np.stack([merged["x"], merged["y"], merged["z"]], 1).astype(float)
        world = xyz @ self.M[:3, :3].T + self.M[:3, 3]
        # igual que get_segmented_indices: una etiqueta por gaussiana dentro de la caja
        self.assertEqual(in_box(world, aabb).sum(), len(labels))
        self.assertEqual(labels[-2:].tolist(), [7, 8])
        self.assertFalse(np.isin(labels, [2, 3]).any())
        self.assertGreater(aabb[1][0], 1.5)

    def test_kept_gaussians_keep_their_label(self):
        merged, labels, aabb, _ = self.merge()
        xyz = np.stack([merged["x"], merged["y"], merged["z"]], 1).astype(float)
        world = xyz @ self.M[:3, :3].T + self.M[:3, 3]
        selected = in_box(world, aabb)
        old = {tuple(np.round(p, 4)): l for p, l in zip(self.world[in_box(self.world, self.aabb)], self.labels)}
        for p, l in zip(world[selected][:-2], labels[:-2]):
            key = tuple(np.round(p, 4))
            if key in old:
                self.assertEqual(l, self.map[int(old[key])])
            else:
                self.assertEqual(l, -1)    # fondo que entró a la caja nueva

    def test_new_gaussians(self):
        merged, *_ = self.merge()
        new = merged[-2:]
        xyz = np.stack([new["x"], new["y"], new["z"]], 1).astype(float)
        np.testing.assert_allclose(xyz @ self.M[:3, :3].T + self.M[:3, 3], self.new_world, atol=1e-5)
        self.assertAlmostEqual(float(np.exp(new["scale_0"][0])) * 2.0, 0.001, places=6)  # escala del splat
        self.assertAlmostEqual(float(new["f_dc_0"][1]) * 0.28209479 + 0.5, 1.0, places=5)
        self.assertEqual(new["rot_0"].tolist(), [1.0, 1.0])
        self.assertTrue(np.all(new["f_rest_3"] == 0))

    def test_wrong_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_splat(self.v, self.labels[:-1], self.M, self.aabb, self.map, self.new_world, np.zeros((2, 3)), [7, 8], 0.001)

    def test_ply_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "point_cloud.ply")
            write_ply(path, self.v)
            back = read_ply(path)
        self.assertEqual(back.dtype.names, self.v.dtype.names)
        self.assertTrue(np.array_equal(back["x"], self.v["x"]))


class FakeBullet:
    POSITION_CONTROL = 2

    def __init__(self):
        self.angle = {}
        self.target = None

    def getJointState(self, body, i): return (self.angle.get(i, 0.0),)
    def getJointInfo(self, body, i): return [None] * 9 + [1.878]

    def setJointMotorControl2(self, body, i, mode, targetPosition, force, maxVelocity):
        self.target = (i, targetPosition, force, maxVelocity)
        self.angle[i] = targetPosition

    def resetJointState(self, body, i, value): self.angle[i] = value
    def stepSimulation(self): pass


class TestSimGripper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """gripper.py sin PyBullet ni renderer: se reemplazan sus dos importaciones."""
        pkg = types.ModuleType("xarm_stub")
        pkg.__path__ = []
        config = types.ModuleType("xarm_stub.config")
        config.GRIPPER_RANGE, config.GRIPPER_MAX_ANGLE, config.SPLAT_GRIPPER_POSE = [0, 0.085], 0.85, 0.5
        sys.modules.update({"xarm_stub": pkg, "xarm_stub.config": config})
        render = types.ModuleType("splatsim.utils.robot_splat_render_utils")
        render.get_curr_link_states = lambda robot, centers: [{"pos": i} for i in range(5)]
        cls._saved = sys.modules.get(render.__name__)
        sys.modules[render.__name__] = render
        spec = importlib.util.spec_from_file_location("xarm_stub.gripper", "splatsim/robots/xarm/gripper.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        name = "splatsim.utils.robot_splat_render_utils"
        sys.modules.pop(name, None)
        if cls._saved is not None:
            sys.modules[name] = cls._saved

    def server(self, cfg=None):
        joint = types.SimpleNamespace(name="gripper_drive_joint", id=3, maxForce=0.52)
        s = types.SimpleNamespace(
            pybullet_client=FakeBullet(), dummy_robot=0, use_gripper=True, robot_name="r", joints=[None] * 3 + [joint],
            object_config={"r": cfg or {}}, use_link_centers=True, initial_link_states=[{"pos": -1}] * 5)
        s.__dict__["__parse_joint_info__"] = lambda: None
        s.joints = [types.SimpleNamespace(name="x", id=i, maxForce=0) for i in range(3)] + [joint]
        return s

    def gripper(self, cfg=None):
        s = self.server(cfg)
        g = self.module.GRIPPERS["hand_held_xl330"](s)
        g.setup()
        return s, g

    def test_command_maps_to_travel(self):
        s, g = self.gripper()
        g.command(1.0)
        self.assertEqual(s.pybullet_client.target[0], 3)
        self.assertAlmostEqual(s.pybullet_client.target[1], 1.878)
        self.assertAlmostEqual(g.get_state(), 1.0)
        self.assertAlmostEqual(g.base_state() / 0.8, 1.0)
        g.command(0.5)
        self.assertAlmostEqual(g.get_state(), 0.5)
        self.assertEqual(s.current_gripper_action, 0.5)

    def test_reset_is_open(self):
        s, g = self.gripper()
        g.move(0.0)     # close_gripper() de la base
        self.assertEqual(g.get_state(), 0.0)

    def test_no_splat_patch_for_the_cad_splat(self):
        s, g = self.gripper()
        s.initial_link_states = None
        g.patch_for_splat()     # con None fallaría si intentara parchear

    def test_splat_patch_for_a_scan(self):
        s, g = self.gripper({"splat_gripper_value": 0.5})
        self.assertAlmostEqual(g.splat_pose, 0.939)
        self.assertFalse(g._patched)


class TestCalibrationFile(unittest.TestCase):
    def test_only_the_two_lines_change(self):
        spec = importlib.util.spec_from_file_location("xl330_cli", "scripts/xarm5/xl330_gripper.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with open("configs/xarm5_real.yaml", "rb") as f:
            original = f.read()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.yaml")
            with open(path, "wb") as f:
                f.write(original)
            cli.save_calibration(path, -35, 1189)
            cli.save_calibration(path, 12, 1300)
            with open(path, "rb") as f:
                saved = f.read()
        import yaml
        gripper = yaml.safe_load(saved)["gripper"]
        self.assertEqual((gripper["open_position"], gripper["closed_position"]), (12, 1300))
        self.assertTrue(XL330Gripper.from_config(gripper).calibrated)
        a, b = original.splitlines(), saved.splitlines()
        self.assertEqual(len(a), len(b))
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 2)


if __name__ == "__main__":
    unittest.main()
