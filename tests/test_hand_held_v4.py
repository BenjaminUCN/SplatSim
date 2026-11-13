"""Gripper hand_held_XL330 v4: cuatro barras y URDF (sin PyBullet ni GPU).

    PYTHONPATH=tests:. python -m unittest test_hand_held_v4
"""
import math
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from splatsim.utils.scene.hand_held import JAW_JOINT, build_urdf, load_design, load_manifest
from splatsim.utils.scene.hand_held_linkage import ROD_JOINT, Linkage
from test_hand_held_xl330 import FIXTURE, fk

V4_DIR = "splatsim/robot_definitions/meshes/hand_held_xl330_v4"


class TestLinkage(unittest.TestCase):
    def setUp(self):
        self.linkage = Linkage.from_manifest(load_design(V4_DIR)["linkage"])

    def test_matches_the_cad(self):
        # construir_hand_held_XL330_v4.py: horn a 269.8° cerrada y 183.9° abierta, bastón de 62.72 mm
        self.assertAlmostEqual(math.degrees(self.linkage.horn_angle(0.0)) % 360, 269.8, delta=0.05)
        self.assertAlmostEqual(math.degrees(self.linkage.horn_angle(1.0)) % 360, 183.9, delta=0.05)
        for t in np.linspace(0, 1, 11):
            gap = np.linalg.norm(self.linkage.horn_pin(t) - self.linkage.yoke_pin(t))
            self.assertAlmostEqual(gap, self.linkage.rod_mm, places=6)

    def test_joints_are_zero_open_and_invert(self):
        np.testing.assert_allclose(self.linkage.joints_at(1.0), (0, 0, 0), atol=1e-12)
        drive, jaw, rod = self.linkage.joints_at(0.0)
        self.assertAlmostEqual(drive, self.linkage.travel_rad)
        self.assertAlmostEqual(jaw, self.linkage.jaw_travel_rad)
        for t in (0.1, 0.4, 0.8):
            d, j, r = self.linkage.joints_at(t)
            np.testing.assert_allclose(self.linkage.jaw_and_rod(d), (j, r), atol=1e-4)


class TestUrdfV4(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.src = os.path.join(self.dir, "xarm5.urdf")
        self.dst = os.path.join(self.dir, "out.urdf")
        with open(self.src, "w") as f:
            f.write(FIXTURE)
        self.linkage = Linkage.from_manifest(load_design(V4_DIR)["linkage"])
        build_urdf(self.src, self.dst, self.linkage.travel_rad, mesh_dir=V4_DIR)
        self.root = ET.parse(self.dst).getroot()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_meshes_go_to_their_own_folder(self):
        for part in load_manifest(V4_DIR):
            self.assertTrue(os.path.exists(os.path.join(self.dir, "hand_held_xl330_v4", part["file"])))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "hand_held_xl330")))

    def test_rod_hangs_from_the_horn_at_its_pin(self):
        joints = {j.get("name"): j for j in self.root.findall("joint")}
        rod = joints[ROD_JOINT]
        self.assertEqual((rod.find("parent").get("link"), rod.find("child").get("link")),
                         ("hand_held_horn", "hand_held_rod"))
        horn = np.array([float(v) for v in joints["gripper_drive_joint"].find("origin").get("xyz").split()])
        pin = np.array([float(v) for v in rod.find("origin").get("xyz").split()])
        np.testing.assert_allclose((horn + pin)[:2] * 1000, self.linkage.horn_pin(1.0), atol=1e-3)
        self.assertAlmostEqual(float(joints["gripper_drive_joint"].find("limit").get("upper")),
                               self.linkage.travel_rad, places=4)
        self.assertIn(JAW_JOINT, joints)

    def test_gripper_center_is_the_v4_one(self):
        rel = np.linalg.inv(fk(self.dst, "hand_held_base")) @ fk(self.dst, "gripper_center_link")
        np.testing.assert_allclose(rel[:3, 3] * 1000, load_design(V4_DIR)["gripper_center_mm"], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
