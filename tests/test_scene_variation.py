"""Variación por episodio (cámaras base y giro del tomate) y focal por escena.

    python -m unittest discover -s tests -p 'test_scene_variation.py'
"""
import math
import types
import unittest

import numpy as np
import pybullet as p

from splatsim.utils.scene.scene_variation import VariationRanges, draw_variation, turn_about_own_z

CAMERAS = ["base_rgb", "base_left_rgb", "base_right_rgb", "wrist_rgb"]


class DrawTest(unittest.TestCase):
    def test_same_key_same_draw(self):
        a = draw_variation(7, CAMERAS, VariationRanges(), seed=3)
        self.assertEqual(a, draw_variation(7, list(reversed(CAMERAS)), VariationRanges(), seed=3))
        self.assertNotEqual(a, draw_variation(8, CAMERAS, VariationRanges(), seed=3))

    def test_within_ranges_and_no_wrist(self):
        ranges = VariationRanges(camera_xyz_m=0.01, camera_rpy_deg=1.0, tomato_yaw_deg=30.0)
        for key in range(50):
            v = draw_variation(key, CAMERAS, ranges)
            self.assertEqual(sorted(v["camera_jitter"]), ["base_left_rgb", "base_rgb", "base_right_rgb"])
            jitter = np.array(list(v["camera_jitter"].values()))
            self.assertLessEqual(np.abs(jitter[:, :3]).max(), 0.01)
            self.assertLessEqual(np.degrees(np.abs(jitter[:, 3:]).max()), 1.0 + 1e-6)
            self.assertLessEqual(abs(math.degrees(v["tomato_yaw"])), 30.0 + 1e-6)

    def test_scaled_keeps_tomato(self):
        r = VariationRanges().scaled(2)
        self.assertEqual((r.camera_xyz_m, r.camera_rpy_deg, r.tomato_yaw_deg), (0.02, 2.0, 180.0))


class TurnTest(unittest.TestCase):
    def test_matches_pybullet(self):
        tilt = p.getQuaternionFromEuler((0.3, -0.2, 0.5))
        got = turn_about_own_z(tilt, 0.7)
        want = p.multiplyTransforms((0, 0, 0), tilt, (0, 0, 0), p.getQuaternionFromEuler((0, 0, 0.7)))[1]
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_stem_axis_stays(self):
        # el eje z del objeto (el tallo) no cambia con el giro
        tilt = p.getQuaternionFromEuler((0.2, 0.1, -0.4))
        z = lambda q: np.reshape(p.getMatrixFromQuaternion(q), (3, 3))[:, 2]  # noqa: E731
        np.testing.assert_allclose(z(turn_about_own_z(tilt, 2.0)), z(tilt), atol=1e-12)


class CameraTest(unittest.TestCase):
    def setUp(self):
        from splatsim.robots.xarm import cameras
        self.cameras = cameras
        self.server = types.SimpleNamespace(robot_name="scene", object_config={"scene": {}}, pybullet_client=p)

    def test_fov_default_and_focal(self):
        default = (1.0, 0.5)
        self.assertEqual(self.cameras.scene_fov(self.server, default), default)
        self.server.object_config["scene"]["camera_focal_px"] = 615.0
        fovx, fovy = self.cameras.scene_fov(self.server, default)
        self.assertAlmostEqual(math.degrees(fovx), 54.98, places=1)
        self.assertAlmostEqual(math.degrees(fovy), 42.64, places=1)
        # píxeles cuadrados: la misma focal en x y en y
        self.assertAlmostEqual(320 / math.tan(fovx / 2), 240 / math.tan(fovy / 2), places=6)

    def test_16_9_matches_base_fov(self):
        from splatsim.robots.xarm.config import BASE_CAM_FOV
        self.assertEqual(self.cameras.scene_image_size(self.server), (640, 480))
        self.server.object_config["scene"].update(camera_image_size=[360, 640], camera_focal_px=462.0)
        self.assertEqual(self.cameras.scene_image_size(self.server), (640, 360))
        # 16:9 a 462 px: lo mismo que ven hoy las cámaras base (FOV_16_9), sin estirar
        for got, want in zip(self.cameras.scene_fov(self.server, None), BASE_CAM_FOV):
            self.assertAlmostEqual(math.degrees(got), math.degrees(want), delta=0.1)

    def test_jitter_moves_pose(self):
        cam = self.cameras.BaseCamera(self.server, "base_rgb", (0.1, 0.2, 0.3), (-1.5, 0.0, -4.7))
        nominal = cam.pose()
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(cam.rpy)), (3, 3))
        np.testing.assert_allclose(nominal[:3, :3], R, atol=1e-12)
        cam.jitter = (0.01, 0.0, -0.01, 0.0, 0.0, math.radians(1))
        moved = cam.pose()
        np.testing.assert_allclose(moved[:3, 3], (0.11, 0.2, 0.29))
        # 1° alrededor del eje z propio de la cámara (el que mira hacia la escena)
        angle = math.degrees(math.acos((np.trace(nominal[:3, :3].T @ moved[:3, :3]) - 1) / 2))
        self.assertAlmostEqual(angle, 1.0, places=6)
        np.testing.assert_allclose(moved[:3, 2], nominal[:3, 2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
