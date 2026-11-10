"""Geometría de la cámara base y reglas del experimento de percepción (sin GPU; PyBullet opcional).

    python -m unittest tests/test_perception_samples.py
"""
import importlib.util
import unittest

import numpy as np

from splatsim.utils.render.camera_view import CameraView, mask_stats, rotation_from_rpy
from splatsim.utils.scene.perception_design import Thresholds, check_arm, check_tomato, farthest_points

FOV = (1.2112585, 0.7417649)           # BASE_CAM_FOV de splatsim/robots/xarm/config.py
BASE_RGB = CameraView((-0.079, 0.0, 0.142), (-1.562, 0.0, -4.708), FOV, 640, 480)


class CameraViewTest(unittest.TestCase):
    def test_base_camera_axes(self):
        """base_rgb mira hacia la planta (-x), con la derecha de la imagen en +y y abajo en -z."""
        rot = BASE_RGB.rotation
        np.testing.assert_allclose(rot[:, 2], [-1, 0, 0], atol=0.02)
        np.testing.assert_allclose(rot[:, 0], [0, 1, 0], atol=0.02)
        np.testing.assert_allclose(rot[:, 1], [0, 0, -1], atol=0.02)

    def test_rotation_matches_scipy(self):
        from scipy.spatial.transform import Rotation
        rpy = (0.3, -0.7, 2.1)
        np.testing.assert_allclose(rotation_from_rpy(rpy), Rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-12)

    def test_projection_center_and_edges(self):
        rot, eye = BASE_RGB.rotation, np.array(BASE_RGB.xyz)
        ahead = eye + 0.8 * rot[:, 2]
        uv, depth = BASE_RGB.project([ahead])
        np.testing.assert_allclose(uv[0], [320, 240], atol=1e-6)
        self.assertAlmostEqual(depth[0], 0.8)
        # en el borde del FOV horizontal: u = ancho
        edge = eye + rot[:, 2] + np.tan(FOV[0] / 2) * rot[:, 0]
        self.assertAlmostEqual(BASE_RGB.project([edge])[0][0, 0], 640, places=6)
        edge_v = eye + rot[:, 2] - np.tan(FOV[1] / 2) * rot[:, 1]
        self.assertAlmostEqual(BASE_RGB.project([edge_v])[0][0, 1], 0, places=6)

    def test_in_frame(self):
        rot, eye = BASE_RGB.rotation, np.array(BASE_RGB.xyz)
        points = [eye + rot[:, 2],                     # adelante
                  eye - rot[:, 2],                     # detrás
                  eye + rot[:, 2] + 2 * rot[:, 0],     # muy a la derecha
                  eye + rot[:, 2] + 0.5 * rot[:, 0]]   # a la derecha, dentro (tan(34.7°)=0.69)
        np.testing.assert_array_equal(BASE_RGB.in_frame(points), [True, False, False, True])
        self.assertFalse(BASE_RGB.in_frame([points[3]], margin=200)[0])

    def test_uv_fraction(self):
        rot, eye = BASE_RGB.rotation, np.array(BASE_RGB.xyz)
        np.testing.assert_allclose(BASE_RGB.uv_fraction(eye + rot[:, 2]), [0.5, 0.5])
        self.assertIsNone(BASE_RGB.uv_fraction(eye - rot[:, 2]))
        # el efector en la pose de reset (gello_3plants) queda por encima de la imagen
        self.assertLess(BASE_RGB.uv_fraction([-0.338, 0.001, 0.508])[1], 0)

    def test_mask_stats(self):
        mask = np.zeros((10, 20), bool)
        self.assertEqual(mask_stats(mask)["pixels"], 0)
        mask[0:2, 0:4] = True
        st = mask_stats(mask)
        self.assertEqual(st["pixels"], 8)
        self.assertEqual(st["bbox"], [0.0, 0.0, 0.2, 0.2])
        self.assertAlmostEqual(st["centroid"][0], 1.5 / 20)


@unittest.skipUnless(importlib.util.find_spec("pybullet"), "sin pybullet")
class PybulletMaskTest(unittest.TestCase):
    def test_segmentation_matches_projection(self):
        """Una esfera vista con las matrices de PyBullet cae donde dice project()."""
        import pybullet as p
        client = p.connect(p.DIRECT)
        try:
            rot, eye = BASE_RGB.rotation, np.array(BASE_RGB.xyz)
            for offset in ([0.0, 0.0], [0.25, -0.15], [-0.3, 0.12]):
                center = eye + 0.9 * rot[:, 2] + offset[0] * rot[:, 0] + offset[1] * rot[:, 1]
                shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, physicsClientId=client)
                body = p.createMultiBody(baseVisualShapeIndex=shape, basePosition=center.tolist(),
                                         physicsClientId=client)
                seg = BASE_RGB.segmentation(_Client(client))
                st = mask_stats(seg == body)
                uv = BASE_RGB.project([center])[0][0] / [640, 480]
                np.testing.assert_allclose(st["centroid"], uv, atol=0.01)
                p.removeBody(body, physicsClientId=client)
        finally:
            p.disconnect(client)


class _Client:
    """Lo que segmentation() usa del pybullet_client del servidor, con un physicsClientId propio."""

    def __init__(self, cid):
        self.cid = cid

    def getCameraImage(self, *args, **kwargs):
        import pybullet as p
        return p.getCameraImage(*args, physicsClientId=self.cid, **kwargs)


class RulesTest(unittest.TestCase):
    t = Thresholds()

    def arm(self, fraction=0.15, ee_uv=(0.5, 0.4), distance=0.3):
        return {"arm_fraction": fraction, "ee_uv": None if ee_uv is None else list(ee_uv), "grasp_distance": distance}

    def test_out_of_view(self):
        self.assertEqual(check_arm("A1", self.arm(fraction=0.0), self.t), [])
        self.assertTrue(check_arm("A2", self.arm(fraction=0.001), self.t))

    def test_base(self):
        """A3: la pose de reset de las demos, con el brazo por encima de la imagen y centrado."""
        self.assertEqual(check_arm("A3", self.arm(fraction=0.0, ee_uv=(0.5, -0.8)), self.t), [])
        self.assertTrue(check_arm("A3", self.arm(fraction=0.1, ee_uv=(0.5, -0.8)), self.t))   # se ve
        self.assertTrue(check_arm("A3", self.arm(fraction=0.0, ee_uv=(0.5, 1.4)), self.t))    # por debajo
        self.assertTrue(check_arm("A3", self.arm(fraction=0.0, ee_uv=(1.2, -0.8)), self.t))   # de lado
        self.assertTrue(check_arm("A3", self.arm(fraction=0.0, ee_uv=None), self.t))          # detrás

    def test_far_near(self):
        """Distancias de las poses iniciales de gello_3plants: cerca 0.22–0.42 m, lejos 0.34–0.61 m."""
        self.assertEqual(check_arm("A4", self.arm(distance=0.45), self.t), [])
        self.assertTrue(check_arm("A4", self.arm(distance=0.25), self.t))
        self.assertEqual(check_arm("A5", self.arm(distance=0.25), self.t), [])
        self.assertTrue(check_arm("A5", self.arm(distance=0.5), self.t))
        self.assertTrue(check_arm("A5", self.arm(fraction=0.01, distance=0.25), self.t))
        self.assertTrue(check_arm("A4", self.arm(fraction=0.6, distance=0.45), self.t))   # pegado a la cámara

    def test_tomato(self):
        seen = {"occlusion": 10.0, "cut_by_frame": False}
        hidden = {"occlusion": 95.0, "cut_by_frame": False}
        out = {"occlusion": None, "cut_by_frame": False}
        for b in ("B1", "B4"):
            self.assertEqual(check_tomato(b, seen, self.t), [])
            self.assertTrue(check_tomato(b, hidden, self.t))
            self.assertTrue(check_tomato(b, out, self.t))
            self.assertTrue(check_tomato(b, {**seen, "cut_by_frame": True}, self.t))
        self.assertEqual(check_tomato("B2", hidden, self.t), [])
        self.assertEqual(check_tomato("B2", out, self.t), [])
        self.assertTrue(check_tomato("B2", seen, self.t))
        self.assertEqual(check_tomato("B3", out, self.t), [])
        self.assertTrue(check_tomato("B3", seen, self.t))


class FarthestPointsTest(unittest.TestCase):
    def test_spreads_and_orders(self):
        rng = np.random.default_rng(0)
        cluster = rng.normal(0, 0.01, (50, 3))
        points = np.vstack([cluster, [[1, 0, 0]], [[0, 1, 0]]])
        order = farthest_points(points, len(points), rng)
        self.assertEqual(sorted(order), list(range(len(points))))       # todos, sin repetir
        self.assertEqual(set(order[:3]) & {50, 51}, {50, 51})            # los aislados, primero

    def test_taken(self):
        points = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0]], float)
        self.assertEqual(farthest_points(points, 1, taken=[[0, 0, 0]]), [2])
        self.assertEqual(farthest_points(points, 5, taken=[[1, 0, 0]])[:1], [0])


if __name__ == "__main__":
    unittest.main()
