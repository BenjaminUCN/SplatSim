"""Segmentación de hojas y configuración de oclusores, sobre splats sintéticos (sin GPU).

    python -m unittest discover -s tests -p 'test_leaf_occluders.py'
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from splatsim.utils.scene.leaf_segmentation import (
    GrowParams,
    LeafSelection,
    SplatCloud,
    canonical_transformation,
    crop_cloud,
    grow_region,
    load_leaf_info,
    load_occluder_config,
    next_leaf_name,
    occluder_path,
    pick_surface,
    place_between,
    save_leaf,
    save_occluder_config,
)

GREEN = (0.2, 0.6, 0.1)
RED = (0.7, 0.1, 0.1)
FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
          *[f"f_rest_{i}" for i in range(45)], "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def make_cloud(xyz, rgb=GREEN, opacity=0.2, sigma=0.002, T=None) -> SplatCloud:
    """Gaussianas isotrópicas; rgb y opacity por gaussiana o una para todas."""
    xyz = np.asarray(xyz, dtype=float)
    v = np.zeros(len(xyz), dtype=[(f, "f4") for f in FIELDS])
    v["x"], v["y"], v["z"] = xyz.T
    rgb = np.broadcast_to(np.asarray(rgb, dtype=float), xyz.shape)
    for i in range(3):
        v[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / 0.28209479
        v[f"scale_{i}"] = np.log(sigma)
    v["opacity"] = np.log(np.asarray(opacity) / (1 - np.asarray(opacity)))
    v["rot_0"] = 1.0
    return SplatCloud(v, np.eye(4) if T is None else T)


def sheet(center, u, v, half=0.04, step=0.002) -> np.ndarray:
    """Hoja plana: una grilla densa de puntos en el plano u, v."""
    grid = np.arange(-half, half + 1e-9, step)
    a, b = np.meshgrid(grid, grid)
    return np.asarray(center) + a.ravel()[:, None] * np.asarray(u) + b.ravel()[:, None] * np.asarray(v)


X, Y, Z = np.eye(3)


class PickTest(unittest.TestCase):
    def test_hits_the_first_opaque_surface(self):
        front, back = sheet((0, 0, 0.5), X, Y), sheet((0, 0, 0.7), X, Y)
        cloud = make_cloud(np.vstack([front, back]))
        hit = pick_surface(cloud, (0, 0, 0), Z)
        self.assertAlmostEqual(cloud.xyz[hit][2], 0.5, places=3)

    def test_thin_veil_is_seen_through(self):
        veil = make_cloud(sheet((0, 0, 0.3), X, Y, step=0.02), opacity=0.02, sigma=0.002)
        cloud_points = np.vstack([veil.xyz, sheet((0, 0, 0.6), X, Y)])
        opacity = np.r_[np.full(len(veil), 0.02), np.full(len(cloud_points) - len(veil), 0.2)]
        cloud = make_cloud(cloud_points, opacity=opacity)
        self.assertAlmostEqual(cloud.xyz[pick_surface(cloud, (0, 0, 0), Z)][2], 0.6, places=3)

    def test_needle_next_to_the_ray_does_not_block_it(self):
        # una aguja larga en x, a 5 cm del rayo en y: su eje mayor pasa por encima, pero no lo toca
        leaf = sheet((0, 0, 0.6), X, Y)
        cloud = make_cloud(np.vstack([[[0, 0.05, 0.3]], leaf]))
        cloud.vertex["scale_0"][0] = np.log(0.2)
        cloud = SplatCloud(cloud.vertex, np.eye(4))
        self.assertAlmostEqual(cloud.xyz[pick_surface(cloud, (0, 0, 0), Z)][2], 0.6, places=3)

    def test_nothing_on_the_ray(self):
        cloud = make_cloud(sheet((0, 0, 0.5), X, Y))
        self.assertIsNone(pick_surface(cloud, (0, 0, 0), -Z))

    def test_uses_world_coordinates(self):
        T = np.diag([0.5, 0.5, 0.5, 1.0])
        T[:3, 3] = (1.0, 0.0, 0.0)
        cloud = make_cloud(sheet((0, 0, 1.0), X, Y), T=T)   # en mundo: z = 0.5, centrada en x = 1
        hit = pick_surface(cloud, (1.0, 0, 0), Z)
        self.assertAlmostEqual(cloud.xyz[hit][2], 0.5, places=3)


class GrowTest(unittest.TestCase):
    def grow_first(self, *sheets, rgb=GREEN, params=None):
        points = np.vstack(sheets)
        cloud = make_cloud(points, rgb=rgb)
        seed = int(np.argmin(np.sum((points - sheets[0].mean(0)) ** 2, axis=1)))
        return set(grow_region(cloud, seed, params).tolist()), len(sheets[0]), len(points)

    def test_covers_a_whole_flat_leaf(self):
        region, n, _ = self.grow_first(sheet((0, 0, 0), X, Y))
        self.assertEqual(len(region), n)

    def test_covers_a_curved_leaf(self):
        leaf = sheet((0, 0, 0), X, Y)
        leaf[:, 2] = 2.0 * leaf[:, 0] ** 2   # radio de 25 cm
        region, n, _ = self.grow_first(leaf)
        self.assertGreater(len(region), 0.98 * n)

    def test_stops_at_a_parallel_leaf_close_behind(self):
        region, n, _ = self.grow_first(sheet((0, 0, 0), X, Y), sheet((0, 0, 0.004), X, Y))
        self.assertLess(len(region), 1.2 * n)

    def test_stops_at_a_leaf_of_another_angle(self):
        # se tocan en x = 0.04: conectadas por distancia, no por la superficie
        region, n, _ = self.grow_first(sheet((0, 0, 0), X, Y), sheet((0.04, 0, 0.04), Z, Y))
        self.assertLess(len(region), 1.2 * n)

    def test_stops_at_another_color(self):
        a = sheet((0, 0, 0), X, Y)
        b = sheet((0.082, 0, 0), X, Y)  # misma superficie, a continuación
        rgb = np.vstack([np.tile(GREEN, (len(a), 1)), np.tile(RED, (len(b), 1))])
        region, n, _ = self.grow_first(a, b, rgb=rgb)
        self.assertLessEqual(len(region), n)

    def test_radius_limits_the_region(self):
        cloud = make_cloud(sheet((0, 0, 0), X, Y, half=0.1))
        seed = int(np.argmin(np.sum(cloud.xyz ** 2, axis=1)))
        region = grow_region(cloud, seed, GrowParams(max_radius_cm=3.0))
        self.assertLess(np.linalg.norm(cloud.xyz[region], axis=1).max(), 0.03)

    def test_lone_gaussian_returns_itself(self):
        region = grow_region(make_cloud([[0, 0, 0], [1, 1, 1]]), 0)
        self.assertEqual(region.tolist(), [0])


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.cloud = make_cloud(sheet((0, 0, 0), X, Y))
        self.center = int(np.argmin(np.sum(self.cloud.xyz ** 2, axis=1)))
        self.selection = LeafSelection(self.cloud)

    def test_erase_then_undo(self):
        self.selection.grow(self.center, GrowParams())
        full = len(self.selection.indices())
        self.selection.paint(self.center, 1.0, erase=True)
        self.assertLess(len(self.selection.indices()), full)
        self.selection.undo()
        self.assertEqual(len(self.selection.indices()), full)

    def test_regrow_redoes_only_a_grow(self):
        self.selection.grow(self.center, GrowParams())
        full = len(self.selection.indices())
        self.assertTrue(self.selection.regrow(GrowParams(max_radius_cm=2.0)))
        self.assertLess(len(self.selection.indices()), full)
        self.selection.paint(self.center, 0.5)
        self.assertFalse(self.selection.regrow(GrowParams()))


class SaveTest(unittest.TestCase):
    def test_saved_leaf_keeps_its_gaussians_and_says_how_to_reach_the_world(self):
        T = np.diag([0.3, 0.3, 0.3, 1.0])
        T[:3, 3] = (-1.0, 0.2, 0.5)
        cloud = make_cloud(sheet((0, 0, 0), X, Y), T=T)
        indices = np.arange(0, len(cloud), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_leaf(cloud, indices, occluder_path("scene", "hoja_1", tmp), "scene")
            self.assertEqual(next_leaf_name("scene", tmp), "hoja_2")
            again = SplatCloud.from_ply(path, load_leaf_info(path)["transformation"])
        self.assertEqual(len(again), len(indices))
        np.testing.assert_allclose(again.xyz, cloud.xyz[indices], atol=1e-5)
        np.testing.assert_array_equal(again.vertex, cloud.vertex[indices])

    def test_canonical_transformation_centers_the_leaf(self):
        T = np.diag([0.3, 0.3, 0.3, 1.0])
        T[:3, 3] = (-1.0, 0.2, 0.5)
        cloud = make_cloud(sheet((0, 0, 0), X, Y), T=T)
        center = cloud.xyz.mean(axis=0)
        C = canonical_transformation(T, center)
        np.testing.assert_allclose(C[:3, :3], T[:3, :3])
        np.testing.assert_allclose((cloud.vertex["x"].mean(), cloud.vertex["y"].mean(), cloud.vertex["z"].mean())
                                   @ C[:3, :3].T + C[:3, 3], 0.0, atol=1e-5)

    def test_crop_keeps_the_box_and_drops_invisible(self):
        points = np.array([[0, 0, 0], [0.01, 0, 0], [5, 0, 0]])
        cloud = make_cloud(points, opacity=np.array([0.2, 0.001, 0.2]))
        self.assertEqual(len(crop_cloud(cloud, (-1, -1, -1), (1, 1, 1))), 1)

    def test_covariance_is_rotated_to_the_world(self):
        T = np.eye(4)
        T[:3, :3] = 0.5 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        cloud = make_cloud([[0, 0, 0]], T=T)
        cloud.vertex["scale_0"][0] = np.log(0.01)  # larga en x del splat -> y del mundo
        cloud = SplatCloud(cloud.vertex, T)
        cov = cloud.covariances()[0]
        self.assertAlmostEqual(cov[1, 1], (0.01 * 0.5) ** 2, places=9)
        self.assertAlmostEqual(cov[0, 0], (0.002 * 0.5) ** 2, places=9)


class ConfigTest(unittest.TestCase):
    def test_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "occluders.yaml"
            self.assertEqual(load_occluder_config(path), [])
            entries = [{"name": "a", "ply": "x.ply", "position": [1.0, 2.0, 3.0],
                        "wxyz": [1.0, 0.0, 0.0, 0.0], "enabled": False}]
            save_occluder_config(path, entries)
            self.assertEqual(load_occluder_config(path), entries)

    def test_place_between(self):
        np.testing.assert_allclose(place_between((0, 0, 0), (1, 0, 0), 0.25), (0.75, 0, 0))


if __name__ == "__main__":
    unittest.main()
