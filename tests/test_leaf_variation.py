"""Hoja por episodio: el sorteo, la geometría y las decisiones, con un servidor falso (sin GPU).

    python -m unittest discover -s tests -p 'test_leaf_variation.py'
"""
import types
import unittest
from pathlib import Path

import numpy as np

from splatsim.utils.scene.leaf_variation import (
    FRACTIONS,
    EpisodeDraw,
    draw_episode,
    fit_with_fractions,
    lateral_direction,
    place_episode_leaf,
    roll_quaternion,
)

LEAVES = [Path("a.ply"), Path("b.ply"), Path("c.ply")]
TARGETS = (0.0, 25.0, 50.0, 75.0)


class DrawTest(unittest.TestCase):
    def test_same_episode_same_draw(self):
        self.assertEqual(draw_episode("007", LEAVES, TARGETS, 1), draw_episode("007", LEAVES, TARGETS, 1))

    def test_seed_changes_the_draw(self):
        draws = {draw_episode("007", LEAVES, TARGETS, s).direction_deg for s in range(5)}
        self.assertEqual(len(draws), 5)

    def test_every_target_and_leaf_comes_up(self):
        draws = [draw_episode(f"{i:03d}", LEAVES, TARGETS) for i in range(200)]
        self.assertEqual({d.target for d in draws}, set(TARGETS))
        self.assertEqual({d.leaf for d in draws}, set(LEAVES))


class GeometryTest(unittest.TestCase):
    def test_lateral_is_unit_and_perpendicular(self):
        axis = np.array([1.0, 2.0, 3.0])
        for degrees in (0, 77, 200):
            v = lateral_direction(axis, [0.3, 1.0, -0.2], degrees)
            self.assertAlmostEqual(np.linalg.norm(v), 1.0)
            self.assertAlmostEqual(np.dot(v, axis), 0.0)

    def test_lateral_opposite_at_180(self):
        axis, right = [0, 0, 1.0], [1.0, 0, 0]
        np.testing.assert_allclose(lateral_direction(axis, right, 180), [-1, 0, 0], atol=1e-9)

    def test_roll_quaternion(self):
        np.testing.assert_allclose(roll_quaternion([0, 0, 2.0], 0), [1, 0, 0, 0])
        np.testing.assert_allclose(roll_quaternion([0, 0, 2.0], 180), [0, 0, 0, 1], atol=1e-9)


class FractionsTest(unittest.TestCase):
    def test_first_distance_is_enough(self):
        fit = fit_with_fractions(lambda f, o: max(0.0, 60.0 - 600 * o), 30.0)
        self.assertEqual((fit["status"], fit["fraction"]), ("ok", FRACTIONS[0]))

    def test_closer_to_the_camera_when_too_small(self):
        fit = fit_with_fractions(lambda f, o: max(0.0, 100 * f - 600 * o), 35.0)
        self.assertEqual((fit["status"], fit["fraction"]), ("ok", FRACTIONS[1]))

    def test_keeps_the_closest_when_none_reaches(self):
        fit = fit_with_fractions(lambda f, o: max(0.0, 100 * f - 600 * o), 90.0)
        self.assertEqual((fit["status"], fit["fraction"]), ("too_small", FRACTIONS[-1]))
        self.assertAlmostEqual(fit["occlusion"], 55.0)

    def test_out_of_view(self):
        self.assertIsNone(fit_with_fractions(lambda f, o: None, 30.0)["occlusion"])

    def test_moves_away_from_the_arm(self):
        fit = fit_with_fractions(lambda f, o: max(0.0, 60.0 - 600 * o), 30.0,
                                 clearance=lambda f, o: 0.02 if f == FRACTIONS[0] else 0.2)
        self.assertEqual((fit["status"], fit["fraction"], fit["arm_distance"]), ("ok", FRACTIONS[1], 0.2))

    def test_clear_of_the_arm_beats_reaching_the_target(self):
        fit = fit_with_fractions(lambda f, o: max(0.0, 100 * f - 600 * o), 40.0,
                                 fractions=(0.3, 0.5),
                                 clearance=lambda f, o: 0.2 if f == 0.3 else 0.02)
        self.assertEqual((fit["status"], fit["fraction"]), ("too_small", 0.3))

    def test_farthest_from_the_arm_when_all_cross(self):
        distances = {FRACTIONS[0]: 0.02, FRACTIONS[1]: 0.08, FRACTIONS[2]: 0.05}
        fit = fit_with_fractions(lambda f, o: max(0.0, 60.0 - 600 * o), 30.0, clearance=lambda f, o: distances[f])
        self.assertEqual(fit["fraction"], FRACTIONS[1])


def result(occlusion, cut=False):
    return {"occlusion": occlusion, "cut_by_frame": cut}


class FakeOccluders:
    def __init__(self):
        self.items = {}

    def set_pose(self, name, position, wxyz):
        self.items[name].position = np.asarray(position)

    def clear(self):
        self.items = {}


class FakeServer:
    """Tomate en el origen. Una hoja tapa top% si está sobre la línea y menos al correrse."""

    def __init__(self, seen, top=60.0):
        self.seen, self.top = seen, top
        self.occluders = FakeOccluders()
        self.apple = types.SimpleNamespace(body_id=0)
        self.cameras = {c: types.SimpleNamespace(xyz=(1.0, 0.0, 1.0), rpy=(0, 0, 0)) for c in seen}
        self.pybullet_client = types.SimpleNamespace(
            getBasePositionAndOrientation=lambda body: ((0.0, 0.0, 0.0), (0, 0, 0, 1)),
            getQuaternionFromEuler=lambda rpy: (0, 0, 0, 1),
            getMatrixFromQuaternion=lambda q: (0, 1, 0, 1, 0, 0, 0, 0, 1))

    def set_episode_occluders(self, entries):
        self.occluders.clear()
        for e in entries:
            self.occluders.items[e["name"]] = types.SimpleNamespace(
                position=np.asarray(e["position"]),
                entry=lambda n=e["name"]: {"name": n, "position": self.occluders.items[n].position.tolist()})
        return [e["name"] for e in entries]

    def measure_occlusion(self, cameras):
        out = {}
        for c in cameras:
            base = self.seen[c]
            if base["occlusion"] is None or not self.occluders.items:
                out[c] = base
                continue
            leaf = next(iter(self.occluders.items.values()))
            line = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
            miss = np.linalg.norm(leaf.position - np.dot(leaf.position, line) * line)
            out[c] = result(max(base["occlusion"], self.top - 600 * miss))
        return out


def draw(target=30.0, pick=0.0):
    return EpisodeDraw(target=target, leaf=Path("/x/hoja.ply"), camera_pick=pick, direction_deg=40.0, roll_deg=10.0)


class PlaceTest(unittest.TestCase):
    def test_reaches_the_target_and_leaves_the_leaf_there(self):
        s = FakeServer({"cam": result(0.0)})
        entries, info = place_episode_leaf(s, draw(30.0), ["cam"])
        self.assertEqual(info["status"], "ok")
        self.assertAlmostEqual(info["occlusion"], 30.0, delta=1.0)
        self.assertEqual(entries[0]["name"], "hoja")
        self.assertAlmostEqual(s.measure_occlusion(["cam"])["cam"]["occlusion"], info["occlusion"], places=6)

    def test_target_zero_has_no_leaf(self):
        s = FakeServer({"cam": result(0.0)})
        entries, info = place_episode_leaf(s, draw(0.0), ["cam"])
        self.assertEqual((entries, info["status"]), ([], "no_leaf"))

    def test_scene_already_covers_the_target(self):
        s = FakeServer({"cam": result(40.0)})
        entries, info = place_episode_leaf(s, draw(30.0), ["cam"])
        self.assertEqual((entries, info["status"], info["occlusion"]), ([], "already_covered", 40.0))
        self.assertEqual(s.occluders.items, {})

    def test_skips_cameras_where_the_tomato_is_cut_or_missing(self):
        s = FakeServer({"cut": result(0.0, cut=True), "none": result(None), "good": result(0.0)})
        _, info = place_episode_leaf(s, draw(30.0, pick=0.0), ["cut", "none", "good"])
        self.assertEqual(info["camera"], "good")

    def test_no_camera_sees_the_whole_tomato(self):
        s = FakeServer({"cut": result(0.0, cut=True)})
        entries, info = place_episode_leaf(s, draw(30.0), ["cut"])
        self.assertEqual((entries, info["status"]), ([], "no_camera"))

    def test_leaf_stays_away_from_the_arm_path(self):
        s = FakeServer({"cam": result(0.0)})
        near_first_try = np.array([[0.25, 0.0, 0.25]])    # el brazo pasa por el primer punto de la línea
        entries, info = place_episode_leaf(s, draw(30.0), ["cam"], near_first_try)
        self.assertEqual(info["status"], "ok")
        self.assertGreater(info["fraction"], FRACTIONS[0])
        self.assertGreaterEqual(info["arm_distance"], 0.1)
        self.assertEqual(len(entries), 1)

    def test_leaf_too_small_keeps_the_best_try(self):
        s = FakeServer({"cam": result(0.0)}, top=20.0)
        entries, info = place_episode_leaf(s, draw(60.0), ["cam"])
        self.assertEqual((info["status"], info["fraction"]), ("too_small", FRACTIONS[0]))
        self.assertEqual(len(entries), 1)


if __name__ == "__main__":
    unittest.main()
