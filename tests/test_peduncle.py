"""Pedúnculo provisional: la forma del tallo, cuándo se corta y la edición de objects.yaml (sin GPU).

    python -m unittest discover -s tests -p 'test_peduncle.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np

from splatsim.utils.scene.hand_held import MESH_DIR, flange_from_assembly, read_stl
from splatsim.utils.scene.peduncle import (
    DROPPED,
    EMBED,
    HELD,
    INTACT,
    cut_check,
    cylinder_points,
    next_stem_state,
    stem_from_tomato,
    to_world,
)

BOTTOM, TOP = [0, 0, 0.06], [0, 0, 0.11]
UP = [0, 0, 1]


class StemTest(unittest.TestCase):
    def test_grows_from_the_top_of_the_tomato(self):
        rng = np.random.default_rng(0)
        tomato = rng.normal([0.01, -0.02, 0.03], 0.01, (5000, 3))
        stem = stem_from_tomato(tomato, 0.05)
        top = np.percentile(tomato[:, 2], 99)
        self.assertAlmostEqual(stem["bottom"][2], top - EMBED, places=3)
        self.assertAlmostEqual(stem["top"][2], top + 0.05, places=3)
        self.assertAlmostEqual(stem["cut"][2], top + 0.025, places=3)
        np.testing.assert_allclose(stem["top"][:2], [0.01, -0.02], atol=0.002)

    def test_points_lie_on_the_cylinder(self):
        points = cylinder_points(BOTTOM, TOP, 0.0025, 0.001, np.random.default_rng(0))
        self.assertGreater(len(points), 500)
        radial = np.linalg.norm(points[:, :2], axis=1)
        self.assertLessEqual(radial.max(), 0.0025 + 1e-9)
        self.assertAlmostEqual(np.percentile(radial, 50), 0.0025, places=6)
        self.assertGreaterEqual(points[:, 2].min(), 0.06 - 1e-9)
        self.assertLessEqual(points[:, 2].max(), 0.11 + 1e-9)

    def test_stem_turns_with_the_tomato(self):
        half_turn = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(to_world([0.1, 0, 0.2], [1, 2, 3], half_turn), [0.9, 2, 3.2])


class CutTest(unittest.TestCase):
    def test_blade_across_the_stem(self):
        self.assertEqual(cut_check(BOTTOM, TOP, [0.005, 0, 0.085], UP), "ok")

    def test_blade_too_far_to_the_side(self):
        self.assertEqual(cut_check(BOTTOM, TOP, [0.03, 0, 0.085], UP), "too_far")

    def test_blade_plane_above_or_below_the_stem(self):
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.13], UP), "misses_plane")
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.03], UP), "misses_plane")

    def test_blade_along_the_stem_doesnt_cut(self):
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.085], [1, 0, 0]), "tilted")

    def test_upside_down_pivot_axis_is_the_same(self):
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.085], [0, 0, -1]), "ok")

    def test_tilt_limit(self):
        tilted = lambda deg: [np.sin(np.radians(deg)), 0, np.cos(np.radians(deg))]  # noqa: E731
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.085], tilted(40)), "ok")
        self.assertEqual(cut_check(BOTTOM, TOP, [0, 0, 0.085], tilted(50)), "tilted")


class CutAndHoldTest(unittest.TestCase):
    ok = staticmethod(lambda: "ok")
    far = staticmethod(lambda: "too_far")

    def test_closing_on_the_stem_holds_it(self):
        self.assertEqual(next_stem_state(INTACT, 0.8, False, self.ok), HELD)

    def test_closing_away_from_the_stem_doesnt_cut(self):
        self.assertEqual(next_stem_state(INTACT, 0.8, False, self.far), INTACT)

    def test_arriving_already_closed_doesnt_cut(self):
        self.assertEqual(next_stem_state(INTACT, 1.0, True, self.ok), INTACT)

    def test_still_closing_doesnt_cut_yet(self):
        self.assertEqual(next_stem_state(INTACT, 0.5, False, self.ok), INTACT)

    def test_partly_open_keeps_holding(self):
        self.assertEqual(next_stem_state(HELD, 0.6, True, self.far), HELD)

    def test_open_enough_drops_it(self):
        self.assertEqual(next_stem_state(HELD, 0.59, False, self.far), DROPPED)

    def test_dropped_stays_dropped_when_closing_again(self):
        self.assertEqual(next_stem_state(DROPPED, 1.0, False, self.ok), DROPPED)


class MountTest(unittest.TestCase):
    def test_default_mount_puts_the_top_cover_down(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
        from build_hand_held_urdf import DEFAULT_ROLL_DEG
        T = flange_from_assembly(DEFAULT_ROLL_DEG)
        cover = T[:3, :3] @ (read_stl(f"{MESH_DIR}/top_cover.stl").reshape(-1, 3).mean(0) / 1000) + T[:3, 3]
        self.assertLess(cover[0], -0.005)   # +x: arriba con joint5 en 0


class ObjectsYamlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
        import add_peduncle
        cls.module = add_peduncle

    TEXT = "# comment\na:\n  x: 1\n  # keep\nb:\n- 1\n- 2\nc: 3\n"

    def test_replaces_only_that_block(self):
        self.assertEqual(self.module.replace_block(self.TEXT, "b", "b: new\n"),
                         "# comment\na:\n  x: 1\n  # keep\nb: new\nc: 3\n")

    def test_appends_a_new_block(self):
        self.assertEqual(self.module.replace_block("a: 1", "z", "z: 2\n"), "a: 1\nz: 2\n")

    def test_replacing_twice_doesnt_duplicate(self):
        once = self.module.replace_block(self.TEXT, "a", "a: 9\n")
        self.assertEqual(self.module.replace_block(once, "a", "a: 9\n"), once)

    def test_key_with_same_prefix_is_a_different_block(self):
        text = "plastic_apple:\n  x: 1\nplastic_apple_peduncle:\n  y: 2\n"
        self.assertEqual(self.module.replace_block(text, "plastic_apple_peduncle", "plastic_apple_peduncle: 3\n"),
                         "plastic_apple:\n  x: 1\nplastic_apple_peduncle: 3\n")


if __name__ == "__main__":
    unittest.main()
