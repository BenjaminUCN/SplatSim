"""Porcentaje de oclusión con un rasterizador falso que mezcla alfa de verdad (sin GPU).

    python -m unittest discover -s tests -p 'test_occlusion.py'
"""
import sys
import types
import unittest
from pathlib import Path

import torch

from splatsim.utils.occlusion import (
    LEVELS,
    OUT_OF_VIEW,
    fit_lateral_offset,
    measure_camera,
    occlusion_level,
)

SIZE = 40
C0 = 0.28209479177387814


class FakeGaussians:
    """Cada gaussiana ocupa un rectángulo de píxeles a una profundidad, con una opacidad."""

    def __init__(self, boxes, depths, opacity, rgb):
        self.boxes = boxes
        n = len(boxes)
        self._xyz = torch.tensor(depths, dtype=torch.float32)[:, None].repeat(1, 3)
        self._rotation = torch.arange(n)[:, None]    # índice a boxes, para que un corte lo conserve
        self._opacity = torch.tensor(opacity, dtype=torch.float32)
        self._scaling = torch.zeros(n, 3)
        self._features_dc = ((torch.tensor(rgb, dtype=torch.float32) - 0.5) / C0)[:, None]
        self._features_rest = torch.full((n, 15, 3), 0.7)   # brillo que debe ignorarse al pintar


def fake_render(camera, g, background):
    """Mezcla alfa de atrás hacia adelante."""
    image = background.reshape(3, 1, 1).repeat(1, SIZE, SIZE).clone()
    for i in torch.argsort(g._xyz[:, 0], descending=True).tolist():
        y0, y1, x0, x1 = g.boxes[g._rotation[i, 0].item()]
        alpha = g._opacity[i]
        color = (g._features_dc[i, 0] * C0 + 0.5).reshape(3, 1, 1)
        color = color + g._features_rest[i, 0].reshape(3, 1, 1) * 0.1  # así se notaría el brillo
        image[:, y0:y1, x0:x1] = alpha * color + (1 - alpha) * image[:, y0:y1, x0:x1]
    return image


TOMATO = (10, 30, 10, 30)
GREEN = (0.2, 0.6, 0.1)
RED = (0.8, 0.1, 0.1)


def measure(boxes, depths, opacity, tomato=TOMATO, tomato_depth=1.0):
    """Tomate opaco (rgb rojo) más otras gaussianas verdes."""
    g = FakeGaussians([*boxes, tomato], [*depths, tomato_depth], [*opacity, 1.0],
                      [GREEN] * len(boxes) + [RED])
    target = slice(len(boxes), len(boxes) + 1)
    before = g._features_dc.clone(), g._features_rest.clone()
    result = measure_camera(fake_render, g, target, None, torch.ones(3))
    torch.testing.assert_close(g._features_dc, before[0])
    torch.testing.assert_close(g._features_rest, before[1])
    return result


class MeasureTest(unittest.TestCase):
    def test_nothing_in_front(self):
        self.assertEqual(measure([], [], [])["occlusion"], 0.0)

    def test_opaque_half_in_front(self):
        self.assertAlmostEqual(measure([(10, 20, 10, 30)], [0.5], [1.0])["occlusion"], 50.0, places=1)

    def test_translucent_in_front_covers_by_its_alpha(self):
        self.assertAlmostEqual(measure([TOMATO], [0.5], [0.3])["occlusion"], 30.0, places=1)

    def test_translucent_layers_add_up(self):
        self.assertAlmostEqual(measure([TOMATO, TOMATO], [0.4, 0.5], [0.5, 0.5])["occlusion"], 75.0, places=1)

    def test_behind_does_not_cover(self):
        self.assertEqual(measure([TOMATO], [2.0], [1.0])["occlusion"], 0.0)

    def test_white_scene_does_not_look_visible(self):
        # delante hay una hoja blanca sobre fondo blanco: el color no debe confundir la medida
        g = FakeGaussians([TOMATO, TOMATO], [0.5, 1.0], [1.0, 1.0], [(1, 1, 1), RED])
        result = measure_camera(fake_render, g, slice(1, 2), None, torch.ones(3))
        self.assertEqual(result["occlusion"], 100.0)

    def test_out_of_view(self):
        result = measure([], [], [], tomato=(0, 0, 0, 0))
        self.assertIsNone(result["occlusion"])
        self.assertEqual(result["level"], OUT_OF_VIEW)

    def test_cut_by_frame(self):
        self.assertFalse(measure([], [], [])["cut_by_frame"])
        self.assertTrue(measure([], [], [], tomato=(0, 20, 10, 30))["cut_by_frame"])


class LevelTest(unittest.TestCase):
    def test_edges(self):
        self.assertEqual([occlusion_level(v) for v in (0, 24.9, 25, 50, 75, 100)],
                         [LEVELS[0], LEVELS[0], LEVELS[1], LEVELS[2], LEVELS[3], LEVELS[3]])


class FitTest(unittest.TestCase):
    @staticmethod
    def leaf(top=80.0, floor=5.0, width=0.1):
        """Tapa top centrada y baja en línea recta hasta floor al alejarse width."""
        return lambda offset: max(floor, top - (top - floor) * offset / width)

    def test_reaches_the_target(self):
        fit = fit_lateral_offset(self.leaf(), 40.0)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["occlusion"], 40.0, delta=1.0)

    def test_leaf_too_small(self):
        self.assertEqual(fit_lateral_offset(self.leaf(top=30.0), 60.0)["status"], "too_small")

    def test_already_covered(self):
        self.assertEqual(fit_lateral_offset(self.leaf(floor=50.0), 20.0)["status"], "already_covered")

    def test_out_of_view(self):
        self.assertEqual(fit_lateral_offset(lambda _: None, 20.0)["status"], OUT_OF_VIEW)


class SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # measure_occlusion.py importa render_recordings: el servidor solo se importa al usarlo
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
        import measure_occlusion
        cls.module = measure_occlusion

    def test_summary_starts_at_the_first_frame_in_view(self):
        s = self.module.episode_summary([None, 30.0, 50.0, 100.0])
        self.assertEqual((s["start"], s["mean"], s["end"], s["level"]), (30.0, 60.0, 100.0, "25-50"))

    def test_summary_warns_when_the_start_was_cut_by_the_frame(self):
        s = self.module.episode_summary([None, 100.0, 40.0], [False, True, False])
        self.assertTrue(s["start_cut"])
        self.assertFalse(self.module.episode_summary([40.0, 100.0], [False, True])["start_cut"])

    def test_summary_never_in_view(self):
        self.assertEqual(self.module.episode_summary([None, None])["level"], OUT_OF_VIEW)

    def test_spaced_frames_include_both_ends_without_repeats(self):
        self.assertEqual(self.module.spaced_frames(list(range(100)), 3), [0, 50, 99])
        self.assertEqual(self.module.spaced_frames(list(range(2)), 8), [0, 1])
        self.assertEqual(self.module.spaced_frames(list(range(5)), 1), [0])


if __name__ == "__main__":
    unittest.main()
