"""train_diffusion.py --camera_dropout: cámaras en negro por muestra durante el entrenamiento."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.xarm5.training import train_diffusion as td  # noqa: E402

BASE, WRIST = "observation.images.base_rgb", "observation.images.wrist_rgb"


def batch(n=4000):
    return {BASE: torch.ones(n, 2, 3, 4, 5), WRIST: torch.ones(n, 2, 3, 4, 5), "action": torch.ones(n, 16, 7)}


def dropped(images):
    return images.flatten(1).eq(0).all(1)


class CameraDropoutTest(unittest.TestCase):
    def test_drops_whole_samples_with_probability_p(self):
        g = torch.Generator().manual_seed(0)
        out = td.drop_cameras(batch(), [BASE, WRIST], ("wrist_rgb",), 0.3, g)
        self.assertAlmostEqual(dropped(out[WRIST]).float().mean().item(), 0.3, delta=0.03)
        self.assertFalse(dropped(out[BASE]).any())
        partly = out[WRIST].flatten(1).eq(0).any(1) & ~dropped(out[WRIST])
        self.assertFalse(partly.any())                      # todos los pasos observados o ninguno
        self.assertTrue(torch.equal(out["action"], batch()["action"]))

    def test_never_drops_every_camera(self):
        g = torch.Generator().manual_seed(0)
        out = td.drop_cameras(batch(), [BASE, WRIST], ("base_rgb", "wrist_rgb"), 0.5, g)
        both = dropped(out[BASE]) & dropped(out[WRIST])
        self.assertFalse(both.any())
        self.assertAlmostEqual(dropped(out[WRIST]).float().mean().item(), 0.5, delta=0.03)

    def test_off_returns_batch_unchanged(self):
        b = batch(10)
        self.assertIs(td.drop_cameras(b, [BASE, WRIST], (), 0.3), b)
        self.assertIs(td.drop_cameras(b, [BASE, WRIST], ("wrist_rgb",), 0.0), b)


if __name__ == "__main__":
    unittest.main()
