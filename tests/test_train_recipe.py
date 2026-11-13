"""Opciones de la receta de Diffusion Policy en train_diffusion.py: EMA y semilla."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import train_diffusion as td  # noqa: E402


class TestEMA(unittest.TestCase):
    def test_decay_follows_diffusion_policy(self):
        ema = td.EMA(torch.nn.Linear(2, 2))
        self.assertEqual(ema.decay(), 0.0)            # paso 0: copia los pesos
        ema.step = 1000
        self.assertAlmostEqual(ema.decay(), 1 - 1001 ** -0.75)
        ema.step = 10**9
        self.assertEqual(ema.decay(), 0.9999)         # recortado al máximo

    def test_update_averages_parameters_and_copies_buffers(self):
        model = torch.nn.BatchNorm1d(1)
        ema = td.EMA(model)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_mean.fill_(5.0)
        ema.update(model)                             # decay 0: igual al modelo
        self.assertEqual(ema.model.weight.item(), 3.0)
        with torch.no_grad():
            model.weight.fill_(7.0)
        ema.update(model)                             # decay 1 - 2^-0.75
        d = 1 - 2 ** -0.75
        self.assertAlmostEqual(ema.model.weight.item(), d * 3.0 + (1 - d) * 7.0, places=5)
        self.assertEqual(ema.model.running_mean.item(), 5.0)
        self.assertFalse(ema.model.weight.requires_grad)

    def test_ema_copy_is_independent(self):
        model = torch.nn.Linear(1, 1)
        ema = td.EMA(model)
        with torch.no_grad():
            model.weight.fill_(1.0)
        self.assertNotEqual(ema.model.weight.data_ptr(), model.weight.data_ptr())


class TestSeed(unittest.TestCase):
    def test_same_seed_same_init(self):
        td.seed_everything(3)
        a = torch.nn.Linear(4, 4).weight.clone()
        td.seed_everything(3)
        b = torch.nn.Linear(4, 4).weight.clone()
        td.seed_everything(4)
        c = torch.nn.Linear(4, 4).weight.clone()
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))


class TestConfig(unittest.TestCase):
    def test_yaml_defaults_keep_the_old_recipe(self):
        cfg = td.load_config(["--config", "configs/train/diffusion.yaml"])
        self.assertEqual(cfg.pretrained_backbone, "ResNet18_Weights.IMAGENET1K_V1")
        self.assertFalse(cfg.group_norm or cfg.lr_scheduler or cfg.ema)
        self.assertIsNone(cfg.seed)

    def test_cli_turns_on_the_recipe(self):
        cfg = td.load_config(["--config", "configs/train/diffusion.yaml", "--pretrained-backbone", "None",
                              "--group-norm", "--lr-scheduler", "--ema", "--seed", "2"])
        self.assertIsNone(cfg.pretrained_backbone)
        self.assertTrue(cfg.group_norm and cfg.lr_scheduler and cfg.ema)
        self.assertEqual(cfg.seed, 2)


if __name__ == "__main__":
    unittest.main()
