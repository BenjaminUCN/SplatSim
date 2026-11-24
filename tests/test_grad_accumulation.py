"""train_diffusion.py --grad_accumulation: lotes chicos que suman el batch efectivo."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.xarm5.training import train_diffusion as td  # noqa: E402


class GradAccumulationTest(unittest.TestCase):
    def test_micro_batch(self):
        self.assertEqual(td.micro_batch(SimpleNamespace(batch_size=32, grad_accumulation=2)), 16)
        self.assertEqual(td.micro_batch(SimpleNamespace(batch_size=32, grad_accumulation=1)), 32)
        with self.assertRaises(SystemExit):
            td.micro_batch(SimpleNamespace(batch_size=32, grad_accumulation=3))
        with self.assertRaises(SystemExit):
            td.micro_batch(SimpleNamespace(batch_size=32, grad_accumulation=0))

    def test_grouped_drops_the_incomplete_tail(self):
        self.assertEqual(list(td.grouped(range(7), 2)), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(list(td.grouped(range(3), 1)), [[0], [1], [2]])

    def test_scaled_micro_losses_give_the_full_batch_gradient(self):
        torch.manual_seed(0)
        model, x, y = torch.nn.Linear(4, 1), torch.randn(32, 4), torch.randn(32, 1)
        torch.nn.functional.mse_loss(model(x), y).backward()
        full = model.weight.grad.clone()
        model.zero_grad()
        for xb, yb in zip(x.split(16), y.split(16)):
            (torch.nn.functional.mse_loss(model(xb), yb) / 2).backward()
        self.assertTrue(torch.allclose(model.weight.grad, full, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
