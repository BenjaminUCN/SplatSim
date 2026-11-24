"""train_diffusion.py: estado para reanudar el entrenamiento después de un corte."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.xarm5.training import train_diffusion as td  # noqa: E402


def setup(seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    return model, td.EMA(model), opt, sched, GradScaler(enabled=False)


def train(model, ema, opt, sched, n):
    for _ in range(n):
        loss = model(torch.randn(8, 3)).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
        ema.update(model)


class ResumeTest(unittest.TestCase):
    def test_resumed_run_matches_uninterrupted(self):
        straight = setup(0)
        train(*straight[:4], 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / td.RESUME_FILE
            first = setup(0)
            train(*first[:4], 6)
            td.save_resume(path, 6, first[0], first[1], first[2], first[3], first[4])
            self.assertFalse(path.with_suffix(".tmp").exists())
            again = setup(1)                         # otro proceso: pesos distintos hasta cargar
            self.assertEqual(td.load_resume(path, again[0], again[1], again[2], again[3], again[4]), 6)
            train(*again[:4], 4)
        self.assertTrue(torch.allclose(again[0].weight, straight[0].weight))
        self.assertTrue(torch.allclose(again[1].model.weight, straight[1].model.weight))
        self.assertEqual(again[1].step, straight[1].step)
        self.assertAlmostEqual(again[3].get_last_lr()[0], straight[3].get_last_lr()[0])

    def test_no_state_starts_at_zero(self):
        m = setup(0)
        self.assertEqual(td.load_resume(Path("/nonexistent/state.pt"), *m), 0)


if __name__ == "__main__":
    unittest.main()
