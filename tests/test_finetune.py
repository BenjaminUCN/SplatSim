"""train_diffusion.py: fine-tuning desde un modelo guardado y mezcla de datasets (sim + real)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import train_diffusion as td  # noqa: E402


class MixWeightsTest(unittest.TestCase):
    def test_each_dataset_gets_its_share(self):
        w = td.mix_sample_weights([300, 20], (0.75, 0.25))
        self.assertAlmostEqual(w[:300].sum().item(), 0.75)
        self.assertAlmostEqual(w[300:].sum().item(), 0.25)
        torch.manual_seed(0)
        picks = torch.multinomial(w, 20000, replacement=True)
        self.assertAlmostEqual((picks >= 300).float().mean().item(), 0.25, delta=0.02)

    def test_empty_means_uniform(self):
        self.assertTrue(torch.equal(td.mix_sample_weights([3, 2], ()), torch.ones(5, dtype=torch.double)))

    def test_wrong_count_or_negative_aborts(self):
        with self.assertRaises(SystemExit):
            td.mix_sample_weights([3, 2], (1.0,))
        with self.assertRaises(SystemExit):
            td.mix_sample_weights([3, 2], (1.0, -0.5))


class CommonKeysTest(unittest.TestCase):
    def test_drops_keys_not_in_every_dataset(self):
        data = [{"action": 1, "observation.state": 2, "extra": 3}]
        self.assertEqual(td.CommonKeys(data, {"action", "observation.state"})[0], {"action": 1, "observation.state": 2})


class InitFromTest(unittest.TestCase):
    def test_weights_round_trip(self):
        torch.manual_seed(0)
        a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            save_file(a.state_dict(), str(Path(tmp) / "model.safetensors"))
            td.load_init_from(b, Path(tmp))
        self.assertTrue(torch.equal(a.weight, b.weight))

    def test_stats_must_match_dataset_root(self):
        saved = {"action.mean": torch.tensor([0.1, 0.2]), "action.std": torch.tensor([1.0, 2.0])}
        with tempfile.TemporaryDirectory() as tmp:
            save_file(saved, str(Path(tmp) / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
            td.check_init_stats(Path(tmp), {"action": {"mean": np.array([0.1, 0.2]), "std": np.array([1.0, 2.0])}})
            with self.assertRaises(SystemExit):
                td.check_init_stats(Path(tmp), {"action": {"mean": np.array([0.1, 0.3]), "std": np.array([1.0, 2.0])}})


if __name__ == "__main__":
    unittest.main()
