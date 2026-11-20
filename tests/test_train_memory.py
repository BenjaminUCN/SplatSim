import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import train_diffusion as td  # noqa: E402

CAMS = [f"observation.images.{c}" for c in ("left", "right", "wrist")]


class FakeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        torch.manual_seed(index)
        item = {k: torch.rand(2, 3, 8, 8) for k in CAMS}
        item[CAMS[0]][0, 0, 0, 0] = 1.3        # el aumento de datos puede salirse de [0, 1]
        item[CAMS[0]][0, 0, 0, 1] = -0.2
        item |= {f"{k}_is_pad": torch.zeros(2, dtype=torch.bool) for k in CAMS}
        item["observation.state"] = torch.rand(2, 7)
        return item


class TestUint8Images(unittest.TestCase):
    def test_quantizes_used_cameras_and_drops_the_rest(self):
        item = td.Uint8Images(FakeDataset(), CAMS[:2])[1]
        self.assertEqual(item[CAMS[0]].dtype, torch.uint8)
        self.assertNotIn(CAMS[2], item)
        self.assertNotIn(f"{CAMS[2]}_is_pad", item)
        self.assertIn(f"{CAMS[0]}_is_pad", item)
        self.assertEqual(item["observation.state"].dtype, torch.float32)

    def test_round_trip_within_half_a_level_and_clipped(self):
        original = FakeDataset()[1][CAMS[0]]
        item = td.Uint8Images(FakeDataset(), CAMS)[1]
        back = td.images_to_float({CAMS[0]: item[CAMS[0]]}, CAMS[:1], torch.device("cpu"))[CAMS[0]]
        self.assertEqual(back.dtype, torch.float32)
        self.assertLessEqual((back - original.clamp(0, 1)).abs().max().item(), 0.5 / 255 + 1e-6)
        self.assertEqual(back[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(back[0, 0, 0, 1].item(), 0.0)

    def test_images_to_float_leaves_the_loader_batch_untouched(self):
        batch = {CAMS[0]: torch.zeros(1, 2, 3, 4, 4, dtype=torch.uint8), "observation.state": torch.ones(1)}
        out = td.images_to_float(batch, CAMS[:1], torch.device("cpu"))
        self.assertEqual(batch[CAMS[0]].dtype, torch.uint8)
        self.assertIs(out["observation.state"], batch["observation.state"])


class TestReleaseStepInputs(unittest.TestCase):
    def test_clears_steps_and_their_sub_steps(self):
        class Step:
            _current_transition = None

        inner, outer = Step(), Step()
        outer.sub = inner
        inner._current_transition = outer._current_transition = {"x": 1}
        td.release_step_inputs(type("Pipeline", (), {"steps": [outer]})())
        self.assertIsNone(outer._current_transition)
        self.assertIsNone(inner._current_transition)


class TestLoaderMemory(unittest.TestCase):
    frames = {k: (2, 3 * 480 * 640) for k in CAMS}     # 3 cámaras 480x640, n_obs_steps 2

    def test_three_cameras_float32_with_old_defaults_was_over_the_limit(self):
        need = td.loader_memory_bytes(self.frames, 32, 12, 3, bytes_per_value=4)
        self.assertAlmostEqual(need / 1e9, 26.9, delta=0.1)
        self.assertGreater(need, td.MAX_LOADER_RAM_FRACTION * 56e9)

    def test_three_cameras_uint8_with_new_defaults_is_small(self):
        need = td.loader_memory_bytes(self.frames, 32, 12, 1)
        self.assertAlmostEqual(need / 1e9, 2.5, delta=0.1)

    def test_no_workers_keeps_one_batch(self):
        self.assertEqual(td.loader_memory_bytes({"a": (1, 10)}, 4, 0, 1), 40)


if __name__ == "__main__":
    unittest.main()
