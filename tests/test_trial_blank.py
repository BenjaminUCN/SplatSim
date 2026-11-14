"""--blank_cameras de run_policy_trials.py: la política recibe esas cámaras en negro, nada más cambia."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import run_policy_trials as rpt  # noqa: E402
from test_trial_stop import FakeAgent, trial  # noqa: E402


class TestBlank(unittest.TestCase):
    def test_blanks_only_the_given_cameras_without_touching_obs(self):
        obs = {"base_rgb": torch.ones(3, 4, 4), "wrist_rgb": np.ones((4, 4, 3), np.uint8), "state": np.ones(7)}
        out = rpt.blank(obs, ("base_rgb", "wrist_rgb"))
        self.assertEqual(out["base_rgb"].abs().sum().item(), 0)
        self.assertIsInstance(out["base_rgb"], torch.Tensor)
        self.assertEqual(out["wrist_rgb"].sum(), 0)
        self.assertEqual(out["wrist_rgb"].dtype, np.uint8)
        self.assertTrue(np.array_equal(out["state"], obs["state"]))
        self.assertEqual(obs["base_rgb"].sum().item(), 48)          # el original no cambia

    def test_no_cameras_returns_the_same_obs(self):
        obs = {"base_rgb": np.ones(3)}
        self.assertIs(rpt.blank(obs, ()), obs)

    def test_trial_feeds_blank_images_to_the_policy(self):
        seen = []

        class Agent(FakeAgent):
            def act(self, obs):
                seen.append(obs["base_rgb"].sum())
                return np.zeros(6)

        from unittest import mock
        import test_trial_stop as tts
        orig = tts.FakeRobot.get_observations

        def with_image(self):
            return {**orig(self), "base_rgb": np.ones((2, 2, 3))}
        with mock.patch.object(tts.FakeRobot, "get_observations", with_image), \
                mock.patch.object(tts, "FakeAgent", Agent):
            trial(lambda t: 0.3, blank_cameras=("base_rgb",))
        self.assertTrue(seen and all(v == 0 for v in seen))


if __name__ == "__main__":
    unittest.main()
