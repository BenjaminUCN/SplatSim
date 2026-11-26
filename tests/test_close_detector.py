"""Detector de cierre: la regla que dispara y cómo la usa LeRobotDiffusionAgent."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from splatsim.agents import close_detector as cd
from splatsim.agents.lerobot_diffusion_agent import LeRobotDiffusionAgent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.xarm5.training import close_detector as script  # noqa: E402


class FakeDetector:
    def __init__(self, fires):
        self.fires = list(fires)
        self.calls = 0

    def fire(self, image):
        self.calls += 1
        return self.fires.pop(0)


def agent_with(detector):
    agent = object.__new__(LeRobotDiffusionAgent)
    agent.close_detector = detector
    agent.detector_fires = 0
    agent._latched = False
    agent._hold_joints = None
    return agent


def obs(q, closed=0.0):
    return {"joint_positions": np.array([q] * 5 + [closed]), "wrist_rgb": np.zeros((4, 4, 3), np.uint8)}


class DetectorCommandTest(unittest.TestCase):
    def test_fire_closes_where_the_arm_is(self):
        agent = agent_with(FakeDetector([True]))
        joints, gripper = agent._detector_command(obs(0.3), np.full(5, 0.9), 0.0)
        self.assertEqual(gripper, 1.0)
        np.testing.assert_allclose(joints, np.full(5, 0.3))
        self.assertEqual(agent.detector_fires, 1)

    def test_no_fire_keeps_the_policy(self):
        agent = agent_with(FakeDetector([False]))
        joints, gripper = agent._detector_command(obs(0.3), np.full(5, 0.9), 0.7)
        self.assertEqual(gripper, 0.7)
        np.testing.assert_allclose(joints, np.full(5, 0.9))

    def test_not_asked_once_closing_or_closed(self):
        detector = FakeDetector([True])
        agent = agent_with(detector)
        agent._detector_command(obs(0.3, closed=0.5), np.zeros(5), 0.0)
        agent._latched = True
        agent._detector_command(obs(0.3), np.zeros(5), 0.0)
        self.assertEqual(detector.calls, 0)


class RuleTest(unittest.TestCase):
    def test_first_fire_needs_consecutive_frames(self):
        p = np.array([0.95, 0.2, 0.95, 0.96, 0.99])
        self.assertEqual(script.first_fire(p, 0.9, 1), 0)
        self.assertEqual(script.first_fire(p, 0.9, 2), 3)
        self.assertIsNone(script.first_fire(p, 0.999, 1))

    def test_image_formats(self):
        a = np.random.randint(0, 255, (6, 8, 3), np.uint8)
        np.testing.assert_array_equal(cd.as_uint8_hwc(a), a)
        np.testing.assert_array_equal(cd.as_uint8_hwc(a.transpose(2, 0, 1)), a)
        np.testing.assert_array_equal(cd.as_uint8_hwc(torch.tensor(a.transpose(2, 0, 1)) / 255.0), a)


class SavedDetectorTest(unittest.TestCase):
    def test_round_trip_and_consecutive(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            model = cd.make_model(pretrained=False)
            torch.nn.init.zeros_(model.fc.weight)
            torch.nn.init.constant_(model.fc.bias, 3.0)       # p = 0.95 siempre
            torch.save(model.state_dict(), Path(tmp) / "detector.pt")
            (Path(tmp) / "detector.json").write_text('{"size": [96, 128]}')
            det = cd.CloseDetector(tmp, threshold=0.9, consecutive=2, device="cpu")
            image = np.zeros((480, 640, 3), np.uint8)
            self.assertAlmostEqual(det.prob(image), 0.9526, places=3)
            self.assertFalse(det.fire(image))
            self.assertTrue(det.fire(image))
            det.reset()
            self.assertFalse(det.fire(image))


if __name__ == "__main__":
    unittest.main()
