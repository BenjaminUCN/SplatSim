"""run_policy_trials.py --leaf_camera wrist_rgb: la pose de pregrasp en que se ajusta la hoja."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import run_policy_trials as rpt  # noqa: E402
from plan_trajectories import grasp_pose  # noqa: E402


class FakeKin:
    """IK que solo alcanza las inclinaciones de `reachable` (grados); anota lo que se le pide."""

    def __init__(self, reachable):
        self.reachable, self.asked = reachable, []

    def ik(self, pos, rot, seed):
        pitch = np.degrees(np.arcsin(rot[2, 2]))      # z de la herramienta = (cos p cos a, cos p sin a, sin p)
        self.asked.append((pos, rot, round(float(pitch), 3)))
        return np.array([pitch, 0, 0, 0, 0]) if any(abs(pitch - r) < 1e-6 for r in self.reachable) else None


class PregraspTest(unittest.TestCase):
    tomato, point = np.array([0.5, 0.1, 0.3]), np.array([0.0, 0.0, 0.15])

    def test_backs_off_along_the_tool_axis(self):
        kin = FakeKin([-27.5])
        q = rpt.pregrasp_q(kin, self.tomato, self.point, [0] * 5, back=0.09, pitch_deg=-27.5)
        self.assertAlmostEqual(q[0], -27.5)
        pos, rot = grasp_pose(self.tomato, np.radians(-27.5), self.point)
        np.testing.assert_allclose(kin.asked[0][0], pos - 0.09 * rot[:, 2])

    def test_falls_back_to_the_closest_pitch(self):
        kin = FakeKin([-50.0, -10.0])
        q = rpt.pregrasp_q(kin, self.tomato, self.point, [0] * 5, back=0.09, pitch_deg=-15.0)
        self.assertAlmostEqual(q[0], -10.0)

    def test_none_when_nothing_reaches(self):
        self.assertIsNone(rpt.pregrasp_q(FakeKin([]), self.tomato, self.point, [0] * 5, 0.09, -27.5))


if __name__ == "__main__":
    unittest.main()
