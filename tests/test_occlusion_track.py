"""run_policy_trials.py --track_occlusion: resumen de la oclusión medida durante el ensayo."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import run_policy_trials as rpt  # noqa: E402


class TrackSummaryTest(unittest.TestCase):
    samples = [(0.0, None), (0.5, 20.0), (1.0, 60.0), (1.5, None), (2.0, 10.0)]

    def test_mean_max_and_in_view(self):
        s = rpt.track_summary("wrist_rgb", self.samples, None)
        self.assertEqual(s["track_wrist_rgb_mean"], 30.0)
        self.assertEqual(s["track_wrist_rgb_max"], 60.0)
        self.assertEqual(s["track_wrist_rgb_pre_success"], 30.0)   # sin éxito: todo el ensayo
        self.assertEqual(s["track_wrist_rgb_in_view"], 0.6)
        self.assertEqual(list(s), rpt.track_columns("wrist_rgb"))

    def test_pre_success_stops_at_success(self):
        self.assertEqual(rpt.track_summary("wrist_rgb", self.samples, 1.0)["track_wrist_rgb_pre_success"], 40.0)

    def test_never_in_view(self):
        s = rpt.track_summary("wrist_rgb", [(0.0, None)], 1.0)
        self.assertEqual((s["track_wrist_rgb_mean"], s["track_wrist_rgb_max"], s["track_wrist_rgb_in_view"]),
                         (None, None, 0.0))


if __name__ == "__main__":
    unittest.main()
