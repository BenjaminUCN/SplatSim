"""render_recordings.py --leaf_reference pregrasp: el frame en que se ajusta la hoja (sin GPU).

    python -m unittest discover -s tests -p 'test_leaf_reference.py'
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
from render_recordings import pregrasp_frame  # noqa: E402

PHASES = {"phases": {"start_pause": 0.3, "reach": 4.7, "settle": 0.3, "approach": 3.4, "hold": 3.4}}


class PregraspFrameTest(unittest.TestCase):
    def test_end_of_reach_and_settle(self):
        times = 100.0 + np.arange(0, 12, 1 / 30)       # 30 Hz, sim_time no empieza en 0
        k = pregrasp_frame(PHASES, times)
        self.assertAlmostEqual(times[k] - times[0], 5.3, delta=1 / 60)

    def test_needs_the_planner_phases(self):
        with self.assertRaises(ValueError):
            pregrasp_frame({}, np.arange(10.0))
        with self.assertRaises(ValueError):
            pregrasp_frame({"phases": {"reach": 1.0}}, np.arange(10.0))


if __name__ == "__main__":
    unittest.main()
