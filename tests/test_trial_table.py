"""Revisión de ensayos en review_recordings.py: trials.csv, éxito, filtros y carpetas anidadas."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

from splatsim.utils.trials import trial_table as tt

FIELDS = ["pos_idx", "trial", "x", "y", "z", "folder", "min_grasp_err", "final_grasp_err", "manual_success",
          "time_to_success", "occlusion_target", "leaf_status", "leaf", "leaf_camera", "occlusion_fit",
          "occlusion_wrist_rgb"]


def write_run(root: Path, rows: list[dict]) -> Path:
    run = root / "model" / "261114_094232"
    for r in rows:
        ep = run / r["folder"]
        ep.mkdir(parents=True)
        (ep / "00001.pkl").write_bytes(b"")
    with open(run / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "folder": str(run / r["folder"])})
    return run


ROWS = [
    {"pos_idx": 0, "trial": 0, "z": -0.02, "folder": "000", "min_grasp_err": 0.09, "occlusion_target": 0.0},
    {"pos_idx": 0, "trial": 0, "z": -0.02, "folder": "001", "min_grasp_err": 0.03, "occlusion_target": 25.0,
     "leaf_status": "ok", "leaf": "hoja_2", "leaf_camera": "wrist_rgb", "occlusion_fit": 25.1, "time_to_success": 7.2},
    {"pos_idx": 12, "trial": 0, "z": 0.34, "folder": "002", "min_grasp_err": 0.02, "occlusion_target": 75.0,
     "occlusion_wrist_rgb": 83.8},
    {"pos_idx": 12, "trial": 0, "z": 0.34, "folder": "003", "min_grasp_err": "", "manual_success": "0",
     "occlusion_target": 0.0},
]


class TrialTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = write_run(self.root, ROWS)
        self.rows = tt.load_trials(self.run)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_by_episode_folder_with_success(self):
        self.assertEqual(sorted(self.rows), ["000", "001", "002", "003"])
        self.assertEqual([self.rows[n]["success"] for n in sorted(self.rows)], [False, True, True, False])

    def test_no_csv_no_trials(self):
        self.assertEqual(tt.load_trials(self.root), {})

    def test_success_threshold_and_manual(self):
        self.assertTrue(tt.is_success({"min_grasp_err": "0.04"}))
        self.assertFalse(tt.is_success({"min_grasp_err": "0.0401"}))
        self.assertTrue(tt.is_success({"min_grasp_err": "0.2", "manual_success": "1"}))
        self.assertIsNone(tt.is_success({"min_grasp_err": ""}))

    def test_options_sorted_as_numbers(self):
        self.assertEqual(tt.options(self.rows, "occlusion_target"), ["all", "0", "25", "75"])
        self.assertEqual(tt.options(self.rows, "z"), ["all", "-0.02", "0.34"])
        self.assertEqual(tt.options(self.rows, "pos_idx"), ["all", "0", "12"])

    def test_filters(self):
        pick = lambda **kw: sorted(n for n, r in self.rows.items() if tt.matches(r, **kw))  # noqa: E731
        self.assertEqual(pick(), ["000", "001", "002", "003"])
        self.assertEqual(pick(result="success"), ["001", "002"])
        self.assertEqual(pick(result="fail", occlusion="0"), ["000", "003"])
        self.assertEqual(pick(z="0.34", result="success"), ["002"])
        self.assertEqual(pick(position="0", occlusion="25"), ["001"])

    def test_episode_without_row_only_unfiltered(self):
        self.assertTrue(tt.matches(None))
        self.assertFalse(tt.matches(None, result="success"))

    def test_labels(self):
        self.assertEqual(tt.short_label(self.rows["001"]), "✓ 3.0 cm · occ 25 % · pos 0")
        text = "\n".join(tt.details(self.rows["001"]))
        self.assertIn("**success**", text)
        self.assertIn("fitted 25.1 %", text)
        self.assertIn("wrist_rgb", text)
        self.assertIn("success at 7.2 s", text)
        self.assertIn("wrist_rgb 83.8 %", "\n".join(tt.details(self.rows["002"])))


class RecordingFoldersTest(unittest.TestCase):
    def test_finds_nested_trial_runs_and_skips_symlinks(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
        from review_recordings import recording_folders
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, ROWS[:1])
            (root / "gello" / "0").mkdir(parents=True)
            (root / "gello" / "0" / "00001.pkl").write_bytes(b"")
            (root / "comparisons" / "x").mkdir(parents=True)
            (root / "comparisons" / "x" / "run").symlink_to(root / "model" / "261114_094232")
            self.assertEqual(recording_folders(root), ["gello", "model/261114_094232"])


if __name__ == "__main__":
    unittest.main()
