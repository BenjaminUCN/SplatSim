"""Filtros de carpetas de review_recordings.py: fecha, tarea, hoja, muestreador y texto de cada corrida."""
import csv
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from splatsim.utils.trials import run_index as ri

FIELDS = ["pos_idx", "trial", "folder", "min_grasp_err", "noise_scheduler", "num_inference_steps",
          "n_action_steps", "occlusion_target", "peduncle_held", "gripper_threshold"]


def write_run(root: Path, model: str, stamp: str, rows: list[dict], meta: dict | None = None) -> str:
    run = root / model / stamp
    for i, r in enumerate(rows):
        ep = run / f"{i:03d}"
        ep.mkdir(parents=True)
        (ep / "00001.pkl").write_bytes(b"")
        if meta is not None:
            (ep / "meta.json").write_text(json.dumps(meta))
    with open(run / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"pos_idx": i % 2, "trial": 0, **r, "folder": str(run / f"{i:03d}")})
    return f"{model}/{stamp}"


class RunIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        ddim = {"noise_scheduler": "DDIM", "num_inference_steps": 16, "n_action_steps": 8}
        self.grasp = write_run(root, "noocc_cam_a4j_s2_30k_1117", "261118_052012", [
            {**ddim, "min_grasp_err": 0.02, "occlusion_target": 0.0},
            {**ddim, "min_grasp_err": 0.09, "occlusion_target": 0.0}],
            meta={"checkpoint": "/m/cam_a4j_s2_30k_1117", "ik_orientation": "off", "blank_cameras": []})
        self.leaf = write_run(root, "occlt_cam_a4_s1_30k_1114", "261115_204816", [
            {"noise_scheduler": "DDPM", "num_inference_steps": 100, "min_grasp_err": 0.03, "occlusion_target": 0.0},
            {"noise_scheduler": "DDPM", "num_inference_steps": 100, "min_grasp_err": 0.08, "occlusion_target": 50.0}],
            meta={"checkpoint": "/m/cam_a4_s1_30k_1114", "ik_orientation": "hybrid", "blank_cameras": ["wrist_rgb"]})
        self.cut = write_run(root, "gt_noocc_cut_a4j_s2_30k_1118", "261118_113107", [
            {**ddim, "min_grasp_err": 0.01, "peduncle_held": 0, "gripper_threshold": 0.5},
            {**ddim, "min_grasp_err": 0.2, "peduncle_held": 1, "gripper_threshold": 0.5}])
        (root / "gello" / "0").mkdir(parents=True)
        os.utime(root / "gello", (datetime(2026, 10, 1).timestamp(),) * 2)   # sin AAMMDD_HHMMSS: su mtime
        self.infos = {n: ri.describe(root, n) for n in (self.grasp, self.leaf, self.cut, "gello")}

    def tearDown(self):
        self.tmp.cleanup()

    def pick(self, **kw):
        return sorted(n for n, i in self.infos.items() if ri.matches(i, **kw))

    def test_describe_run(self):
        g = self.infos[self.grasp]
        self.assertEqual((g.when, g.trials, g.successes, g.task, g.leaf, g.sampler),
                         (datetime(2026, 11, 18, 5, 20, 12), 2, 1, "grasp", "no leaf", "DDIM-16"))
        self.assertEqual(g.params["model"], "cam_a4j_s2_30k_1117")
        self.assertNotIn("IK", g.params)
        leaf = self.infos[self.leaf]
        self.assertEqual((leaf.leaf, leaf.sampler, leaf.params["IK"], leaf.params["blanked"]),
                         ("leaf", "DDPM-100", "hybrid", "wrist_rgb"))
        cut = self.infos[self.cut]
        self.assertEqual((cut.task, cut.successes, cut.params["gripper threshold"]), ("cut", 1, "0.5"))
        self.assertFalse(self.infos["gello"].is_run)
        self.assertEqual(self.infos["gello"].date, "2026-10-01")
        self.assertEqual(self.infos["gello"].task, "recordings")

    def test_filters(self):
        self.assertEqual(self.pick(date="2026-11-18"), sorted([self.grasp, self.cut]))
        self.assertEqual(self.pick(task="cut"), [self.cut])
        self.assertEqual(self.pick(task="recordings"), ["gello"])
        self.assertEqual(self.pick(leaf="leaf"), [self.leaf])
        self.assertEqual(self.pick(sampler="DDIM-16"), sorted([self.grasp, self.cut]))
        self.assertEqual(self.pick(search="A4J s2"), sorted([self.grasp, self.cut]))
        self.assertEqual(self.pick(search="a4j -cut"), [self.grasp])
        self.assertEqual(self.pick(search="hybrid"), [self.leaf])   # también busca en los parámetros

    def test_options_and_order(self):
        infos = [self.infos[n] for n in (self.grasp, self.leaf, self.cut)]
        self.assertEqual(ri.date_options(infos), ["all", "2026-11-18", "2026-11-15"])
        self.assertEqual(ri.sampler_options(infos), ["all", "DDIM-16", "DDPM-100"])
        self.assertEqual([i.path for i in ri.ordered(infos)], [self.cut, self.grasp, self.leaf])
        self.assertEqual([i.path for i in ri.ordered(infos, "oldest first")], [self.leaf, self.grasp, self.cut])
        self.assertEqual([i.path for i in ri.ordered(infos, "name")], [self.cut, self.grasp, self.leaf])

    def test_labels(self):
        self.assertEqual(ri.label(self.infos[self.cut]), "11-18 11:31 · gt_noocc_cut_a4j_s2_30k_1118 · 1/2 cuts")
        self.assertEqual(ri.label(self.infos[self.grasp], short=False),
                         f"11-18 05:20 · {self.grasp} · 1/2 ✓")
        text = ri.details(self.infos[self.leaf])
        self.assertIn("1/2 successes (50 %)", text)
        self.assertIn("sampler `DDPM-100`", text)


if __name__ == "__main__":
    unittest.main()
