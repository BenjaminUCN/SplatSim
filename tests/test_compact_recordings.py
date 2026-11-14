"""compact_recordings.py: saca los tensores de imagen de los .pkl sin perder la imagen."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import compact_recordings as cr  # noqa: E402


class CompactRecordingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ep = Path(self.tmp.name) / "000"
        (self.ep / "image_1").mkdir(parents=True)
        g = torch.Generator().manual_seed(0)
        self.frames = []
        for i in range(1, 4):
            frame = {"action": np.arange(7.0) + i, "base_rgb": torch.rand(3, 8, 10, generator=g),
                     "wrist_rgb": torch.rand(3, 8, 10, generator=g)}
            self.frames.append(frame)
            with open(self.ep / f"{i:05d}.pkl", "wb") as f:
                pickle.dump(frame, f)
            pos = i - 1   # base_rgb del frame en image_1 (como SplatSim); wrist_rgb no está
            Image.fromarray(cr.to_uint8(frame["base_rgb"])).save(self.ep / "image_1" / f"base_rgb_{2 * pos + 1:05d}.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_compacts_and_keeps_every_image(self):
        self.assertEqual(cr.compact_episode(self.ep), (3, 3, 3, []))
        manifest = json.loads((self.ep / cr.MANIFEST).read_text())
        for i, frame in enumerate(self.frames, 1):
            with open(self.ep / f"{i:05d}.pkl", "rb") as f:
                small = pickle.load(f)
            self.assertEqual(set(small), {"action"})
            np.testing.assert_array_equal(small["action"], frame["action"])
            refs = manifest[f"{i:05d}"]
            self.assertTrue(refs["base_rgb"].startswith("image_1/"))
            self.assertEqual(refs["wrist_rgb"], f"pkl_images/wrist_rgb_{i:05d}.png")
            for cam in ("base_rgb", "wrist_rgb"):
                png = np.asarray(Image.open(self.ep / refs[cam]))
                self.assertLessEqual(np.abs(png.astype(int) - cr.to_uint8(frame[cam])).max(), 1)

    def test_rerun_is_a_no_op(self):
        cr.compact_episode(self.ep)
        before = (self.ep / cr.MANIFEST).read_text()
        self.assertEqual(cr.compact_episode(self.ep), (0, 0, 0, []))
        self.assertEqual((self.ep / cr.MANIFEST).read_text(), before)

    def test_broken_pkl_is_left_alone(self):
        (self.ep / "00002.pkl").write_bytes(b"\x00" * 64)
        done, _, _, broken = cr.compact_episode(self.ep)
        self.assertEqual((done, len(broken)), (2, 1))
        self.assertEqual((self.ep / "00002.pkl").read_bytes(), b"\x00" * 64)


if __name__ == "__main__":
    unittest.main()
