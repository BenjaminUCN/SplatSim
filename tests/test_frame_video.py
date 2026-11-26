"""Cuadros en video sin pérdida (splatsim/utils/data/frame_video.py). Necesita ffmpeg y opencv.

    python -m unittest discover -s tests -p 'test_frame_video.py'
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from splatsim.utils.data import frame_video


@unittest.skipUnless(shutil.which("ffmpeg"), "falta ffmpeg")
class FrameVideoTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        self.frames = {cam: [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(6)]
                       for cam in ("base_rgb", "wrist_rgb")}
        for cam, frames in self.frames.items():
            for k, a in enumerate(frames):
                Image.fromarray(a).save(self.dir / f"{cam}_{k + 1:05d}.png")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_pack_is_lossless_and_removes_the_pngs(self):
        self.assertEqual(frame_video.pack(self.dir), {"base_rgb": None, "wrist_rgb": None})
        self.assertEqual(list(self.dir.glob("*.png")), [])
        self.assertEqual(frame_video.load_manifest(self.dir), {"base_rgb": list(range(1, 7)),
                                                               "wrist_rgb": list(range(1, 7))})
        cams = frame_video.video_cameras(self.dir)
        for cam, frames in self.frames.items():
            self.assertEqual(len(cams[cam]), 6)
            for k, a in enumerate(frames):
                np.testing.assert_array_equal(np.asarray(frame_video.open_frame(cams[cam][k])), a)

    def test_restore_gives_back_the_same_files(self):
        frame_video.pack(self.dir)
        frame_video.restore(self.dir)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         sorted(f"{c}_{k:05d}.png" for c in self.frames for k in range(1, 7)))
        for cam, frames in self.frames.items():
            for k, a in enumerate(frames):
                np.testing.assert_array_equal(np.asarray(Image.open(self.dir / f"{cam}_{k + 1:05d}.png")), a)

    def test_resume_after_the_manifest_deletes_the_leftover_pngs(self):
        frame_video.pack(self.dir)
        for cam, frames in self.frames.items():          # como si se hubiera cortado antes de borrar
            for k, a in enumerate(frames):
                Image.fromarray(a).save(self.dir / f"{cam}_{k + 1:05d}.png")
        self.assertEqual(frame_video.pack(self.dir), {"base_rgb": None, "wrist_rgb": None})
        self.assertEqual(list(self.dir.glob("*.png")), [])

    def test_gaps_and_alpha_are_left_alone(self):
        (self.dir / "base_rgb_00003.png").unlink()
        Image.fromarray(np.zeros((24, 32, 4), np.uint8)).save(self.dir / "depth_rgb_00001.png")
        result = frame_video.pack(self.dir)
        self.assertEqual(result["base_rgb"], "números no consecutivos")
        self.assertTrue(result["depth_rgb"].startswith("modo"))
        self.assertIsNone(result["wrist_rgb"])
        self.assertEqual(len(list(self.dir.glob("base_rgb_*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
