"""La foto más nítida de cada grupo de fotos consecutivas (sin video ni GPU).

    python -m unittest discover -s tests -p 'test_sharpness.py'
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from splatsim.utils.scene.sharpness import pick_sharpest, sharpness, weak_windows, write_report


def checkerboard(size: int = 64, square: int = 8) -> Image.Image:
    y, x = np.mgrid[:size, :size]
    return Image.fromarray((((x // square + y // square) % 2) * 255).astype(np.uint8)).convert("RGB")


class SharpnessTest(unittest.TestCase):
    def test_blur_lowers_sharpness(self):
        sharp = checkerboard()
        self.assertGreater(sharpness(sharp), 3 * sharpness(sharp.filter(ImageFilter.GaussianBlur(2))))

    def test_flat_image_is_zero(self):
        self.assertEqual(sharpness(Image.new("RGB", (32, 32), (200, 30, 30))), 0.0)

    def test_wide_images_are_measured_downscaled(self):
        self.assertGreater(sharpness(checkerboard(2600, 40)), 0.0)


class PickTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        # blur por foto: grupos de 3 → (2, 0, 1) (1, 3, 0) (2, 2)
        self.blurs = [2, 0, 1, 1, 3, 0, 2, 2]
        self.photos = []
        for i, radius in enumerate(self.blurs):
            image = checkerboard()
            if radius:
                image = image.filter(ImageFilter.GaussianBlur(radius))
            path = folder / f"{i:05d}.png"
            image.save(path)
            self.photos.append(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sharpest_of_each_window(self):
        frames = pick_sharpest(self.photos, 3)
        self.assertEqual([f.path.name for f in frames if f.kept], ["00001.png", "00005.png", "00006.png"])
        self.assertEqual([f.window for f in frames], [0, 0, 0, 1, 1, 1, 2, 2])

    def test_window_one_keeps_all(self):
        self.assertTrue(all(f.kept for f in pick_sharpest(self.photos, 1)))

    def test_weak_window_is_the_all_blurred_group(self):
        weak = weak_windows(pick_sharpest(self.photos, 3))
        self.assertEqual([f.path.name for f in weak], ["00006.png"])

    def test_report_lists_every_frame(self):
        frames = pick_sharpest(self.photos, 3)
        report = Path(self.tmp.name) / "sharpness.csv"
        write_report(frames, report)
        rows = list(csv.DictReader(open(report)))
        self.assertEqual(len(rows), len(self.blurs))
        self.assertEqual(sum(int(r["kept"]) for r in rows), 3)


if __name__ == "__main__":
    unittest.main()
