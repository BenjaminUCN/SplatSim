"""Máscaras de un objeto por foto de COLMAP: nombres y máscaras reusadas (sin rembg).

    python -m unittest discover -s tests -p 'test_object_masks.py'
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from splatsim.utils.scene.object_masks import MASK_FOLDER, compute_masks, mask_path


def left_half_masker(image: Image.Image) -> Image.Image:
    mask = np.zeros((image.height, image.width), np.uint8)
    mask[:, : image.width // 2] = 255
    return Image.fromarray(mask)


class ComputeMasksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scene = Path(self.tmp.name)
        (self.scene / "images").mkdir()
        photo = np.full((8, 10, 3), 40, np.uint8)
        Image.fromarray(photo).save(self.scene / "images" / "00001.jpg", quality=100)
        Image.fromarray(photo).save(self.scene / "images" / "00002.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_png_mask_per_photo(self):
        self.assertEqual(compute_masks(self.scene, left_half_masker, log=lambda _: None), 2)
        self.assertEqual(sorted(p.name for p in (self.scene / MASK_FOLDER).iterdir()), ["00001.png", "00002.png"])
        self.assertEqual(mask_path(self.scene, "00001.jpg"), self.scene / MASK_FOLDER / "00001.png")
        mask = np.asarray(Image.open(self.scene / MASK_FOLDER / "00002.png"))
        self.assertEqual(mask.shape, (8, 10))
        self.assertTrue(np.all(mask[:, :5] == 255) and np.all(mask[:, 5:] == 0))

    def test_reuse_masks_keeps_hand_edits_and_fills_missing(self):
        compute_masks(self.scene, left_half_masker, log=lambda _: None)
        Image.fromarray(np.full((8, 10), 255, np.uint8)).save(self.scene / MASK_FOLDER / "00002.png")
        (self.scene / MASK_FOLDER / "00001.png").unlink()
        self.assertEqual(compute_masks(self.scene, left_half_masker, reuse_masks=True, log=lambda _: None), 1)
        self.assertTrue(np.all(np.asarray(Image.open(self.scene / MASK_FOLDER / "00002.png")) == 255))

    def test_without_reuse_masks_recomputes_all(self):
        compute_masks(self.scene, left_half_masker, log=lambda _: None)
        self.assertEqual(compute_masks(self.scene, left_half_masker, log=lambda _: None), 2)

    def test_no_images_fails(self):
        with self.assertRaises(SystemExit):
            compute_masks(self.scene / "nada", left_half_masker, log=lambda _: None)


if __name__ == "__main__":
    unittest.main()
