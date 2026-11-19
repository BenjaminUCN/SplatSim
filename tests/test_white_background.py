"""Fotos sobre fondo blanco para train.py -w: composición, nombres y máscaras reusadas (sin rembg).

    python -m unittest discover -s tests -p 'test_white_background.py'
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from splatsim.utils.scene.white_background import MASK_FOLDER, WHITE_FOLDER, composite_on_white, whiten_folder


def center_masker(image: Image.Image) -> Image.Image:
    """Objeto = mitad izquierda de la foto."""
    mask = np.zeros((image.height, image.width), np.uint8)
    mask[:, : image.width // 2] = 255
    return Image.fromarray(mask)


class CompositeTest(unittest.TestCase):
    def test_object_kept_background_white_edge_blended(self):
        rgb = np.full((1, 3, 3), 100, np.uint8)
        mask = np.array([[255, 0, 128]], np.uint8)
        out = composite_on_white(rgb, mask)
        np.testing.assert_array_equal(out[0, 0], [100, 100, 100])
        np.testing.assert_array_equal(out[0, 1], [255, 255, 255])
        self.assertTrue(np.all((out[0, 2] > 100) & (out[0, 2] < 255)))

    def test_size_mismatch_fails(self):
        with self.assertRaises(ValueError):
            composite_on_white(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2), np.uint8))


class WhitenFolderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scene = Path(self.tmp.name)
        (self.scene / "images").mkdir()
        photo = np.full((8, 10, 3), 40, np.uint8)
        Image.fromarray(photo).save(self.scene / "images" / "00001.jpg", quality=100)
        Image.fromarray(photo).save(self.scene / "images" / "00002.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_names_and_sizes_as_colmap_images(self):
        computed = whiten_folder(self.scene, center_masker, log=lambda _: None)
        self.assertEqual(computed, 2)
        names = sorted(p.name for p in (self.scene / WHITE_FOLDER).iterdir())
        self.assertEqual(names, ["00001.jpg", "00002.png"])
        out = np.asarray(Image.open(self.scene / WHITE_FOLDER / "00002.png"))
        self.assertEqual(out.shape, (8, 10, 3))
        self.assertTrue(np.all(out[:, :5] == 40) and np.all(out[:, 5:] == 255))
        self.assertTrue((self.scene / MASK_FOLDER / "00001.png").is_file())

    def test_reuse_masks_keeps_hand_edits_and_fills_missing(self):
        whiten_folder(self.scene, center_masker, log=lambda _: None)
        Image.fromarray(np.full((8, 10), 255, np.uint8)).save(self.scene / MASK_FOLDER / "00002.png")
        (self.scene / MASK_FOLDER / "00001.png").unlink()
        computed = whiten_folder(self.scene, center_masker, reuse_masks=True, log=lambda _: None)
        self.assertEqual(computed, 1)
        out = np.asarray(Image.open(self.scene / WHITE_FOLDER / "00002.png"))
        self.assertTrue(np.all(out == 40))      # la máscara corregida a mano manda

    def test_without_reuse_masks_recomputes_all(self):
        whiten_folder(self.scene, center_masker, log=lambda _: None)
        Image.fromarray(np.full((8, 10), 255, np.uint8)).save(self.scene / MASK_FOLDER / "00002.png")
        self.assertEqual(whiten_folder(self.scene, center_masker, log=lambda _: None), 2)

    def test_no_images_fails(self):
        with self.assertRaises(SystemExit):
            whiten_folder(self.scene / "nada", center_masker, log=lambda _: None)


if __name__ == "__main__":
    unittest.main()
