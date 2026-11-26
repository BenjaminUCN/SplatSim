"""Recorte del splat de un objeto con sus máscaras, con cámaras sintéticas alrededor de una esfera.

    python -m unittest discover -s tests -p 'test_carve.py'
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from splatsim.utils.scene.carve import View, carve_scene, count_votes, keep_mask, load_mask

SIZE, F = 100, 100.0
RADIUS = 0.3          # el objeto: una esfera en el origen


def look_at_origin(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t (mundo → cámara, convención de COLMAP: z hacia adelante, y hacia abajo)."""
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ position


def ring_views(count: int = 8, distance: float = 2.0) -> list[View]:
    views = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        R, t = look_at_origin(np.array([distance * np.cos(angle), distance * np.sin(angle), 0.5]))
        views.append(View(f"{i:05d}.jpg", R, t, F, F, SIZE / 2, SIZE / 2, SIZE, SIZE))
    return views


def silhouette(view: View) -> np.ndarray:
    """Máscara de la esfera: los píxeles cuyo rayo pasa a menos de RADIUS del origen."""
    center = -view.R.T @ view.t
    v, u = np.mgrid[:SIZE, :SIZE] + 0.5
    rays = np.stack([(u - view.cx) / view.fx, (v - view.cy) / view.fy, np.ones_like(u)], -1) @ view.R
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    closest = np.linalg.norm(np.cross(rays, -center), axis=-1)
    return closest < RADIUS


POINTS = np.array([
    [0.0, 0.0, 0.0],      # centro del objeto
    [0.25, 0.0, 0.0],     # dentro, cerca del borde
    [0.0, 0.0, 1.0],      # sobre el objeto, fuera de la silueta desde los lados
    [1.0, 1.0, 0.0],      # fondo
])


class VotesTest(unittest.TestCase):
    def setUp(self):
        self.views = ring_views()
        self.masks = [silhouette(v) for v in self.views]

    def test_object_inside_background_outside(self):
        seen, outside = count_votes(POINTS, self.views, self.masks)
        self.assertTrue(np.all(seen[:2] == len(self.views)))
        self.assertTrue(np.all(outside[:2] == 0))
        self.assertEqual(seen[2], 0)                       # sobre el objeto: fuera del cuadro en todas
        self.assertGreater(outside[3], 0.5 * seen[3])      # fondo: fuera de la silueta en la mayoría
        self.assertEqual(keep_mask(seen, outside, 0.2, 3).tolist(), [True, True, False, False])

    def test_one_bad_mask_is_tolerated(self):
        masks = [np.zeros_like(self.masks[0])] + self.masks[1:]     # una foto perdió el objeto
        seen, outside = count_votes(POINTS, self.views, masks)
        self.assertEqual(outside[0], 1)
        self.assertTrue(keep_mask(seen, outside, 0.2, 3)[0])

    def test_behind_camera_and_out_of_frame_do_not_vote(self):
        far = np.array([[50.0, 0.0, 0.0]])          # detrás de la cámara 0 y fuera del cuadro del resto
        seen, outside = count_votes(far, self.views[:1], self.masks[:1])
        self.assertEqual((seen[0], outside[0]), (0, 0))
        self.assertFalse(keep_mask(seen, outside, 0.2, 3)[0])

    def test_mask_of_another_size_is_scaled(self):
        small = [np.asarray(Image.fromarray(m).resize((50, 50))) for m in self.masks]
        seen, outside = count_votes(POINTS, self.views, small)
        self.assertEqual(keep_mask(seen, outside, 0.2, 3).tolist(), [True, True, False, False])


class CarveSceneTest(unittest.TestCase):
    def test_writes_only_the_object(self):
        views = ring_views()
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "masks").mkdir()
            for view in views:
                Image.fromarray((silhouette(view) * 255).astype(np.uint8)).save(scene / "masks" / f"{view.name[:-4]}.png")
            vertex = np.zeros(len(POINTS), [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
            vertex["x"], vertex["y"], vertex["z"] = POINTS.T
            vertex["opacity"] = np.arange(len(POINTS))
            ply = scene / "point_cloud.ply"
            PlyData([PlyElement.describe(vertex, "vertex")]).write(str(ply))

            import splatsim.utils.scene.carve as carve
            original = carve.read_views
            carve.read_views = lambda _: views
            try:
                kept, total = carve_scene(scene, ply, scene / "carved.ply", dilate_px=1, log=lambda _: None)
            finally:
                carve.read_views = original
            out = PlyData.read(str(scene / "carved.ply"))["vertex"].data
            self.assertEqual((kept, total), (2, 4))
            self.assertEqual(out["opacity"].tolist(), [0.0, 1.0])     # todas las propiedades, las mismas filas

    def test_dilation_grows_the_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            mask = np.zeros((20, 20), np.uint8)
            mask[10, 10] = 255
            Image.fromarray(mask).save(path)
            self.assertEqual(load_mask(path, 0).sum(), 1)
            self.assertEqual(load_mask(path, 2).sum(), 25)


if __name__ == "__main__":
    unittest.main()
