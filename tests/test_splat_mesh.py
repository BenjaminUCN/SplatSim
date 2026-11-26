"""Malla de un objeto a partir de su splat (splatsim/utils/scene/splat_mesh.py).

    python -m unittest discover -s tests -p test_splat_mesh.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatsim.utils.scene.splat_mesh import (PALETTE_LEVELS, face_uv, palette_texture, splat_mesh,  # noqa: E402
                                             write_object_urdf)


def sphere_shell(n=4000, radius=0.035, seed=0):
    """Gaussianas en la superficie de una esfera, como el splat de un fruto: rojas arriba, verdes abajo."""
    d = np.random.default_rng(seed).normal(size=(n, 3))
    xyz = radius * d / np.linalg.norm(d, axis=1, keepdims=True)
    rgb = np.where(xyz[:, 2:] > 0, [0.8, 0.1, 0.1], [0.1, 0.6, 0.1])
    return xyz, np.full(n, 0.5), rgb


class SplatMeshTest(unittest.TestCase):
    def test_sphere(self):
        xyz, opacity, rgb = sphere_shell()
        v, f, c = splat_mesh(xyz, opacity, rgb, [-0.04] * 3, [0.04] * 3, faces=4000)
        self.assertLessEqual(len(f), 4000)
        r = np.linalg.norm(v, axis=1)
        # la superficie (no una bola rellena): un poco afuera de la cáscara, porque el suavizado (sigma) engorda una
        # cáscara densa; en tomate_tab, más rala, la malla queda a ~1 mm de las gaussianas
        self.assertGreater(r.min(), 0.03)
        self.assertLess(np.median(r) - 0.035, 0.007)
        top, bottom = v[:, 2] > 0.01, v[:, 2] < -0.01
        self.assertGreater(c[top, 0].mean(), 0.6)
        self.assertGreater(c[bottom, 1].mean(), 0.45)

    def test_palette(self):
        """El uv de un color cae en un píxel de la paleta con ese color (a 1/(niveles-1) de distancia)."""
        colors = np.random.default_rng(1).random((200, 3))
        uv = face_uv(colors)
        tex = palette_texture()
        side = tex.shape[0]
        px = tex[((1 - uv[:, 1]) * side).astype(int), (uv[:, 0] * side).astype(int)] / 255
        self.assertLessEqual(np.abs(px - colors).max(), 0.5 / (PALETTE_LEVELS - 1) + 1e-6)

    def test_urdf(self):
        """El URDF conserva masa y colisión del template y carga en PyBullet con la malla nueva como visual."""
        import pybullet as p
        xyz, opacity, rgb = sphere_shell()
        mesh = splat_mesh(xyz, opacity, rgb, [-0.04] * 3, [0.04] * 3, faces=2000)
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "template" / "model.urdf"
            template.parent.mkdir()
            (template.parent / "col.obj").write_text("v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n")
            template.write_text(
                '<robot name="t"><link name="base"><inertial><mass value="0.1"/>'
                '<inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>'
                '<visual><geometry><mesh filename="textured.obj"/></geometry></visual>'
                '<collision><geometry><mesh filename="col.obj"/></geometry></collision></link></robot>')
            urdf = write_object_urdf(Path(tmp) / "out", template, *mesh)
            for name in ("visual.obj", "visual.mtl", "visual.png", "visual_colors.ply", "collision.obj"):
                self.assertTrue((urdf.parent / name).exists(), name)
            client = p.connect(p.DIRECT)
            try:
                body = p.loadURDF(str(urdf), physicsClientId=client)
                self.assertAlmostEqual(p.getDynamicsInfo(body, -1, physicsClientId=client)[0], 0.1)
                visual = p.getVisualShapeData(body, physicsClientId=client)
                self.assertEqual([os.path.basename(v[4].decode()) for v in visual], ["visual.obj"])
            finally:
                p.disconnect(client)


if __name__ == "__main__":
    unittest.main()
