"""Marco del cuerpo de un tomate real: un fruto rojo con su tallo verde, girado y escalado al azar.

    python -m unittest discover -s tests -p 'test_real_object.py'
"""
import unittest

import numpy as np

from splatsim.utils.scene.real_object import SH_C0, fit_tomato

RED, GREEN = (0.8, 0.15, 0.1), (0.3, 0.5, 0.2)
R_FRUIT = 0.4          # unidades de COLMAP
STEM = 0.3             # largo del tallo recto sobre el fruto
BRANCH = 0.3           # la rama, inclinada hacia +x del fruto


def vertices(points: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(len(points), [(k, "f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")])
    out["x"], out["y"], out["z"] = points.T
    dc = (rgb - 0.5) / SH_C0
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = dc.T
    out["opacity"] = 3.0
    return out


def random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def tomato(rng, gap: float = 0.0):
    """Puntos en el marco del fruto (centro en el origen, tallo en +z), y en uno arbitrario de COLMAP."""
    sphere = rng.normal(size=(3000, 3))
    sphere = R_FRUIT * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    t = rng.uniform(R_FRUIT + gap, R_FRUIT + STEM, 400)
    stem = np.c_[rng.normal(0, 0.01, (400, 2)), t]
    s = rng.uniform(0, BRANCH, 300)
    branch = np.c_[s * 0.8, rng.normal(0, 0.01, 300), R_FRUIT + STEM + s * 0.6]
    floaters = rng.uniform(5, 8, (50, 3))
    local = np.concatenate([sphere, stem, branch, floaters])
    rgb = np.array([RED] * len(sphere) + [GREEN] * (len(stem) + len(branch)) + [RED] * len(floaters))
    R, offset, scale = random_rotation(rng), rng.normal(size=3), 1.7
    return local, vertices(scale * local @ R.T + offset, rgb), len(local) - len(floaters)


class FitTomatoTest(unittest.TestCase):
    def check(self, gap: float = 0.0):
        rng = np.random.default_rng(1)
        local, v, n_object = tomato(rng, gap)
        fit = fit_tomato(v, diameter=0.07)
        self.assertEqual(fit.keep.sum(), n_object)                 # sin los restos sueltos
        self.assertFalse(fit.keep[n_object:].any())
        xyz = np.stack([v[k] for k in "xyz"], axis=1).astype(float)
        body = xyz @ fit.matrix[:3, :3].T + fit.matrix[:3, 3]
        m_per_unit = 0.035 / R_FRUIT                                # el radio del fruto pasa a 3.5 cm
        fruit = body[:3000]
        self.assertAlmostEqual(np.percentile(fruit[:, 2], 0.5), 0.0, delta=0.002)            # base en z = 0
        self.assertLess(np.abs(np.median(fruit[:, :2], axis=0)).max(), 0.002)               # centrado en x, y
        self.assertAlmostEqual(np.percentile(np.linalg.norm(fruit[:, :2], axis=1), 98), 0.035, delta=0.001)
        stem = body[3000:3400]
        self.assertLess(np.median(np.linalg.norm(stem[:, :2], axis=1)), 0.003)               # tallo sobre z
        self.assertGreater(body[3400:n_object, 0].mean(), 0.005)                              # rama hacia +x
        p = fit.peduncle
        top_of_fruit = 2 * R_FRUIT * m_per_unit
        self.assertAlmostEqual(p["bottom"][2], top_of_fruit, delta=0.004)
        self.assertAlmostEqual(p["top"][2], (2 * R_FRUIT + STEM) * m_per_unit, delta=0.006)
        self.assertLess(p["bottom"][2], p["cut"][2])
        self.assertLess(p["cut"][2], p["top"][2])
        return fit, m_per_unit

    def test_frame_scale_and_peduncle(self):
        self.check()

    def test_sepals_hide_the_base_of_the_stem(self):
        fit, m_per_unit = self.check(gap=0.1)
        visible = (R_FRUIT * 2 + 0.1) * m_per_unit, (2 * R_FRUIT + STEM) * m_per_unit
        self.assertAlmostEqual(fit.peduncle["cut"][2], np.mean(visible), delta=0.005)     # mitad de lo visible

    def test_not_a_tomato(self):
        rng = np.random.default_rng(0)
        v = vertices(rng.normal(size=(500, 3)), np.array([GREEN] * 500))
        with self.assertRaises(SystemExit):
            fit_tomato(v, diameter=0.07)


if __name__ == "__main__":
    unittest.main()
