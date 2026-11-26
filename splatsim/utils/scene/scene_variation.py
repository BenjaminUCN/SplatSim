"""Variación por episodio para la transferencia sim2real: pose de las cámaras base y giro del tomate.

Cada cámara base se corre y se gira un poco respecto de su pose nominal (configs/camera_poses.yaml), como
un error de calibración que queda fijo durante el episodio. El tomate gira alrededor de su eje z, el del
tallo, así que el punto de corte no se mueve y las trayectorias planificadas siguen sirviendo; solo cambia
la cara que ven las cámaras. La cámara de muñeca no varía (está atornillada al brazo).

El sorteo depende solo de la semilla y del episodio (como el de la hoja, pero en otra secuencia), así que no
cambia al repartir los episodios entre procesos ni al continuar un render a medias.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VariationRanges:
    camera_xyz_m: float = 0.01        # ± por eje, en el mundo
    camera_rpy_deg: float = 1.0       # ± en roll, pitch y yaw de la cámara
    tomato_yaw_deg: float = 180.0     # ± alrededor del eje del tallo

    def scaled(self, factor: float) -> "VariationRanges":
        """Las cámaras factor veces más lejos de la nominal (B3: el doble del entrenamiento); el giro del
        tomate queda igual."""
        return VariationRanges(self.camera_xyz_m * factor, self.camera_rpy_deg * factor, self.tomato_yaw_deg)

    def as_dict(self) -> dict:
        return asdict(self)


def draw_variation(key: int, cameras: list[str], ranges: VariationRanges, seed: int = 0) -> dict:
    """{"camera_jitter": {cámara base: [dx, dy, dz (m), droll, dpitch, dyaw (rad)]}, "tomato_yaw": rad}.
    key: el número del episodio (o del ensayo). Las cámaras de muñeca no se sortean."""
    rng = np.random.default_rng([seed, int(key), 1])
    jitter = {}
    for name in sorted(cameras):       # el orden de --cameras no cambia el sorteo de cada una
        xyz = rng.uniform(-ranges.camera_xyz_m, ranges.camera_xyz_m, 3)
        rpy = np.radians(rng.uniform(-ranges.camera_rpy_deg, ranges.camera_rpy_deg, 3))
        if "wrist" not in name:
            jitter[name] = [round(float(v), 6) for v in (*xyz, *rpy)]
    yaw = math.radians(rng.uniform(-ranges.tomato_yaw_deg, ranges.tomato_yaw_deg))
    return {"camera_jitter": jitter, "tomato_yaw": round(yaw, 6)}


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """xyzw (PyBullet) del giro yaw alrededor de z."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def turn_about_own_z(quat_xyzw, yaw: float) -> tuple[float, float, float, float]:
    """quat seguido de un giro yaw alrededor del eje z del propio objeto (el tallo del tomate)."""
    x1, y1, z1, w1 = quat_xyzw
    x2, y2, z2, w2 = yaw_quaternion(yaw)
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
