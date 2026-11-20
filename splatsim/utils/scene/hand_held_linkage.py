"""Cuatro barras del gripper hand_held_XL330 v4: horn -> bastón -> pasador del pruner -> mandíbula.

Es la cinemática de construir_hand_held_XL330_v4.py (CAD en FreeCAD), en planta y en mm, con los mismos
nombres de parámetros que el manifest.json de las mallas ("linkage"). t es la apertura de la mandíbula:
0 = cerrada (como las mallas), 1 = abierta (girada -jaw_travel alrededor del pivote).

En el URDF todo vale 0 con el gripper abierto (t = 1), igual que el horn del v3:
  gripper_drive_joint  = horn(t) - horn(1)          crece al cerrar
  hand_held_jaw_joint  = (1 - t) * jaw_travel
  hand_held_rod_joint  = rod(t) - rod(1) - drive     el bastón respecto del horn
La relación no es lineal; jaw_and_rod() la invierte con una tabla.
"""
import math
from dataclasses import dataclass, field

import numpy as np

ROD_JOINT = "hand_held_rod_joint"


@dataclass
class Linkage:
    horn_axis_mm: tuple          # eje del horn (x, y)
    pivot_mm: tuple              # pivote de la mandíbula (x, y)
    yoke_mm: tuple               # centro de la ranura de fixed_steel: el pasador del pruner a mitad de recorrido
    jaw_travel_rad: float        # de cerrada a abierta
    crank_mm: float              # radio del pasador del bastón en el horn
    rod_mm: float                # bastón, centro a centro
    branch: int = 1              # solución de la manivela (el CAD usa 1: hacia -y)
    _table: tuple = field(default=None, repr=False)

    @classmethod
    def from_manifest(cls, data: dict) -> "Linkage":
        return cls(tuple(data["horn_axis_mm"]), tuple(data["pivot_mm"]), tuple(data["yoke_mm"]),
                   float(data["jaw_travel_rad"]), float(data["crank_mm"]), float(data["rod_mm"]),
                   int(data.get("branch", 1)))

    # ── en planta, en función de la apertura t ──────────────────────────────
    def yoke_pin(self, t: float) -> np.ndarray:
        """Pasador del pruner (pasador_yugo del CAD)."""
        px, py = self.pivot_mm
        r = math.hypot(self.yoke_mm[0] - px, self.yoke_mm[1] - py)
        mid = math.atan2(self.yoke_mm[1] - py, self.yoke_mm[0] - px)
        fi = mid + self.jaw_travel_rad / 2 - t * self.jaw_travel_rad
        return np.array([px + r * math.cos(fi), py + r * math.sin(fi)])

    def horn_angle(self, t: float) -> float:
        """Ángulo del pasador del bastón en el horn (manivela del CAD), en rad."""
        p = self.yoke_pin(t)
        vx, vy = p[0] - self.horn_axis_mm[0], p[1] - self.horn_axis_mm[1]
        d = math.hypot(vx, vy)
        c = (self.crank_mm ** 2 + d * d - self.rod_mm ** 2) / (2 * self.crank_mm * d)
        if abs(c) > 1:
            raise ValueError(f"the rod does not reach with crank {self.crank_mm} mm at t = {t}")
        return math.atan2(vy, vx) + self.branch * math.acos(c)

    def horn_pin(self, t: float) -> np.ndarray:
        a = self.horn_angle(t)
        return np.array(self.horn_axis_mm) + self.crank_mm * np.array([math.cos(a), math.sin(a)])

    def rod_angle(self, t: float) -> float:
        """Dirección del bastón, del pasador del horn al del pruner."""
        h, p = self.horn_pin(t), self.yoke_pin(t)
        return math.atan2(p[1] - h[1], p[0] - h[0])

    # ── joints del URDF ─────────────────────────────────────────────────────
    def joints_at(self, t: float) -> tuple[float, float, float]:
        """(drive, jaw, rod) en rad para la apertura t."""
        drive = _wrap(self.horn_angle(t) - self.horn_angle(1.0))
        rod = _wrap(self.rod_angle(t) - self.rod_angle(1.0) - drive)
        return drive, (1.0 - t) * self.jaw_travel_rad, rod

    @property
    def travel_rad(self) -> float:
        """Recorrido del horn de abierta a cerrada."""
        return self.joints_at(0.0)[0]

    def jaw_and_rod(self, drive: float) -> tuple[float, float]:
        """jaw y rod para un ángulo del horn (gripper_drive_joint)."""
        if self._table is None:
            t = np.linspace(1.0, 0.0, 401)
            table = np.array([self.joints_at(v) for v in t])
            if np.any(np.diff(table[:, 0]) <= 0):
                raise ValueError("the horn angle is not monotonic over the jaw travel")
            self._table = (table[:, 0], table[:, 1], table[:, 2])
        d, jaw, rod = self._table
        drive = float(np.clip(drive, d[0], d[-1]))
        return float(np.interp(drive, d, jaw)), float(np.interp(drive, d, rod))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
