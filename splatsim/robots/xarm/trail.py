"""Dibuja en PyBullet el recorrido del efector final como una línea."""
from collections import deque

import numpy as np


class EETrail:
    def __init__(self, server, color=(1, 0, 0), width=3, min_dist=0.002, max_segments=3000):
        self.s = server
        self.color = list(color)
        self.width = width
        self.min_dist = min_dist                  # no dibuja si se movió menos de esto (m)
        self._ids = deque(maxlen=max_segments)    # límite para no saturar el visor
        self._prev = None

    def update(self, active: bool = True) -> None:
        """Llamar cada tick. Con active=False corta el trazo pero conserva lo dibujado."""
        if not active:
            self._prev = None
            return

        pos, _ = self.s.get_current_ee_pose()
        pos = np.array(pos)

        if self._prev is None:          # inicio de un trazo nuevo: borra el anterior
            self.clear()
            self._prev = pos
            return

        if np.linalg.norm(pos - self._prev) < self.min_dist:
            return
        if len(self._ids) == self._ids.maxlen:    # el más viejo sale del deque, hay que borrarlo
            self.s.pybullet_client.removeUserDebugItem(self._ids[0])
        self._ids.append(self.s.pybullet_client.addUserDebugLine(
            self._prev.tolist(), pos.tolist(), self.color, self.width, lifeTime=0))
        self._prev = pos

    def clear(self) -> None:
        for i in self._ids:
            self.s.pybullet_client.removeUserDebugItem(i)
        self._ids.clear()
