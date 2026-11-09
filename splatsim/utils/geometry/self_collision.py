"""Revisa si una trayectoria articular del xArm5 choca consigo misma, con la URDF en PyBullet.

En el robot real una autocolisión detiene el brazo (y la reproducción mide el choque, no el
seguimiento), así que replay_recordings.py revisa cada trayectoria antes de mandarla. Los pares de
links que ya se tocan en la pose de referencia (p.ej. el reset: dedos y nudillos del gripper,
links vecinos que la malla hace solaparse) se ignoran: no son choques, son la geometría del modelo.
"""
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

ARM_JOINTS = 5
MARGIN = 0.0     # m: distancia bajo la que dos links cuentan como en contacto (0 = penetración)


class SelfCollisionChecker:
    def __init__(self, urdf_path: str, reference_q, margin: float = MARGIN):
        self.c = BulletClient(connection_mode=p.DIRECT)
        self.body = self.c.loadURDF(urdf_path, useFixedBase=True,
                                    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        n = self.c.getNumJoints(self.body)
        self.names = {i: self.c.getJointInfo(self.body, i)[12].decode() for i in range(n)}
        self.names[-1] = "base"
        self.arm = [i for i in range(n) if self.c.getJointInfo(self.body, i)[2] != p.JOINT_FIXED][:ARM_JOINTS]
        self.margin = margin
        # link5 y lo que cuelga de él (gripper, cámara) se mueven juntos: entre ellos solo cambia
        # la apertura de la pinza, no es una autocolisión del brazo
        wrist = self.arm[-1]
        self.wrist_group = {i for i in range(n) if i >= wrist}
        self.ignored = self._pairs(reference_q)     # ya en contacto en la pose de referencia

    def _pairs(self, q) -> set[tuple[int, int]]:
        for i, v in zip(self.arm, q[:ARM_JOINTS]):
            self.c.resetJointState(self.body, i, float(v))
        self.c.performCollisionDetection()
        pts = self.c.getClosestPoints(self.body, self.body, distance=self.margin)
        return {tuple(sorted((pt[3], pt[4]))) for pt in pts if pt[3] != pt[4] and pt[8] <= self.margin
                and not {pt[3], pt[4]} <= self.wrist_group}

    def min_distance(self, q, reach: float = 0.3) -> tuple[float, str, str]:
        """Menor distancia (m) entre dos links que podrían chocar, y cuáles."""
        for i, v in zip(self.arm, q[:ARM_JOINTS]):
            self.c.resetJointState(self.body, i, float(v))
        self.c.performCollisionDetection()
        best = (reach, "", "")
        for pt in self.c.getClosestPoints(self.body, self.body, distance=reach):
            pair = tuple(sorted((pt[3], pt[4])))
            if pt[3] != pt[4] and pair not in self.ignored and not set(pair) <= self.wrist_group \
                    and pt[8] < best[0]:
                best = (pt[8], self.names[pair[0]], self.names[pair[1]])
        return best

    def collisions(self, q) -> list[tuple[str, str]]:
        """Pares de links en contacto en la pose q (sin los de la referencia)."""
        return sorted((self.names[a], self.names[b]) for a, b in self._pairs(q) - self.ignored)

    def check(self, joints, step: int = 1) -> tuple[int, list[tuple[str, str]]] | None:
        """Primer frame (cada step) con autocolisión y sus pares, o None si no choca."""
        for k in range(0, len(joints), step):
            pairs = self.collisions(joints[k])
            if pairs:
                return k, pairs
        return None
