"""Corte del pedúnculo en la simulación, con una cortadora cut-and-hold: al terminar de cerrar sobre
él lo corta y lo sujeta, así que el tomate se mueve con ella; si se abre lo suficiente, cae.

El tallo no tiene cuerpo en PyBullet: no lo empuja nada y se mueve con el tomate.

Con cut_range_mm en el pedúnculo (objects.yaml), el corte se decide con la hoja real (real_cut_check) en ese
tramo del filo, y el error es la distancia al punto del filo al que apuntan las demos: el medio del tramo,
cut_target_dtheta() grados hacia el lado abierto. Sin cut_range_mm, como antes: a lo más BLADE_REACH del
centro del filo (gripper_center_link), que también es el punto del error.
"""
import numpy as np

from splatsim.utils.scene.hand_held import OPENING_TABLES, real_blade_point_mm
from splatsim.utils.scene.peduncle import (CLOSED_FRACTION, DROPPED, HELD, INTACT, cut_check, cut_target_dtheta,
                                           next_stem_state, real_cut_check, to_world)

from .config import APPLE_NAME

BLADE_LINK = "gripper_center_link"   # mitad del filo
PIVOT_LINK = "hand_held_base"        # su eje z es el del pivote de la mandíbula


class PeduncleCutter:
    opening = OPENING_TABLES["b2"]   # apertura de la cortadora real; la escena la cambia con pruner_opening
    def __init__(self, server, stem: dict):
        self.s = server
        self.stem = stem
        client = server.pybullet_client
        robot = server.dummy_robot
        links = {client.getJointInfo(robot, i)[12].decode(): i for i in range(client.getNumJoints(robot))}
        self.blade_link = links.get(BLADE_LINK)   # None: esta escena no tiene cortadora, solo se ve el tallo
        self.pivot_link = links.get(PIVOT_LINK)
        self.cut_range_mm = tuple(stem["cut_range_mm"]) if stem.get("cut_range_mm") else None
        # apertura de la cortadora real: pruner_opening de la escena (b2 por defecto, b3 para la pinza v6)
        self.opening = OPENING_TABLES[server.object_config[server.robot_name].get("pruner_opening", "b2")]
        self._target_mm = None   # cut_range_mm: punto del filo al que apuntar, en hand_held_base (mm)
        self.state = INTACT
        self.last_check = None
        self._was_closed = False
        self._grip = None    # pose del tomate en el marco del filo mientras está sujeto

    @property
    def cut(self) -> bool:
        return self.state != INTACT

    @property
    def held(self) -> bool:
        return self.state == HELD

    @classmethod
    def for_server(cls, server):
        stem = server.object_config[APPLE_NAME].get("peduncle")
        return cls(server, stem) if stem else None

    def stem_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bottom, top y cut del tallo en el mundo, ahora."""
        return self._stem_world()

    def _stem_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.s.pybullet_client
        pos, quat = c.getBasePositionAndOrientation(self.s.apple.body_id)
        rot = np.reshape(c.getMatrixFromQuaternion(quat), (3, 3))
        return tuple(to_world(self.stem[k], pos, rot) for k in ("bottom", "top", "cut"))

    def _blade(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.s.pybullet_client
        robot = self.s.dummy_robot
        blade = c.getLinkState(robot, self.blade_link, computeForwardKinematics=True)[4]
        quat = c.getLinkState(robot, self.pivot_link, computeForwardKinematics=True)[5]
        return np.array(blade), np.reshape(c.getMatrixFromQuaternion(quat), (3, 3))[:, 2]

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        """Posición y rotación de hand_held_base (el eje del pivote es su z)."""
        c = self.s.pybullet_client
        state = c.getLinkState(self.s.dummy_robot, self.pivot_link, computeForwardKinematics=True)
        return np.array(state[4]), np.reshape(c.getMatrixFromQuaternion(state[5]), (3, 3))

    def _to_base_mm(self, points) -> list[np.ndarray]:
        pos, rot = self._base()
        return [rot.T @ (np.asarray(x) - pos) * 1000 for x in points]

    def plane_z_mm(self) -> float:
        """z del plano de la hoja en hand_held_base: la de gripper_center_link (a media altura de la mandíbula)."""
        return float(self._to_base_mm([self._blade()[0]])[0][2])

    def target_mm(self) -> np.ndarray:
        """cut_range_mm: el punto del filo al que apuntar, en hand_held_base (mm)."""
        if self._target_mm is None:
            self._target_mm = real_blade_point_mm(float(np.mean(self.cut_range_mm)), self.target_dtheta(),
                                                  self.plane_z_mm())
        return self._target_mm

    def target_dtheta(self) -> float:
        """Grados del filo cerrado hacia el lado abierto a los que apuntar, con la apertura de esta cortadora."""
        return cut_target_dtheta(opening=self.opening)

    def error(self) -> float | None:
        """Distancia (m) del punto del filo (el medio de cut_range_mm o, sin él, el centro del filo) al punto
        de corte del tallo."""
        if self.blade_link is None:
            return None
        if self.cut_range_mm is not None:
            pos, rot = self._base()
            return float(np.linalg.norm(pos + rot @ self.target_mm() / 1000 - self._stem_world()[2]))
        return float(np.linalg.norm(self._blade()[0] - self._stem_world()[2]))

    def would_cut(self) -> bool | None:
        """Si la cortadora cortara al cerrar ahora, con el brazo quieto: la misma regla que update() aplica al
        llegar a CLOSED_FRACTION, sobre la posición actual del tallo. None sin cortadora o con el tallo ya cortado.
        Es información de la simulación, que la política no ve (--close_when_cuttable la usa como oráculo)."""
        if self.blade_link is None or self.state != INTACT:
            return None
        bottom, top, _ = self._stem_world()
        if self.cut_range_mm is not None:
            b, t = self._to_base_mm([bottom, top])
            return real_cut_check(b, t, self.plane_z_mm(), self.cut_range_mm,
                                  radius_mm=self.stem.get("radius", 0.0025) * 1000, opening=self.opening) == "ok"
        return cut_check(bottom, top, *self._blade()) == "ok"

    def reset(self) -> None:
        """El tomate volvió a su lugar: tiene el tallo entero."""
        self.state = INTACT
        self.last_check = None
        self._grip = None
        self._was_closed = self.s.gripper.get_state() >= CLOSED_FRACTION

    def update(self) -> None:
        """Llamar después de avanzar la física. Entrar con la cortadora ya cerrada no corta."""
        if self.blade_link is None:
            return
        gripper = self.s.gripper.get_state()
        before = self.state
        self.state = next_stem_state(before, gripper, self._was_closed, self._check)
        self._was_closed = gripper >= CLOSED_FRACTION
        if self.state == HELD and before != HELD:
            self._take()
        elif self.state == DROPPED and before != DROPPED:
            self._drop()
        if self.state == HELD:
            self._follow()

    def _check(self) -> str:
        bottom, top, _ = self._stem_world()
        if self.cut_range_mm is not None:
            b, t = self._to_base_mm([bottom, top])
            self.last_check = real_cut_check(b, t, self.plane_z_mm(), self.cut_range_mm,
                                             radius_mm=self.stem.get("radius", 0.0025) * 1000, opening=self.opening)
        else:
            self.last_check = cut_check(bottom, top, *self._blade())
        print(f"[xArm5] cortadora cerrada, pedúnculo: {self.last_check}")
        return self.last_check

    def _blade_pose(self) -> tuple:
        state = self.s.pybullet_client.getLinkState(self.s.dummy_robot, self.blade_link,
                                                    computeForwardKinematics=True)
        return state[4], state[5]

    def _take(self) -> None:
        c = self.s.pybullet_client
        inverse = c.invertTransform(*self._blade_pose())
        self._grip = c.multiplyTransforms(*inverse, *c.getBasePositionAndOrientation(self.s.apple.body_id))

    def _follow(self) -> None:
        c = self.s.pybullet_client
        pose = c.multiplyTransforms(*self._blade_pose(), *self._grip)
        c.resetBasePositionAndOrientation(self.s.apple.body_id, *pose)   # sigue estático: no choca ni cae

    def _drop(self) -> None:
        print("[xArm5] cortadora abierta: el tomate cae")
        self._grip = None
        mass = self.s.urdf_object_mass_list[self.s.splat_object_name_list.index(APPLE_NAME)]
        self.s.pybullet_client.changeDynamics(self.s.apple.body_id, -1, mass=mass)   # estaba estático
