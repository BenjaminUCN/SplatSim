"""Corte del pedúnculo en la simulación, con una cortadora cut-and-hold: al terminar de cerrar sobre
él lo corta y lo sujeta, así que el tomate se mueve con ella; si se abre lo suficiente, cae.

El tallo no tiene cuerpo en PyBullet: no lo empuja nada y se mueve con el tomate.
"""
import numpy as np

from splatsim.utils.scene.peduncle import CLOSED_FRACTION, DROPPED, HELD, INTACT, cut_check, next_stem_state, to_world

from .config import APPLE_NAME

BLADE_LINK = "gripper_center_link"   # mitad del filo
PIVOT_LINK = "hand_held_base"        # su eje z es el del pivote de la mandíbula


class PeduncleCutter:
    def __init__(self, server, stem: dict):
        self.s = server
        self.stem = stem
        client = server.pybullet_client
        robot = server.dummy_robot
        links = {client.getJointInfo(robot, i)[12].decode(): i for i in range(client.getNumJoints(robot))}
        self.blade_link = links.get(BLADE_LINK)   # None: esta escena no tiene cortadora, solo se ve el tallo
        self.pivot_link = links.get(PIVOT_LINK)
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

    def error(self) -> float | None:
        """Distancia (m) del centro del filo al punto de corte."""
        if self.blade_link is None:
            return None
        return float(np.linalg.norm(self._blade()[0] - self._stem_world()[2]))

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
