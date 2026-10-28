"""Manzana de la escena: cuerpo estático + sliders opcionales para moverla."""
import numpy as np

from splatsim.utils.apple_grid import APPLE_GRID_PATH, load_grid, load_random_range, save_grid_file

from .config import APPLE_INIT, APPLE_NAME, APPLE_RANGE

MIN_RANGE_SIZE = 0.01   # m: lado mínimo del rango aleatorio


class AppleController:
    def __init__(self, server, movable: bool = False):
        self.s = server
        self.movable = movable
        # rango de "Random Position": el guardado en APPLE_GRID_PATH o APPLE_RANGE
        self.range = load_random_range(APPLE_RANGE)

        idx = server.splat_object_name_list.index(APPLE_NAME)
        self.body_id = server.urdf_object_list[idx]
        print(f"[xArm5] apple body_id: {self.body_id}")

        # mass=0 => cuerpo estático (no solo "sin gravedad")
        server.pybullet_client.changeDynamics(self.body_id, -1, mass=0)
        print("[xArm5] gravedad desactivada para la apple")

        # parte en la primera posición de la grilla, si hay
        grid = load_grid()
        if grid:
            self.move_to(grid[0])
            print(f"[xArm5] apple en la posición #0 de la grilla: {grid[0]}")

        if movable:
            self._create_sliders()

    def _create_sliders(self):
        c = self.s.pybullet_client
        self._prev = list(APPLE_INIT)
        self._sliders = [
            c.addUserDebugParameter(f"apple_{axis}", *APPLE_RANGE[axis], APPLE_INIT[i])
            for i, axis in enumerate("xyz")
        ]
        self._rand_btn = c.addUserDebugParameter("Set Random Apple Position", 1, 0, 0)
        self._prev_rand = 0
        self._save_btn = c.addUserDebugParameter("Save Apple Position to Grid", 1, 0, 0)
        self._prev_save = 0

    def move_to(self, xyz):
        self.s.set_object_pose(APPLE_NAME, np.array(xyz), np.array([0, 0, 0, 1]), use_gravity=False)

    def random_position(self):
        return [np.random.uniform(*self.range[a]) for a in "xyz"]

    def set_range(self, lo, hi) -> None:
        """Rango aleatorio entre las esquinas lo y hi (se ordenan; lado mínimo MIN_RANGE_SIZE)."""
        rng = {}
        for i, a in enumerate("xyz"):
            a_lo, a_hi = sorted((float(lo[i]), float(hi[i])))
            rng[a] = (a_lo, max(a_hi, a_lo + MIN_RANGE_SIZE))
        self.range = rng

    def range_corners(self):
        return (tuple(self.range[a][0] for a in "xyz"), tuple(self.range[a][1] for a in "xyz"))

    def save_range(self) -> None:
        save_grid_file(random_range=self.range)
        print(f"[xArm5] Rango aleatorio {self.range} guardado en {APPLE_GRID_PATH}")

    def update(self) -> None:
        """Llamar cada tick."""
        if not self.movable:
            return
        c = self.s.pybullet_client

        vals = [c.readUserDebugParameter(s) for s in self._sliders]
        if any(abs(v - p) > 1e-6 for v, p in zip(vals, self._prev)):
            self._prev = vals
            self.move_to(vals)

        btn = c.readUserDebugParameter(self._rand_btn)
        if btn != self._prev_rand:  # en PyBullet un botón se detecta porque cambia su valor
            self._prev_rand = btn
            print("[xArm5] Setting random apple position.")
            self.move_to(self.random_position())

        btn = c.readUserDebugParameter(self._save_btn)
        if btn != self._prev_save:
            self._prev_save = btn
            self.save_to_grid()

    def save_to_grid(self) -> None:
        """Agrega la posición actual de la manzana a APPLE_GRID_PATH."""
        pos, _ = self.s.pybullet_client.getBasePositionAndOrientation(self.body_id)
        pos = [round(float(v), 4) for v in pos]

        positions = load_grid()
        if pos in positions:
            print(f"[xArm5] La posición {pos} ya está en la grilla (#{positions.index(pos)}).")
            return
        positions.append(pos)
        save_grid_file(positions=positions)
        print(f"[xArm5] Posición #{len(positions) - 1} {pos} guardada en {APPLE_GRID_PATH}")
