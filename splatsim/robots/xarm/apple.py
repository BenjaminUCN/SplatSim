"""Manzana de la escena: cuerpo estático + sliders opcionales para moverla."""
import numpy as np
import yaml

from .config import APPLE_GRID_PATH, APPLE_INIT, APPLE_NAME, APPLE_RANGE

_GRID_HEADER = """\
# Tomato positions (x, y, z) in PyBullet world coordinates used by scripts/run_policy_trials.py.
# Add positions with the "Save Apple Position to Grid" button (launch_nodes.py --move_apple)
# or edit this list by hand.
"""


class AppleController:
    def __init__(self, server, movable: bool = False):
        self.s = server
        self.movable = movable

        idx = server.splat_object_name_list.index(APPLE_NAME)
        self.body_id = server.urdf_object_list[idx]
        print(f"[xArm5] apple body_id: {self.body_id}")

        # mass=0 => cuerpo estático (no solo "sin gravedad")
        server.pybullet_client.changeDynamics(self.body_id, -1, mass=0)
        print("[xArm5] gravedad desactivada para la apple")

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

    def _move_to(self, xyz):
        self.s.set_object_pose(APPLE_NAME, list(xyz), [0, 0, 0, 1], use_gravity=False)

    def update(self) -> None:
        """Llamar cada tick."""
        if not self.movable:
            return
        c = self.s.pybullet_client

        vals = [c.readUserDebugParameter(s) for s in self._sliders]
        if any(abs(v - p) > 1e-6 for v, p in zip(vals, self._prev)):
            self._prev = vals
            self._move_to(vals)

        btn = c.readUserDebugParameter(self._rand_btn)
        if btn != self._prev_rand:  # en PyBullet un botón se detecta porque cambia su valor
            self._prev_rand = btn
            print("[xArm5] Setting random apple position.")
            self._move_to([np.random.uniform(*APPLE_RANGE[a]) for a in "xyz"])

        btn = c.readUserDebugParameter(self._save_btn)
        if btn != self._prev_save:
            self._prev_save = btn
            self._save_to_grid()

    def _save_to_grid(self) -> None:
        """Agrega la posición actual de la manzana a APPLE_GRID_PATH."""
        pos, _ = self.s.pybullet_client.getBasePositionAndOrientation(self.body_id)
        pos = [round(float(v), 4) for v in pos]

        try:
            with open(APPLE_GRID_PATH) as f:
                positions = (yaml.safe_load(f) or {}).get("positions") or []
        except FileNotFoundError:
            positions = []
        if pos in positions:
            print(f"[xArm5] La posición {pos} ya está en la grilla (#{positions.index(pos)}).")
            return
        positions.append(pos)

        with open(APPLE_GRID_PATH, "w") as f:
            f.write(_GRID_HEADER)
            yaml.safe_dump({"positions": positions}, f, default_flow_style=None)
        print(f"[xArm5] Posición #{len(positions) - 1} {pos} guardada en {APPLE_GRID_PATH}")
