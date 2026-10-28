"""Panel de control del sim (viser): el panel compartido (splatsim/robots/station/panel.py) más
la escena: tomate y grilla de posiciones, poses de las cámaras base y el reloj de la simulación.
"""
import numpy as np
import viser

from splatsim.robots.station.panel import StationPanel, wxyz_to_xyzw, xyzw_to_wxyz
from splatsim.utils.apple_grid import APPLE_GRID_PATH, load_grid

from .config import BASE_CAM_FOV, SIM_CLOCKS

RANGE_COLOR = (255, 170, 0)   # caja del rango aleatorio (como las esferas de la grilla)
from .debug_ui import save_camera_poses


class SimPanel(StationPanel):
    def __init__(self, server, base_cams, port: int = 8080):
        self.base_cams = base_cams
        super().__init__(server, port=port, label="xArm5 sim")

    def _build_robot_extra(self):
        # fixed: cada comando avanza 1/hz simulado; realtime: la física sigue al reloj (GELLO)
        clock = self.v.gui.add_dropdown("Sim clock", list(SIM_CLOCKS), initial_value=self.s.sim_clock)
        self._on_user(clock, lambda h: self.s.set_sim_clock(h.value))

    def _build_tabs(self, tabs):
        with tabs.add_tab("Scene", viser.Icon.ADJUSTMENTS):
            self._build_apple_gui()
            self._build_camera_gui()

    def _update_status_extra(self):
        # la manzana también se mueve desde fuera (p.ej. run_policy_trials.py)
        pos = self._apple_pos()
        if np.abs(np.subtract(pos, self._apple_xyz.value)).max() > 1e-3:
            self._sync_apple_ui(pos)

    # ── cámaras base ─────────────────────────────────────────────────────────
    def _build_camera_gui(self):
        fovx, fovy = BASE_CAM_FOV
        aspect = np.tan(fovx / 2) / np.tan(fovy / 2)
        self._cam_ui = {}
        with self.v.gui.add_folder("Base cameras"):
            tabs = self.v.gui.add_tab_group()
            for cam in self.base_cams:
                with tabs.add_tab(cam.name):
                    xyz = self.v.gui.add_vector3("xyz (m)", cam.xyz, step=0.001)
                    rpy = self.v.gui.add_vector3("rpy (rad)", cam.rpy, step=0.001)
                    show = self.v.gui.add_checkbox("Show gizmo", True)
                gizmo = self.v.scene.add_transform_controls(
                    f"/cams/{cam.name}", scale=0.15, position=cam.xyz)
                self.v.scene.add_camera_frustum(
                    f"/cams/{cam.name}/frustum", fov=fovy, aspect=aspect, scale=0.08,
                    color=(30, 120, 255), thickness=1.5, thickness_units="screen")
                self.v.scene.add_label(f"/cams/{cam.name}/label", cam.name)
                self._cam_ui[cam.name] = (xyz, rpy, gizmo)

                self._on_user(xyz, lambda _, c=cam: self._set_cam(c, xyz=self._cam_ui[c.name][0].value))
                self._on_user(rpy, lambda _, c=cam: self._set_cam(c, rpy=self._cam_ui[c.name][1].value))
                gizmo.on_update(lambda e, c=cam: self._do(self._cam_from_gizmo, c))
                show.on_update(lambda _, g=gizmo, s=show: setattr(g, "visible", s.value))
            self._on_click(self.v.gui.add_button("Save Camera Poses"), self._save_cam_poses)
        # tras crear todo: orientar los gizmos con la pose actual
        for cam in self.base_cams:
            self._do(self._set_cam, cam)

    def _set_cam(self, cam, xyz=None, rpy=None):
        cam.set_pose(xyz if xyz is not None else cam.xyz, rpy if rpy is not None else cam.rpy)
        xyz_ui, rpy_ui, gizmo = self._cam_ui[cam.name]
        xyz_ui.value, rpy_ui.value = cam.xyz, cam.rpy
        gizmo.position = cam.xyz
        gizmo.wxyz = xyzw_to_wxyz(self.s.pybullet_client.getQuaternionFromEuler(cam.rpy))

    def _cam_from_gizmo(self, cam):
        _, _, gizmo = self._cam_ui[cam.name]
        rpy = self.s.pybullet_client.getEulerFromQuaternion(wxyz_to_xyzw(gizmo.wxyz))
        cam.set_pose(tuple(float(v) for v in gizmo.position), tuple(float(v) for v in rpy))
        xyz_ui, rpy_ui, _ = self._cam_ui[cam.name]
        xyz_ui.value, rpy_ui.value = cam.xyz, cam.rpy

    def _save_cam_poses(self):
        save_camera_poses(self.base_cams)

    # ── manzana ──────────────────────────────────────────────────────────────
    def _build_apple_gui(self):
        apple = self.s.apple
        pos = self._apple_pos()
        with self.v.gui.add_folder("Tomato"):
            self._apple_xyz = self.v.gui.add_vector3("xyz (m)", pos, step=0.001)
            rand = self.v.gui.add_button("Random Position")
            save = self.v.gui.add_button("Save Position to Grid")
            self._grid_info = self.v.gui.add_markdown("")
            # cargar una posición guardada (también se puede hacer clic en su esfera)
            self._grid_select = self.v.gui.add_dropdown("Grid position", ["-"])
            load = self.v.gui.add_button("Load Grid Position")
            self._build_range_gui()
        self._apple_gizmo = self.v.scene.add_transform_controls(
            "/apple", scale=0.12, position=pos, disable_rotations=True)
        self.v.scene.add_icosphere("/apple/sphere", radius=0.03, color=(230, 40, 30))

        self._on_user(self._apple_xyz, lambda h: self._move_apple(h.value))
        self._apple_gizmo.on_update(
            lambda _: self._do(self._move_apple, tuple(float(v) for v in self._apple_gizmo.position)))
        self._on_click(rand, lambda: self._move_apple(apple.random_position()))
        self._on_click(save, self._save_apple)
        self._on_user(self._grid_select, lambda _: self._load_grid_position())
        self._on_click(load, self._load_grid_position)
        self._grid_positions = []
        self._show_grid()

    # ── rango de la posición aleatoria ───────────────────────────────────────
    def _build_range_gui(self):
        """Caja donde "Random Position" pone el tomate: esquinas editables con números o con
        los dos gizmos de la escena; Save Range la guarda en APPLE_GRID_PATH."""
        lo, hi = self.s.apple.range_corners()
        with self.v.gui.add_folder("Random range", expand_by_default=False):
            self._range_min = self.v.gui.add_vector3("Min (m)", lo, step=0.01)
            self._range_max = self.v.gui.add_vector3("Max (m)", hi, step=0.01)
            self._range_show = self.v.gui.add_checkbox("Show box", False, hint="The range as a box in the 3D view")
            self._range_edit = self.v.gui.add_checkbox("Corner gizmos", False,
                                                       hint="Drag the box corners in the 3D view (with Show box)")
            save = self.v.gui.add_button("Save Range")
            self._range_info = self.v.gui.add_markdown("")
        self._range_gizmos = [
            self.v.scene.add_transform_controls(f"/apple_range/{name}", scale=0.08,
                                                disable_rotations=True, position=corner)
            for name, corner in (("min", lo), ("max", hi))]
        self._on_user(self._range_min, lambda _: self._set_range(self._range_min.value, self._range_max.value))
        self._on_user(self._range_max, lambda _: self._set_range(self._range_min.value, self._range_max.value))
        for gizmo in self._range_gizmos:
            gizmo.on_update(lambda _: self._do(self._set_range, *(g.position for g in self._range_gizmos)))
        self._range_show.on_update(lambda _: self._draw_range())
        self._range_edit.on_update(lambda _: self._draw_range())
        self._on_click(save, self._save_range)
        self._range_saved = self.s.apple.range
        self._draw_range()

    def _set_range(self, lo, hi):
        self.s.apple.set_range(lo, hi)
        lo, hi = self.s.apple.range_corners()        # ordenadas y con lado mínimo
        self._range_min.value, self._range_max.value = lo, hi
        for gizmo, corner in zip(self._range_gizmos, (lo, hi)):
            if np.abs(np.subtract(gizmo.position, corner)).max() > 1e-4:
                gizmo.position = corner             # no se pisa al gizmo mientras se arrastra
        self._draw_range()

    def _save_range(self):
        self.s.apple.save_range()
        self._range_saved = self.s.apple.range
        self._draw_range()

    def _draw_range(self):
        lo, hi = (np.array(c) for c in self.s.apple.range_corners())
        show = self._range_show.value
        center, size = tuple((lo + hi) / 2), tuple(hi - lo)
        # caja translúcida y su contorno, del color de las esferas de la grilla
        self.v.scene.add_box("/apple_range/box", color=RANGE_COLOR, dimensions=size, position=center,
                             opacity=0.08, visible=show)
        self.v.scene.add_box("/apple_range/edges", color=RANGE_COLOR, dimensions=size, position=center,
                             wireframe=True, visible=show)
        for gizmo in self._range_gizmos:
            gizmo.visible = show and self._range_edit.value
        saved = self.s.apple.range == self._range_saved
        dims = ", ".join(f"{a} ∈ [{self.s.apple.range[a][0]:.2f}, {self.s.apple.range[a][1]:.2f}]" for a in "xyz")
        self._range_info.content = (f"{dims} m" + ("" if saved else "  \n**not saved**: Random Position "
                                                   "uses it now; Save Range keeps it for next time"))

    def _move_apple(self, xyz):
        self.s.apple.move_to(xyz)
        self._sync_apple_ui(self._apple_pos())

    def _sync_apple_ui(self, pos):
        self._apple_xyz.value = pos
        if np.abs(np.subtract(pos, self._apple_gizmo.position)).max() > 1e-3:
            self._apple_gizmo.position = pos  # no se pisa al gizmo mientras se arrastra

    def _apple_pos(self):
        pos, _ = self.s.pybullet_client.getBasePositionAndOrientation(self.s.apple.body_id)
        return tuple(round(float(v), 4) for v in pos)

    def _save_apple(self):
        self.s.apple.save_to_grid()
        self._show_grid()

    def _show_grid(self):
        """Dibuja las posiciones guardadas en APPLE_GRID_PATH como esferas numeradas."""
        positions = load_grid()
        self._grid_positions = [tuple(p) for p in positions]
        for i, p in enumerate(self._grid_positions):
            sphere = self.v.scene.add_icosphere(f"/grid/{i}", radius=0.012, color=(255, 170, 0),
                                                opacity=0.6, position=p)
            self.v.scene.add_label(f"/grid/{i}/label", str(i))
            sphere.on_click(lambda _, i=i: self._do(self._load_grid_position, i))
        self._grid_info.content = f"{len(positions)} positions in `{APPLE_GRID_PATH}`"
        self._grid_select.options = [f"#{i} {p}" for i, p in enumerate(self._grid_positions)] or ["-"]

    def _load_grid_position(self, index=None):
        """Mueve la manzana a la posición `index` de la grilla (por defecto la elegida en la lista)."""
        if index is None:
            if not self._grid_positions:
                return
            index = self._grid_select.options.index(self._grid_select.value)
        self._grid_select.value = self._grid_select.options[index]
        self._move_apple(self._grid_positions[index])
