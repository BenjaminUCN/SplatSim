"""Panel de control del sim (viser): el panel compartido (splatsim/robots/station/panel.py) más
la escena: tomate y grilla de posiciones, poses de las cámaras base y el reloj de la simulación.
"""
import functools
import os

import numpy as np
import viser

from splatsim.robots.station.panel import StationPanel, wxyz_to_xyzw, xyzw_to_wxyz
from splatsim.utils.scene.apple_grid import (
    APPLE_GRID_PATH,
    EVAL_GRID_PATH,
    RANGE_SHAPES,
    NO_X_LIMIT,
    NO_Y_LIMIT,
    hemisphere_lines,
    in_hemisphere,
    load_grid,
)
from splatsim.utils.scene.camera_tripod import tripod_mesh
from splatsim.utils.scene.hand_held import CAMERA_MESH_DIR, D435I_MESHES, read_stl
from splatsim.utils.scene.leaf_segmentation import place_between, saved_leaves
from splatsim.utils.scene.peduncle import DROPPED, EMBED, HELD, INTACT
from splatsim.utils.render.occlusion import fit_lateral_offset, fit_message

from .config import APPLE_NAME, BASE_CAM_FOV, SIM_CLOCKS

RANGE_COLOR = (255, 170, 0)   # caja del rango aleatorio (como las esferas de la grilla)
# tallo en la vista 3D: entero (el verde del splat), cortado y sujeto, cortado y suelto
STEM_COLORS = {INTACT: (69, 117, 33), HELD: (40, 200, 220), DROPPED: (140, 140, 140)}
CUT_COLOR = (255, 230, 0)     # punto de corte del tallo
TRIPOD_COLOR = (35, 35, 38)   # trípode de mesa de las cámaras base


def fruit_sphere(config: dict) -> tuple[np.ndarray, float]:
    """Centro (marco del cuerpo) y radio de una esfera que cubre el tomate, desde su aabb en
    objects.yaml; con pedúnculo, el aabb llega a la punta del tallo, así que el tomate termina
    donde empieza el tallo (bottom + EMBED)."""
    box = config.get("aabb", {}).get("bounding_box")
    if box is None:
        return np.zeros(3), 0.03
    lo, hi = np.array(box[0], float), np.array(box[1], float)
    stem = config.get("peduncle")
    if stem:
        hi[2] = min(hi[2], stem["bottom"][2] + EMBED)
    return (lo + hi) / 2, float(np.mean(hi - lo) / 2)


@functools.cache
def d435i_meshes() -> dict:
    """Mallas de la D435i (scripts/xarm5/build_d435i_mesh.py): vértices, caras y color por parte."""
    meshes = {}
    for filename, rgba in D435I_MESHES:
        triangles = read_stl(os.path.join(CAMERA_MESH_DIR, filename)).reshape(-1, 3)
        vertices, faces = np.unique(triangles.round(7), axis=0, return_inverse=True)
        color = tuple(int(float(v) * 255) for v in rgba.split()[:3])
        meshes[filename.removesuffix(".stl")] = (vertices.astype(np.float32), faces.reshape(-1, 3).astype(np.uint32), color)
    return meshes


def z_to(axis) -> tuple:
    """wxyz del giro más corto que lleva el eje z a axis (unitario)."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, axis))
    if c < -1 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)              # 180° en torno a x
    w = np.r_[1.0 + c, np.cross(z, axis)]
    return tuple(w / np.linalg.norm(w))


# grillas que el panel puede mostrar: (archivo, color de las esferas). Solo la propia se edita con
# Save / Remove; la de evaluación es fija para que todas las evaluaciones usen las mismas posiciones
GRIDS = {"custom": (APPLE_GRID_PATH, RANGE_COLOR), "eval 4x4": (EVAL_GRID_PATH, (40, 120, 215))}
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
            if self.s.occluders is not None:
                self._build_occluder_gui()
            self._build_occlusion_gui()

    def _update_status_extra(self):
        # la manzana también se mueve desde fuera (p.ej. run_policy_trials.py)
        pos = self._apple_pos()
        if np.abs(np.subtract(pos, self._apple_xyz.value)).max() > 1e-3:
            self._sync_apple_ui(pos)
        self._draw_fruit()
        if self._stem is not None:
            self._draw_stem()
        if self.s.peduncle is not None and self.s.peduncle.blade_link is not None:
            p = self.s.peduncle
            state = ("**cut, held**" if p.held else "**cut, dropped**") if p.cut else (f"not cut: {p.last_check}" if p.last_check else "not cut")
            self._peduncle_text.content = f"Peduncle: blade at {p.error() * 100:.1f} cm, {state}"

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
                # la D435i con su lente RGB en la pose de la cámara (mallas en el marco óptico de la RGB)
                for part, (vertices, faces, color) in d435i_meshes().items():
                    self.v.scene.add_mesh_simple(f"/cams/{cam.name}/{part}", vertices, faces, color=color)
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
        self._draw_tripod(cam)

    def _cam_from_gizmo(self, cam):
        _, _, gizmo = self._cam_ui[cam.name]
        rpy = self.s.pybullet_client.getEulerFromQuaternion(wxyz_to_xyzw(gizmo.wxyz))
        cam.set_pose(tuple(float(v) for v in gizmo.position), tuple(float(v) for v in rpy))
        xyz_ui, rpy_ui, _ = self._cam_ui[cam.name]
        xyz_ui.value, rpy_ui.value = cam.xyz, cam.rpy
        self._draw_tripod(cam)

    def _draw_tripod(self, cam):
        """El trípode queda vertical bajo el tornillo de la cámara, así que va en el mundo y no bajo el gizmo."""
        c = self.s.pybullet_client
        T = np.eye(4)
        T[:3, :3] = np.array(c.getMatrixFromQuaternion(c.getQuaternionFromEuler(cam.rpy))).reshape(3, 3)
        T[:3, 3] = cam.xyz
        mesh = tripod_mesh(T)
        if mesh is None:
            self.v.scene.remove_by_name(f"/tripods/{cam.name}")
        else:
            self.v.scene.add_mesh_simple(f"/tripods/{cam.name}", *mesh, color=TRIPOD_COLOR)

    def _save_cam_poses(self):
        save_camera_poses(self.base_cams)

    # ── oclusores (hojas recortadas del splat) ───────────────────────────────
    def _build_occluder_gui(self):
        self._occ_gizmos = {}
        self._saved_leaves = {p.stem: p for p in saved_leaves(self.s.robot_name)}
        with self.v.gui.add_folder("Occluders"):
            self._occ_saved = self.v.gui.add_dropdown(
                "Saved leaf", list(self._saved_leaves) or ["-"],
                hint="Leaves cut with scripts/xarm5/segment_leaf.py")
            add = self.v.gui.add_button("Add to Scene", hint="The same leaf can be added several times")
            self._occ_select = self.v.gui.add_dropdown("Occluder", ["-"])
            self._occ_enabled = self.v.gui.add_checkbox("Enabled", True)
            self._occ_xyz = self.v.gui.add_vector3("xyz (m)", (0.0, 0.0, 0.0), step=0.001)
            self._occ_camera = self.v.gui.add_dropdown("Camera", [c.name for c in self.base_cams])
            self._occ_fraction = self.v.gui.add_slider(
                "Tomato → camera", 0.0, 1.0, 0.01, 0.2,
                hint="Where on the line from the tomato to the camera: 0 = at the tomato, 1 = at the camera")
            between = self.v.gui.add_button("Put Between Camera and Tomato")
            self._occ_target = self.v.gui.add_slider(
                "Target occlusion (%)", 0.0, 100.0, 1.0, 40.0,
                hint="For Fit to Target: slides the leaf sideways, at the distance of 'Tomato → camera', "
                     "until it covers this much of the tomato in the chosen camera")
            fit = self.v.gui.add_button("Fit to Target")
            remove = self.v.gui.add_button("Remove from Scene", color="red")
            save = self.v.gui.add_button("Save Occluders", hint=f"Writes {self.s.occluders.path}")
            self._occ_info = self.v.gui.add_markdown("")
        self._on_click(add, self._add_occluder)
        self._on_click(between, self._occluder_between)
        self._on_click(fit, self._fit_occluder)
        self._on_click(remove, self._remove_occluder)
        self._on_click(save, self._save_occluders)
        self._on_user(self._occ_select, lambda _: self._sync_occluder_ui())
        self._on_user(self._occ_enabled, lambda h: self._occ_current() and
                      setattr(self._occ_current(), "enabled", bool(h.value)))
        self._on_user(self._occ_xyz, lambda h: self._move_occluder(position=h.value))
        for item in self.s.occluders.items.values():
            self._add_occluder_gizmo(item)
        self._refresh_occluder_list()

    def _occ_current(self):
        return self.s.occluders.items.get(self._occ_select.value)

    def _add_occluder_gizmo(self, item):
        gizmo = self.v.scene.add_transform_controls(
            f"/occluders/{item.name}", scale=0.12, position=item.position, wxyz=item.wxyz)
        self.v.scene.add_label(f"/occluders/{item.name}/label", item.name)
        gizmo.on_update(lambda _, n=item.name: self._do(self._occluder_from_gizmo, n))
        self._occ_gizmos[item.name] = gizmo

    def _occluder_from_gizmo(self, name):
        gizmo = self._occ_gizmos[name]
        self.s.occluders.set_pose(name, tuple(float(v) for v in gizmo.position),
                                  tuple(float(v) for v in gizmo.wxyz))
        if name == self._occ_select.value:
            self._occ_xyz.value = tuple(float(v) for v in gizmo.position)

    def _move_occluder(self, position):
        item = self._occ_current()
        if item is None:
            return
        self.s.occluders.set_pose(item.name, tuple(float(v) for v in position), item.wxyz)
        self._occ_gizmos[item.name].position = item.position
        self._occ_xyz.value = item.position

    def _occluder_between(self):
        cam = next(c for c in self.base_cams if c.name == self._occ_camera.value)
        self._move_occluder(place_between(cam.xyz, self._apple_pos(), self._occ_fraction.value))

    def _fit_occluder(self):
        item = self._occ_current()
        if item is None:
            self._occ_info.content = "Add a leaf first."
            return
        cam = next(c for c in self.base_cams if c.name == self._occ_camera.value)
        center = place_between(cam.xyz, self._apple_pos(), self._occ_fraction.value)
        client = self.s.pybullet_client
        right = np.reshape(client.getMatrixFromQuaternion(client.getQuaternionFromEuler(cam.rpy)), (3, 3))[:, 0]
        item.enabled = self._occ_enabled.value = True

        def occlusion_at(offset):
            self.s.occluders.set_pose(item.name, center + offset * right, item.wxyz)
            return self.s.measure_occlusion([cam.name])[cam.name]["occlusion"]

        fit = fit_lateral_offset(occlusion_at, float(self._occ_target.value))
        self._move_occluder(center + fit["offset"] * right)
        self._occ_info.content = fit_message(fit, cam.name)
        self._measure_occlusion()

    def _build_occlusion_gui(self):
        with self.v.gui.add_folder("Occlusion"):
            button = self.v.gui.add_button(
                "Measure Occlusion", hint="How much of the tomato is covered in each camera, with the "
                                          "plant, the arm and the occluders as they are now")
            self._occlusion_text = self.v.gui.add_markdown("")
        self._on_click(button, self._measure_occlusion)

    def _measure_occlusion(self):
        lines = []
        for name, r in self.s.measure_occlusion().items():
            text = "not in view" if r["occlusion"] is None else f"**{r['occlusion']:.0f}%**"
            lines.append(f"{name}: {text}" + (" (partly out of frame)" if r["cut_by_frame"] else ""))
        self._occlusion_text.content = "  \n".join(lines) or "No cameras."

    def _add_occluder(self):
        ply = self._saved_leaves.get(self._occ_saved.value)
        if ply is None:
            self._occ_info.content = "No saved leaves for this scene: cut one with `segment_leaf.py`."
            return
        item = self.s.occluders.add(self.s.occluders.free_name(ply.stem), str(ply))
        self._add_occluder_gizmo(item)
        self._refresh_occluder_list(select=item.name)

    def _remove_occluder(self):
        item = self._occ_current()
        if item is None:
            return
        self.s.occluders.remove(item.name)
        self._occ_gizmos.pop(item.name).remove()
        self._refresh_occluder_list()

    def _save_occluders(self):
        self.s.occluders.save()
        self._occ_info.content = f"Saved {len(self.s.occluders.items)} occluders."

    def _refresh_occluder_list(self, select=None):
        names = list(self.s.occluders.items)
        self._occ_select.options = names or ["-"]
        self._occ_select.value = select or (names[0] if names else "-")
        self._sync_occluder_ui()

    def _sync_occluder_ui(self):
        item = self._occ_current()
        if item is not None:
            self._occ_enabled.value = item.enabled
            self._occ_xyz.value = item.position

    # ── manzana ──────────────────────────────────────────────────────────────
    def _build_apple_gui(self):
        apple = self.s.apple
        pos = self._apple_pos()
        self._grid_positions = []
        self._grid_handles = []
        self._grid_note = ""
        with self.v.gui.add_folder("Tomato"):
            if self.s.peduncle is not None:
                self._peduncle_text = self.v.gui.add_markdown("")
            self._apple_xyz = self.v.gui.add_vector3("xyz (m)", pos, step=0.001)
            rand = self.v.gui.add_button("Random Position")
            self._grid_which = self.v.gui.add_dropdown(
                "Grid", list(GRIDS), initial_value="custom",
                hint=f"custom: {APPLE_GRID_PATH} (editable). eval 4x4: {EVAL_GRID_PATH}, the positions "
                     "run_policy_trials.py evaluates (numbers = pos_idx; read-only)")
            self._grid_save = self.v.gui.add_button("Save Position to Grid")
            self._grid_info = self.v.gui.add_markdown("")
            # cargar una posición guardada (también se puede hacer clic en su esfera)
            self._grid_select = self.v.gui.add_dropdown("Grid position", ["-"])
            load = self.v.gui.add_button("Load Grid Position")
            self._grid_remove = remove = self.v.gui.add_button(
                "Remove Grid Position", color="red",
                hint="Removes the selected position from the grid file; the later ones move down one "
                     "number (trials already run keep their old numbers)")
            self._grid_confirm = self.v.gui.add_button_group("Confirm", ["Remove", "Cancel"], visible=False)
            self._build_range_gui()
        self._apple_gizmo = self.v.scene.add_transform_controls(
            "/apple", scale=0.12, position=pos, disable_rotations=True)
        self._fruit_center, radius = fruit_sphere(self.s.object_config[APPLE_NAME])
        self._fruit = self.v.scene.add_icosphere("/apple/sphere", radius=radius, color=(230, 40, 30))
        self._draw_fruit()
        self._stem = None
        if self.s.peduncle is not None:
            # el tallo va en el marco del mundo (el gizmo del tomate no gira), con su punto de corte
            p = self.s.peduncle
            self._stem = self.v.scene.add_cylinder(
                "/peduncle", radius=max(p.stem.get("radius", 0.0025), 0.002), height=0.05,
                color=STEM_COLORS[INTACT], radial_segments=12)
            self._stem_cut = self.v.scene.add_icosphere("/peduncle/cut", radius=0.004, color=CUT_COLOR)
            self._draw_stem()

        self._on_user(self._apple_xyz, lambda h: self._move_apple(h.value))
        self._apple_gizmo.on_update(
            lambda _: self._do(self._move_apple, tuple(float(v) for v in self._apple_gizmo.position)))
        self._on_click(rand, self._random_apple)
        self._on_click(self._grid_save, self._save_apple)
        self._on_user(self._grid_which, lambda _: self._switch_grid())
        self._on_user(self._grid_select, lambda _: self._load_grid_position())
        self._on_click(load, self._load_grid_position)
        self._on_click(remove, self._ask_remove_grid)
        self._grid_confirm.on_click(lambda e: self._do(self._remove_grid, e.target.value == "Remove"))
        self._show_grid()
        self._draw_range()                          # cuenta las posiciones dentro del hemisferio

    # ── rango de la posición aleatoria ───────────────────────────────────────
    def _build_range_gui(self):
        """Dónde "Random Position" pone el tomate: una caja (esquinas editables con números o con
        dos gizmos) o un hemisferio alrededor del hombro, al alcance del brazo (centro con un
        gizmo); Save Range guarda ambos y la forma elegida en APPLE_GRID_PATH."""
        apple = self.s.apple
        lo, hi = apple.range_corners()
        h = apple.hemisphere
        with self.v.gui.add_folder("Random range", expand_by_default=False):
            self._range_shape = self.v.gui.add_dropdown(
                "Shape", list(RANGE_SHAPES), initial_value=apple.shape,
                hint="box: min / max per axis. hemisphere: half shell around the shoulder, facing the "
                     "plant, so the tomato stays within the arm's reach")
            self._range_min = self.v.gui.add_vector3("Min (m)", lo, step=0.01)
            self._range_max = self.v.gui.add_vector3("Max (m)", hi, step=0.01)
            self._hemi_center = self.v.gui.add_vector3(
                "Center (m)", h["center"], step=0.01, hint="Default: the shoulder (joint2), 0.267 m above the base")
            self._hemi_radius = self.v.gui.add_vector2(
                "Radius min/max (m)", h["radius"], step=0.01,
                hint="Distance from the center; the arm reaches ~0.93 m from the shoulder with the grasp point")
            self._hemi_facing = self.v.gui.add_slider(
                "Facing (deg)", -180, 180, 5, h["facing"],
                hint="Horizontal direction of the hemisphere's axis: 180 = -x (towards the plant)")
            self._hemi_z = self.v.gui.add_vector2("Z min/max (m)", h["z"], step=0.01,
                                                  hint="Cuts the hemisphere at these heights")
            self._hemi_x = self.v.gui.add_vector2(
                "X min/max (m)", h["x"], step=0.01,
                hint="Cuts the hemisphere along x (e.g. keep the tomato away from the robot or the wall); "
                     "±1.5 = no cut")
            self._hemi_y = self.v.gui.add_vector2(
                "Y min/max (m)", h["y"], step=0.01,
                hint="Cuts the hemisphere sideways (e.g. keep the tomato in front of the plant); ±1.5 = no cut")
            self._range_show = self.v.gui.add_checkbox("Show range", False, hint="The range in the 3D view")
            self._range_edit = self.v.gui.add_checkbox(
                "Gizmos", False, hint="Drag the box corners or the hemisphere center in the 3D view (with Show range)")
            save = self.v.gui.add_button("Save Range")
            self._range_info = self.v.gui.add_markdown("")
        self._range_gizmos = [
            self.v.scene.add_transform_controls(f"/apple_range/{name}", scale=0.08,
                                                disable_rotations=True, position=corner)
            for name, corner in (("min", lo), ("max", hi))]
        self._hemi_gizmo = self.v.scene.add_transform_controls(
            "/apple_range/center", scale=0.1, disable_rotations=True, position=h["center"])
        self._on_user(self._range_shape, lambda hd: self._set_shape(hd.value))
        self._on_user(self._range_min, lambda _: self._set_range(self._range_min.value, self._range_max.value))
        self._on_user(self._range_max, lambda _: self._set_range(self._range_min.value, self._range_max.value))
        for gizmo in self._range_gizmos:
            gizmo.on_update(lambda _: self._do(self._set_range, *(g.position for g in self._range_gizmos)))
        for handle in (self._hemi_center, self._hemi_radius, self._hemi_facing, self._hemi_z, self._hemi_x,
                       self._hemi_y):
            self._on_user(handle, lambda _: self._set_hemisphere())
        self._hemi_gizmo.on_update(
            lambda _: self._do(self._set_hemisphere, tuple(float(v) for v in self._hemi_gizmo.position)))
        self._range_show.on_update(lambda _: self._draw_range())
        self._range_edit.on_update(lambda _: self._draw_range())
        self._on_click(save, self._save_range)
        self._range_saved = self._range_state()
        self._draw_range()

    def _range_state(self):
        a = self.s.apple
        return a.shape, dict(a.range), dict(a.hemisphere)

    def _set_shape(self, shape):
        self.s.apple.shape = shape
        self._draw_range()

    def _set_range(self, lo, hi):
        self.s.apple.set_range(lo, hi)
        lo, hi = self.s.apple.range_corners()        # ordenadas y con lado mínimo
        self._range_min.value, self._range_max.value = lo, hi
        for gizmo, corner in zip(self._range_gizmos, (lo, hi)):
            if np.abs(np.subtract(gizmo.position, corner)).max() > 1e-4:
                gizmo.position = corner             # no se pisa al gizmo mientras se arrastra
        self._draw_range()

    def _set_hemisphere(self, center=None):
        apple = self.s.apple
        apple.set_hemisphere(center or self._hemi_center.value, self._hemi_radius.value,
                             self._hemi_facing.value, self._hemi_z.value, self._hemi_x.value,
                             self._hemi_y.value)
        h = apple.hemisphere                         # ordenado y válido
        self._hemi_center.value, self._hemi_radius.value = h["center"], h["radius"]
        self._hemi_facing.value, self._hemi_z.value = h["facing"], h["z"]
        self._hemi_x.value, self._hemi_y.value = h["x"], h["y"]
        if np.abs(np.subtract(self._hemi_gizmo.position, h["center"])).max() > 1e-4:
            self._hemi_gizmo.position = h["center"]
        self._draw_range()

    def _save_range(self):
        self.s.apple.save_range()
        self._range_saved = self._range_state()
        self._draw_range()

    def _draw_range(self):
        apple = self.s.apple
        box = apple.shape == "box"
        show = self._range_show.value
        for handle in (self._range_min, self._range_max):
            handle.visible = box
        for handle in (self._hemi_center, self._hemi_radius, self._hemi_facing, self._hemi_z, self._hemi_x,
                       self._hemi_y):
            handle.visible = not box
        lo, hi = (np.array(c) for c in apple.range_corners())
        center, size = tuple((lo + hi) / 2), tuple(hi - lo)
        # caja translúcida y su contorno, del color de las esferas de la grilla
        self.v.scene.add_box("/apple_range/box", color=RANGE_COLOR, dimensions=size, position=center,
                             opacity=0.08, visible=show and box)
        self.v.scene.add_box("/apple_range/edges", color=RANGE_COLOR, dimensions=size, position=center,
                             wireframe=True, visible=show and box)
        # hemisferio: meridianos y paralelos de sus dos cáscaras
        lines = hemisphere_lines(apple.hemisphere)
        self.v.scene.add_line_segments("/apple_range/hemisphere", lines if len(lines) else np.zeros((0, 2, 3)),
                                       RANGE_COLOR, thickness=1.5, thickness_units="screen",
                                       visible=show and not box)
        for gizmo in self._range_gizmos:
            gizmo.visible = show and box and self._range_edit.value
        self._hemi_gizmo.visible = show and not box and self._range_edit.value
        if box:
            text = ", ".join(f"{a} ∈ [{apple.range[a][0]:.2f}, {apple.range[a][1]:.2f}]" for a in "xyz") + " m"
        else:
            h = apple.hemisphere
            inside = int(in_hemisphere(self._grid_positions, h).sum()) if self._grid_positions else 0
            text = (f"r ∈ [{h['radius'][0]:.2f}, {h['radius'][1]:.2f}] m facing {h['facing']:.0f}°, "
                    f"z ∈ [{h['z'][0]:.2f}, {h['z'][1]:.2f}] m"
                    + (f", x ∈ [{h['x'][0]:.2f}, {h['x'][1]:.2f}] m" if tuple(h["x"]) != NO_X_LIMIT else "")
                    + (f", y ∈ [{h['y'][0]:.2f}, {h['y'][1]:.2f}] m" if tuple(h["y"]) != NO_Y_LIMIT else "")
                    + f"; {inside}/{len(self._grid_positions)} "
                    "grid positions inside")
        saved = self._range_state() == self._range_saved
        self._range_info.content = text + ("" if saved else "  \n**not saved**: Random Position "
                                           "uses it now; Save Range keeps it for next time")

    def _move_apple(self, xyz):
        self.s.apple.move_to(xyz)
        self._sync_apple_ui(self._apple_pos())

    def _sync_apple_ui(self, pos):
        self._apple_xyz.value = pos
        if np.abs(np.subtract(pos, self._apple_gizmo.position)).max() > 1e-3:
            self._apple_gizmo.position = pos  # no se pisa al gizmo mientras se arrastra

    def _draw_fruit(self):
        """La esfera del tomate en su centro real: el origen del cuerpo está en su base, no en el
        centro. Es hija del gizmo (que no gira), así que el desplazamiento se gira aquí."""
        c = self.s.pybullet_client
        _, quat = c.getBasePositionAndOrientation(self.s.apple.body_id)
        rot = np.reshape(c.getMatrixFromQuaternion(quat), (3, 3))
        self._fruit.position = rot @ self._fruit_center

    def _draw_stem(self):
        """El cilindro del tallo donde está ahora (sigue al tomate, también sujeto o al caer); el
        color dice si está entero, cortado y sujeto, o cortado y suelto."""
        p = self.s.peduncle
        bottom, top, cut = p.stem_world()
        axis = top - bottom
        height = float(np.linalg.norm(axis))
        if height < 1e-6:
            return
        axis /= height
        self._stem.position = (bottom + top) / 2
        self._stem.wxyz = z_to(axis)
        self._stem.height = height
        self._stem.color = STEM_COLORS[p.state]
        # el punto de corte en el marco del cilindro (z = su eje, origen en el centro)
        self._stem_cut.position = (0.0, 0.0, float(np.dot(cut - (bottom + top) / 2, axis)))

    def _apple_pos(self):
        pos, _ = self.s.pybullet_client.getBasePositionAndOrientation(self.s.apple.body_id)
        return tuple(round(float(v), 4) for v in pos)

    def _random_apple(self):
        try:
            self._move_apple(self.s.apple.random_position())
        except ValueError as e:                     # hemisferio vacío (radios o alturas)
            self._range_info.content = f"**Random Position**: {e}"

    def _save_apple(self):
        self.s.apple.save_to_grid()
        self._show_grid()
        self._draw_range()

    def _switch_grid(self):
        """Cambia la grilla mostrada; Save / Remove solo valen para la propia (custom)."""
        self._grid_note = ""
        self._grid_confirm.visible = False
        editable = self._grid_which.value == "custom"
        self._grid_save.disabled = self._grid_remove.disabled = not editable
        self._show_grid()
        self._grid_select.value = self._grid_select.options[0]
        self._draw_range()                          # el conteo de posiciones dentro del hemisferio

    def _show_grid(self):
        """Dibuja las posiciones de la grilla elegida (GRIDS) como esferas numeradas."""
        path, color = GRIDS[self._grid_which.value]
        positions = load_grid(path)
        self._grid_positions = [tuple(p) for p in positions]
        for handle in self._grid_handles:          # las de la grilla anterior, o las que sobran tras quitar una
            handle.remove()
        self._grid_handles = []
        for i, p in enumerate(self._grid_positions):
            sphere = self.v.scene.add_icosphere(f"/grid/{i}", radius=0.012, color=color,
                                                opacity=0.6, position=p)
            self.v.scene.add_label(f"/grid/{i}/label", str(i))
            sphere.on_click(lambda _, i=i: self._do(self._load_grid_position, i))
            self._grid_handles.append(sphere)
        self._grid_info.content = "  \n".join(
            t for t in (f"{len(positions)} positions in `{path}`", self._grid_note) if t)
        self._grid_select.options = [f"#{i} {p}" for i, p in enumerate(self._grid_positions)] or ["-"]

    def _selected_grid_index(self):
        if not self._grid_positions or self._grid_select.value not in self._grid_select.options:
            return None
        return self._grid_select.options.index(self._grid_select.value)

    def _ask_remove_grid(self):
        index = self._selected_grid_index()
        if index is None:
            return
        self._grid_note = f"**Remove #{index} {list(self._grid_positions[index])}?**"
        self._grid_confirm.visible = True
        self._show_grid()

    def _remove_grid(self, confirmed: bool):
        self._grid_confirm.visible = False
        index = self._selected_grid_index()
        self._grid_note = ""
        if confirmed and index is not None:
            removed = self.s.apple.remove_from_grid(index)
            if removed is not None:
                self._grid_note = f"Removed #{index} {removed}"
        self._show_grid()
        if self._grid_positions:
            self._grid_select.value = self._grid_select.options[min(index or 0, len(self._grid_positions) - 1)]
        self._draw_range()                          # el conteo de posiciones dentro del hemisferio

    def _load_grid_position(self, index=None):
        """Mueve la manzana a la posición `index` de la grilla (por defecto la elegida en la lista)."""
        if index is None:
            if not self._grid_positions:
                return
            index = self._grid_select.options.index(self._grid_select.value)
        self._grid_select.value = self._grid_select.options[index]
        self._move_apple(self._grid_positions[index])
