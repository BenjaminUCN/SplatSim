"""Panel de control web (viser): robot, grabación, cámaras base y manzana.

Abre http://localhost:<puerto> en el navegador. Los callbacks de viser corren en sus
propios hilos, así que solo encolan acciones; update() las ejecuta desde el serve_loop
con el lock de la simulación tomado (mismo patrón que las peticiones del recorder).
"""
import queue
import time

import numpy as np
import viser

from .apple import load_grid
from .debug_ui import save_camera_poses
from .config import APPLE_GRID_PATH, BASE_CAM_FOV, IMAGE_H, IMAGE_W

STATUS_PERIOD = 0.2    # s entre refrescos de estado, pose del efector y de la manzana
PREVIEW_PERIOD = 0.5   # s entre renders de la vista previa (cuesta un render del splat)


def _xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _wxyz_to_xyzw(q):
    return (q[1], q[2], q[3], q[0])


class ControlPanel:
    def __init__(self, server, base_cams, port: int = 8080):
        self.s = server
        self.base_cams = base_cams
        self._actions = queue.SimpleQueue()
        self._last_status = 0.0
        self._last_preview = 0.0

        self.v = viser.ViserServer(port=port, label="SplatSim xArm5", verbose=False)
        self.v.scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)
        self.v.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1, section_size=0.5)
        self._ee = self.v.scene.add_frame("/ee", axes_length=0.06, axes_radius=0.004)

        self._build_robot_gui()
        self._build_recording_gui()
        self._build_camera_gui()
        self._build_apple_gui()
        self._build_preview_gui()
        print(f"[Panel] Panel de control en http://localhost:{port}")

    # ── utilidades ───────────────────────────────────────────────────────────
    def _do(self, fn, *args):
        """Encola fn(*args) para ejecutarla en el serve_loop."""
        self._actions.put((fn, args))

    def _on_user(self, handle, fn):
        """Registra fn(handle) para cambios hechos por el usuario (no por código)."""
        @handle.on_update
        def _(event):
            if event.client is not None:
                self._do(fn, handle)

    def _on_click(self, button, fn):
        button.on_click(lambda _: self._do(fn))

    # ── robot ────────────────────────────────────────────────────────────────
    def _build_robot_gui(self):
        with self.v.gui.add_folder("Robot"):
            self._on_click(self.v.gui.add_button("Reset Robot Joints"), self.s.reset_robot)

    # ── grabación ────────────────────────────────────────────────────────────
    def _build_recording_gui(self):
        rec = self.s.recorder
        with self.v.gui.add_folder("Recording"):
            self._rec_status = self.v.gui.add_markdown("")
            start = self.v.gui.add_button("Start", color="red")
            stop = self.v.gui.add_button("Stop")
        # mismo camino que el teclado: tick() atiende la petición
        start.on_click(lambda _: setattr(rec, "_request", "start"))
        stop.on_click(lambda _: setattr(rec, "_request", "stop"))

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
        gizmo.wxyz = _xyzw_to_wxyz(self.s.pybullet_client.getQuaternionFromEuler(cam.rpy))

    def _cam_from_gizmo(self, cam):
        _, _, gizmo = self._cam_ui[cam.name]
        rpy = self.s.pybullet_client.getEulerFromQuaternion(_wxyz_to_xyzw(gizmo.wxyz))
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

    # ── vista previa ─────────────────────────────────────────────────────────
    def _build_preview_gui(self):
        # una casilla y una imagen por cámara; cada vista activa cuesta un render
        self._previews = {}
        with self.v.gui.add_folder("Camera previews"):
            for name in self.s.cameras:
                show = self.v.gui.add_checkbox(name, False)
                image = self.v.gui.add_image(
                    np.zeros((IMAGE_H // 2, IMAGE_W // 2, 3), np.uint8), format="jpeg", visible=False)
                show.on_update(lambda _, s=show, i=image: setattr(i, "visible", s.value))
                self._previews[name] = (show, image)

    def _update_preview(self):
        names = [name for name, (show, _) in self._previews.items() if show.value]
        if not names:
            return
        for name, img in self.s.render_preview(names).items():
            if img is None:
                continue
            img = (np.clip(img.detach().permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            self._previews[name][1].image = img[::2, ::2]

    # ── loop ─────────────────────────────────────────────────────────────────
    def update(self) -> None:
        """Llamar cada tick del serve_loop, con el lock tomado."""
        while not self._actions.empty():
            fn, args = self._actions.get()
            fn(*args)

        now = time.monotonic()
        if now - self._last_status > STATUS_PERIOD:
            self._last_status = now
            self._update_status()
        if now - self._last_preview > PREVIEW_PERIOD:
            self._last_preview = now
            self._update_preview()

    def _update_status(self):
        rec = self.s.recorder
        self._rec_status.content = (
            f"**● Recording** — {rec._length} frames in `{rec._folder}`" if rec.active
            else "Idle (also `r`/`s` + Enter in the terminal)")

        pos, quat = self.s.get_current_ee_pose()
        self._ee.position, self._ee.wxyz = pos, _xyzw_to_wxyz(quat)

        # la manzana también se mueve desde fuera (p.ej. run_policy_trials.py)
        pos = self._apple_pos()
        if np.abs(np.subtract(pos, self._apple_xyz.value)).max() > 1e-3:
            self._sync_apple_ui(pos)
