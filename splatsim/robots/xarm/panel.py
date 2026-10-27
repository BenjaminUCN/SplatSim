"""Panel de control web (viser): robot, grabación, cámaras base y manzana.

Abre http://localhost:<puerto> en el navegador. Los callbacks de viser corren en sus
propios hilos, así que solo encolan acciones; update() las ejecuta desde el serve_loop
con el lock de la simulación tomado (mismo patrón que las peticiones del recorder).
"""
import atexit
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import viser

from splatsim.robots.joint_controller import MODES
from splatsim.utils.camera_setup import load_camera_setup
from splatsim.utils.registry import load_registry, loadable_models, model_label

from .apple import load_grid
from .debug_ui import save_camera_poses
from .config import APPLE_GRID_PATH, BASE_CAM_FOV, IMAGE_H, IMAGE_W

STATUS_PERIOD = 0.2    # s entre refrescos de estado, pose del efector y de la manzana
PREVIEW_PERIOD = 0.5   # s entre renders de la vista previa (cuesta un render del splat)
PREVIEW_DEFAULT = ("base_rgb", "wrist_rgb")  # vistas activas al abrir el panel
PREVIEW_PANEL_WIDTH = 360
MAIN_PANEL_WIDTH = 380
STOP_TIMEOUT = 5.0     # s que se espera a que la política termine tras Ctrl+C antes de matarla


def _parse_indices(text: str, n: int) -> list[int] | None:
    """'0,2,5-7' -> [0, 2, 5, 6, 7]; vacío -> None (todas). Valida contra n posiciones."""
    text = text.strip()
    if not text:
        return None
    out = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        out += range(lo, hi + 1)
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"{bad} out of range (grid has {n} positions)")
    return out


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

        # panel principal a la derecha, en pestañas por tarea; vistas de cámara en
        # una ventana aparte a la izquierda (se puede mover, acoplar o minimizar)
        tabs = self.v.gui.add_tab_group()
        with tabs.add_tab("Run", viser.Icon.PLAYER_PLAY):
            self._build_robot_gui()
            self._build_policy_gui()
            self._build_recording_gui()
        with tabs.add_tab("Scene", viser.Icon.ADJUSTMENTS):
            self._build_apple_gui()
            self._build_camera_gui()
        self.v.gui.main_panel.dock_right()
        self.v.gui.main_panel.set_width(MAIN_PANEL_WIDTH)
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
            # cómo se siguen los objetivos articulares (ver splatsim/robots/joint_controller.py);
            # se puede cambiar con una política corriendo: retoma desde la posición actual
            mode = self.v.gui.add_dropdown("Control mode", list(MODES),
                                           initial_value=self.s.joint_ctrl.mode)
        self._on_user(mode, lambda h: self.s.set_control_mode(h.value))

    # ── política ─────────────────────────────────────────────────────────────
    def _build_policy_gui(self):
        """Corre un modelo de configs/registry.yaml con run_env_sim.py en otro proceso."""
        self._models = {}            # etiqueta del dropdown -> (carpeta, entrada del registro)
        self._policy_proc = None
        self._policy_label = None
        self._policy_trials = False
        self._trial_info = {}
        self._stop_deadline = None
        with self.v.gui.add_folder("Policy"):
            self._model_select = self.v.gui.add_dropdown("Model", ["-"])
            self._model_info = self.v.gui.add_markdown("")
            self._reset_first = self.v.gui.add_checkbox("Reset robot first", True)
            run = self.v.gui.add_button("Run", color="green")
            stop = self.v.gui.add_button("Stop")
            reload = self.v.gui.add_button("Reload Registry")
            self._policy_status = self.v.gui.add_markdown("No policy running")
            # ensayos automáticos sobre la grilla de la manzana (run_policy_trials.py)
            with self.v.gui.add_folder("Trials", expand_by_default=False):
                self._trials_per_pos = self.v.gui.add_number("Trials per position", 3, min=1, step=1)
                self._trial_seconds = self.v.gui.add_number("Trial seconds", 15.0, min=1.0, step=1.0)
                self._trial_positions = self.v.gui.add_text(
                    "Positions", "", hint="Grid indices, e.g. 0,2,5-7 (empty = all)")
                run_trials = self.v.gui.add_button("Run Trials", color="blue")
                self._trial_progress = self.v.gui.add_progress_bar(0.0, visible=False)
                self._trial_status = self.v.gui.add_markdown("")
        self._model_select.on_update(lambda _: self._show_model_info())
        self._on_click(run, self._run_policy)
        self._on_click(run_trials, self._run_trials)
        self._on_click(stop, self._stop_policy)
        reload.on_click(lambda _: self._load_models())
        self._load_models()
        atexit.register(self._kill_policy)

    def _load_models(self):
        registry = load_registry()
        self._datasets = registry.get("datasets") or {}
        models = sorted(loadable_models(registry).items(),
                        key=lambda kv: kv[1].get("started") or "", reverse=True)  # recientes primero
        self._models = {}
        for key, model in models:
            label = model_label(key, model)
            if label in self._models:      # nombres repetidos: se agrega la carpeta
                label = f"{label} ({key.split('/')[-1]})"
            self._models[label] = (key, model)
        self._model_select.options = list(self._models) or ["-"]
        self._show_model_info()

    def _missing_cameras(self, model) -> list:
        return [c for c in model.get("cameras", []) if c not in self.s.camera_names]

    def _show_model_info(self):
        if self._model_select.value not in self._models:
            self._model_info.content = "No models: run `python scripts/build_registry.py`"
            return
        key, m = self._models[self._model_select.value]
        ds = self._datasets.get(m.get("dataset")) or {}
        lines = [
            f"*{m['description']}*" if m.get("description") else "",
            f"`{key}`",
            f"cameras: {', '.join(m.get('cameras', []))}",
            f"dataset: `{m.get('dataset')}` ({ds.get('episodes', '?')} ep)",
            f"horizon {m.get('horizon')}, n_action_steps {m.get('n_action_steps')}, "
            f"n_obs_steps {m.get('n_obs_steps')}",
            m.get("notes") or "",
        ]
        missing = self._missing_cameras(m)
        if missing:
            lines.append(f"**⚠ the server does not render {', '.join(missing)}** (launch_nodes --cameras)")
        self._model_info.content = "  \n".join(line for line in lines if line)

    def _selected_model(self):
        """(etiqueta, modelo) elegido, o None si no se puede correr (avisa en el estado)."""
        if self._policy_proc is not None:
            self._policy_status.content = f"**Already running** {self._policy_label}: press Stop first"
            return None
        if self._model_select.value not in self._models:
            return None
        label = self._model_select.value
        _, model = self._models[label]
        missing = self._missing_cameras(model)
        if missing:
            self._policy_status.content = f"**Cannot run**: the server does not render {', '.join(missing)}"
            return None
        return label, model

    def _run_policy(self):
        selected = self._selected_model()
        if selected is None:
            return
        label, model = selected
        if self._reset_first.value:
            self.s.reset_robot()
        self._start_client(
            ["scripts/run_env_sim.py", "--agent", "lerobot_diffusion",
             "--hz", str(self.s.control_hz), "--robot_port", str(self.s.zmq_port),
             "--checkpoint", model["path"]],
            label, trials=False)

    def _run_trials(self):
        selected = self._selected_model()
        if selected is None:
            return
        label, model = selected
        grid = load_grid()
        try:
            positions = _parse_indices(self._trial_positions.value, len(grid))
        except ValueError as e:
            self._trial_status.content = f"**Positions**: {e}"
            return
        if not grid:
            self._trial_status.content = "**The grid is empty**: save positions in the Scene tab"
            return
        cmd = ["scripts/run_policy_trials.py", "--checkpoint", model["path"], "--run_name", label,
               "--robot_port", str(self.s.zmq_port),
               "--trials_per_position", str(int(self._trials_per_pos.value)),
               "--trial_seconds", str(float(self._trial_seconds.value))]
        if positions is not None:
            cmd += ["--positions", *map(str, positions)]
        n = len(positions if positions is not None else grid) * int(self._trials_per_pos.value)
        self._trial_info = {"done": 0, "total": n, "last": "", "root": None}
        self._trial_progress.value, self._trial_progress.visible = 0.0, True
        self._trial_status.content = f"0/{n} trials; loading the model..."
        self._start_client(cmd, label, trials=True)

    def _start_client(self, args, label, trials: bool):
        """Lanza un cliente (política o ensayos) en otro proceso; su salida va a esta terminal."""
        cmd = [sys.executable, *args]
        print(f"[Panel] {' '.join(cmd)}")
        if trials:
            # se lee su salida para mostrar el avance; sin buffer para que llegue al momento
            self._policy_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            threading.Thread(target=self._read_trials_output, args=(self._policy_proc,),
                             daemon=True).start()
        else:
            self._policy_proc = subprocess.Popen(cmd)
        self._policy_label = label
        self._policy_trials = trials
        self._stop_deadline = None
        what = "trials of " if trials else ""
        self._policy_status.content = (f"**▶ Running** {what}{label} (pid {self._policy_proc.pid}); "
                                       "loading takes ~10 s")

    def _read_trials_output(self, proc):
        """Hilo: reenvía la salida de run_policy_trials.py y extrae el avance."""
        info = self._trial_info
        for line in proc.stdout:
            sys.stdout.write(line)
            if m := re.search(r"\[Trials\] (\d+)/(\d+)", line):
                info["done"] = int(m.group(1)) - 1     # el ensayo m.group(1) recién empieza
                info["last"] = line.split("— ", 1)[-1].strip()
            elif m := re.search(r"\[Trials\] (\d+) frames, dist final ([\d.]+) m, mín ([\d.]+) m", line):
                info["done"] += 1
                info["result"] = f"last: final {float(m.group(2)):.3f} m, min {float(m.group(3)):.3f} m"
            elif m := re.search(r"\[Trials\] Resultados en (.+)", line):
                info["root"] = m.group(1).strip()

    def _stop_policy(self):
        if self._policy_proc is None:
            return
        self._policy_proc.send_signal(signal.SIGINT)   # como Ctrl+C: los scripts cierran limpio
        self._stop_deadline = time.monotonic() + STOP_TIMEOUT
        self._policy_status.content = f"Stopping {self._policy_label}..."

    def _poll_policy(self):
        proc = self._policy_proc
        if proc is None:
            return
        if self._policy_trials:
            self._show_trials_progress()
        code = proc.poll()
        if code is None:
            if self._stop_deadline is not None and time.monotonic() > self._stop_deadline:
                proc.kill()
            return
        if self._stop_deadline is not None:
            self._policy_status.content = f"Stopped {self._policy_label}"
        elif code == 0:
            self._policy_status.content = f"Finished {self._policy_label}"
        else:
            self._policy_status.content = f"**{self._policy_label} failed** (code {code}); see the terminal"
        if self._policy_trials:
            self._show_trials_progress()
        self._policy_proc = None

    def _show_trials_progress(self):
        info = self._trial_info
        done, total = info["done"], info["total"]
        self._trial_progress.value = 100.0 * done / max(total, 1)
        lines = [f"{done}/{total} trials" + (f" — now {info['last']}" if info["last"] and done < total else "")]
        if info.get("result"):
            lines.append(info["result"])
        if info["root"]:
            lines.append(f"results in `{info['root']}`")
        self._trial_status.content = "  \n".join(lines)

    def _kill_policy(self):
        if self._policy_proc is not None and self._policy_proc.poll() is None:
            self._policy_proc.kill()

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
        # (salvo que un cliente las esté pidiendo: entonces se reusan sus observaciones)
        self._previews = {}
        panel = self.v.gui.add_panel()
        arrangements = load_camera_setup()["arrangements"]
        with panel.add_tab("Cameras", viser.Icon.CAMERA):
            # un arreglo (configs/camera_poses.yaml) marca de una vez sus cámaras
            arrangement = self.v.gui.add_dropdown(
                "Arrangement", ["custom", *arrangements], initial_value="custom")
            checks = {name: self.v.gui.add_checkbox(name, name in PREVIEW_DEFAULT)
                      for name in self.s.cameras}
            for name, show in checks.items():
                image = self.v.gui.add_image(
                    np.zeros((IMAGE_H // 2, IMAGE_W // 2, 3), np.uint8), label=name,
                    format="jpeg", visible=show.value)
                show.on_update(lambda _, s=show, i=image: setattr(i, "visible", s.value))
                self._previews[name] = (show, image)
        panel.dock_left()
        panel.set_width(PREVIEW_PANEL_WIDTH)

        @arrangement.on_update
        def _(event):
            if event.client is None or arrangement.value == "custom":
                return
            selected = arrangements[arrangement.value]
            for name, check in checks.items():
                check.value = name in selected

    def _update_preview(self):
        names = [name for name, (show, _) in self._previews.items() if show.value]
        if not names:
            return
        images = self.s.recent_images(max_age=2 * PREVIEW_PERIOD)
        to_render = [n for n in names if n not in images]
        if to_render:
            images.update(self.s.render_preview(to_render))
        for name in names:
            img = images.get(name)
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
        self._poll_policy()
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
