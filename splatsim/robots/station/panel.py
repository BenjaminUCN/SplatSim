"""Panel de control web (viser) compartido por el sim y el xArm5 real: robot, política (Run,
Record as trial, Trials), grabación y vistas de cámara.

Abre http://localhost:<puerto> en el navegador. Los callbacks de viser corren en sus propios
hilos, así que solo encolan acciones; update() las ejecuta desde el lazo del servidor con su lock
tomado (mismo patrón que las peticiones del recorder).

El servidor (backend) expone:
  recorder, control_hz, zmq_port, camera_names (las que ve la política)
  reset_robot(), control_mode, set_control_mode(mode), get_current_ee_pose() -> (pos, xyzw) | None
  preview_cameras, preview_images(names, max_age) -> {nombre: uint8 HxWx3}
Las subclases agregan lo suyo con _build_robot_extra, _build_tabs y _update_status_extra
(SimPanel en splatsim/robots/xarm/panel.py: escena, tomate, cámaras base, sim clock).
"""
import atexit
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

from splatsim.robots.joint_controller import MODES
from splatsim.utils.apple_grid import load_grid
from splatsim.utils.camera_setup import load_camera_setup
from splatsim.utils.registry import (
    dataset_for_checkpoint,
    describe_dataset,
    describe_model,
    load_local_models,
    load_registry,
    loadable_models,
    model_label,
    save_local_model,
)

from splatsim.utils.viser_page import link_html, set_page_title

LOCAL_SUFFIX = " · local"   # modelos cargados con "Load model" (configs/local_models.yaml)

STATUS_PERIOD = 0.2    # s entre refrescos de estado y pose del efector
PREVIEW_PERIOD = 0.5   # s entre vistas previas (en el sim cuesta un render del splat)
PREVIEW_DEFAULT = ("base_rgb", "wrist_rgb")  # vistas activas al abrir el panel
PREVIEW_H, PREVIEW_W = 240, 320              # tamaño de las vistas (la mitad de la imagen)
PREVIEW_PANEL_WIDTH = 360
MAIN_PANEL_WIDTH = 380
STOP_TIMEOUT = 5.0     # s que se espera a que la política termine tras Ctrl+C antes de matarla
TOOLS_PERIOD = 1.0     # s entre revisiones del estado de las herramientas
REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_LOG_DIR = Path("~/.cache/splatsim/tools").expanduser()
# herramientas con su propio panel viser que se pueden lanzar desde la pestaña Tools
TOOLS = (
    ("Recordings review", "scripts/xarm5/review_recordings.py", 8081,
     "GELLO recordings: trails, episode selection and stop labels"),
    ("Training queue", "scripts/xarm5/train_server.py", 8082,
     "Queue trainings; stopping this panel does not stop a training in progress"),
)


def parse_indices(text: str, n: int) -> list[int] | None:
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


def xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def wxyz_to_xyzw(q):
    return (q[1], q[2], q[3], q[0])


class StationPanel:
    def __init__(self, server, port: int = 8080, label: str = "SplatSim xArm5"):
        self.s = server
        self._actions = queue.SimpleQueue()
        self._last_status = 0.0
        self._last_preview = 0.0

        self.v = viser.ViserServer(port=port, label=label, verbose=False)
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
        self._build_tabs(tabs)
        with tabs.add_tab("Tools", viser.Icon.TOOL):
            self._build_tools_gui()
        set_page_title(self.v, label)
        self.v.gui.main_panel.dock_right()
        self.v.gui.main_panel.set_width(MAIN_PANEL_WIDTH)
        self._build_preview_gui()
        print(f"[Panel] Panel de control en http://localhost:{port}")

    def _build_tabs(self, tabs):
        """Pestañas propias de cada backend, después de Run."""

    # ── herramientas (revisor de grabaciones, cola de entrenamiento) ─────────
    def _build_tools_gui(self):
        self._tools = {}
        self._last_tools = 0.0
        for name, script, port, about in TOOLS:
            with self.v.gui.add_folder(name):
                self.v.gui.add_markdown(f"*{about}*")
                status = self.v.gui.add_html("")
                start = self.v.gui.add_button("Start", color="green")
                stop = self.v.gui.add_button("Stop")
            self._tools[name] = {"script": script, "port": port, "proc": None, "status": status}
            self._on_click(start, lambda n=name: self._start_tool(n))
            self._on_click(stop, lambda n=name: self._stop_tool(n))
        self._update_tools()

    @staticmethod
    def _port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.05)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _start_tool(self, name: str):
        tool = self._tools[name]
        if self._port_open(tool["port"]):
            self._update_tools()            # ya corre (de aquí o de otra terminal): solo el enlace
            return
        TOOLS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(TOOLS_LOG_DIR / f"{Path(tool['script']).stem}.log", "w")
        cmd = [sys.executable, "-u", tool["script"], "--port", str(tool["port"])]
        print(f"[Panel] {' '.join(cmd)} (salida en {log.name})")
        # sesión propia: sigue corriendo si se detiene el servidor del robot (Ctrl+C)
        tool["proc"] = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        tool["log"] = log.name
        self._update_tools()

    def _stop_tool(self, name: str):
        tool = self._tools[name]
        proc = tool["proc"]
        if proc is None or proc.poll() is not None:
            return                        # no lo lanzó este panel (el estado lo dice)
        os.killpg(proc.pid, signal.SIGINT)    # como Ctrl+C
        try:
            proc.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        tool["proc"] = None
        self._update_tools()

    def _update_tools(self):
        for name, tool in self._tools.items():
            proc, port = tool["proc"], tool["port"]
            ours = proc is not None and proc.poll() is None
            if self._port_open(port):
                who = f"pid {proc.pid}" if ours else "started elsewhere: stop it from its terminal"
                text = f"● Running ({who}) — {link_html(port, f'open :{port}')}"
            elif ours:
                text = "Starting..."
            elif proc is not None:
                text = f"Exited (code {proc.returncode}); see <code>{tool.get('log')}</code>"
            else:
                text = "Not running"
            tool["status"].content = text

    # ── utilidades ───────────────────────────────────────────────────────────
    def _do(self, fn, *args):
        """Encola fn(*args) para ejecutarla en el lazo del servidor."""
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
                                           initial_value=self.s.control_mode)
            self._build_robot_extra()
        self._on_user(mode, lambda h: self.s.set_control_mode(h.value))

    def _build_robot_extra(self):
        """Controles del robot propios de cada backend (dentro de la carpeta Robot)."""

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
            self._as_trial = self.v.gui.add_checkbox(
                "Record as trial", False,
                hint="Run one trial at the current tomato position (not moved): recording starts "
                     "when the policy starts acting and lasts Trial seconds (Trials folder); "
                     "saved with the model's trials")
            run = self.v.gui.add_button("Run", color="green")
            stop = self.v.gui.add_button("Stop")
            reload = self.v.gui.add_button("Reload Registry")
            self._policy_status = self.v.gui.add_markdown("No policy running")
            # preguntas de run_policy_trials.py al operador (en el real: poner el tomate, resultado)
            self._op_folder = self.v.gui.add_folder("Operator", visible=False)
            with self._op_folder:
                self._op_text = self.v.gui.add_markdown("")
            self._op_buttons = None
            self._op_shown = None
            # ensayos automáticos sobre la grilla de la manzana (run_policy_trials.py)
            with self.v.gui.add_folder("Trials", expand_by_default=False):
                self._trials_per_pos = self.v.gui.add_number("Trials per position", 3, min=1, step=1)
                self._trial_seconds = self.v.gui.add_number("Trial seconds", 15.0, min=1.0, step=1.0)
                self._trial_positions = self.v.gui.add_text(
                    "Positions", "", hint="Grid indices, e.g. 0,2,5-7 (empty = all)")
                run_trials = self.v.gui.add_button("Run Trials", color="blue")
                self._trial_progress = self.v.gui.add_progress_bar(0.0, visible=False)
                self._trial_status = self.v.gui.add_markdown("")
            # un modelo y su dataset por ruta, p.ej. en otra máquina donde el registro no sirve
            with self.v.gui.add_folder("Load model", expand_by_default=False):
                self._load_model_path = self.v.gui.add_text(
                    "Model folder", "", hint="Final folder of a trained model (config.json + model.safetensors)")
                self._load_dataset_path = self.v.gui.add_text(
                    "Dataset folder", "", hint="LeRobot dataset it was trained on (meta/info.json); "
                                               "empty = the one in configs/registry.yaml")
                load_model = self.v.gui.add_button("Load")
                self._load_status = self.v.gui.add_markdown(
                    "Loaded models are kept in `configs/local_models.yaml` (this machine only)")
        self._model_select.on_update(lambda _: self._show_model_info())
        self._on_click(run, self._run_policy)
        self._on_click(run_trials, self._run_trials)
        self._on_click(stop, self._stop_policy)
        self._on_click(load_model, self._load_model_from_paths)
        reload.on_click(lambda _: self._load_models())
        self._load_models()
        atexit.register(self._kill_policy)

    def _load_models(self):
        registry = load_registry()
        self._datasets = registry.get("datasets") or {}
        models = sorted(loadable_models(registry).items(),
                        key=lambda kv: kv[1].get("started") or "", reverse=True)  # recientes primero
        self._models = {}
        for label, model in load_local_models().items():   # primero: en otra máquina son los que sirven
            self._models[label + LOCAL_SUFFIX] = (model["path"], model)
        for key, model in models:
            label = model_label(key, model)
            if label in self._models:      # nombres repetidos: se agrega la carpeta
                label = f"{label} ({key.split('/')[-1]})"
            self._models[label] = (key, model)
        self._model_select.options = list(self._models) or ["-"]
        self._show_model_info()

    def _load_model_from_paths(self):
        """Valida el modelo y su dataset, los agrega a la lista (y a configs/local_models.yaml)."""
        try:
            model = describe_model(self._load_model_path.value)
            ds_path = self._load_dataset_path.value.strip()
            if not ds_path:
                found = dataset_for_checkpoint(model["path"])
                if found is None or not os.path.isdir(found["path"]):
                    raise ValueError("this model's dataset is not in configs/registry.yaml on this "
                                     "machine: set Dataset folder")
                ds_path = found["path"]
            ds = describe_dataset(ds_path)
            missing = [c for c in model["cameras"] if c not in ds["cameras"]]
            if missing:
                raise ValueError(f"the dataset has no {', '.join(missing)}, which the model uses "
                                 f"(dataset cameras: {', '.join(ds['cameras'])})")
        except ValueError as e:
            self._load_status.content = f"**Cannot load**: {e}"
            return
        label = os.path.basename(model["path"])
        save_local_model(label, {**model, "dataset_root": ds["path"],
                                 "dataset_episodes": ds["episodes"], "dataset_fps": ds["fps"]})
        self._load_models()
        self._model_select.value = label + LOCAL_SUFFIX
        self._show_model_info()
        self._load_status.content = (f"Loaded **{label}** ({', '.join(model['cameras'])}) with dataset "
                                     f"`{os.path.basename(ds['path'])}` ({ds['episodes']} ep, {ds['fps']} fps)")

    def _missing_cameras(self, model) -> list:
        return [c for c in model.get("cameras", []) if c not in self.s.camera_names]

    def _show_model_info(self):
        if self._model_select.value not in self._models:
            self._model_info.content = "No models: run `python scripts/xarm5/build_registry.py`"
            return
        key, m = self._models[self._model_select.value]
        ds = self._datasets.get(m.get("dataset")) or {}
        lines = [
            f"*{m['description']}*" if m.get("description") else "",
            f"`{key}`",
            f"cameras: {', '.join(m.get('cameras', []))}",
            (f"dataset: `{os.path.basename(m['dataset_root'])}` ({m.get('dataset_episodes', '?')} ep)"
             if m.get("dataset_root") else f"dataset: `{m.get('dataset')}` ({ds.get('episodes', '?')} ep)"),
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
        if self._as_trial.value:         # los ensayos siempre parten del reset
            self._start_trials(label, model, ["--here", "--trials_per_position", "1"], 1)
            return
        if self._reset_first.value:
            self.s.reset_robot()
        self._start_client(
            ["scripts/run_env_sim.py", "--agent", "lerobot_diffusion",
             "--hz", str(self.s.control_hz), "--robot_port", str(self.s.zmq_port),
             "--checkpoint", model["path"], *self._dataset_args(model)],
            label, trials=False)

    @staticmethod
    def _dataset_args(model) -> list:
        """--dataset_root para los modelos cargados a mano (los del registro lo buscan solos)."""
        return ["--dataset_root", model["dataset_root"]] if model.get("dataset_root") else []

    def _run_trials(self):
        selected = self._selected_model()
        if selected is None:
            return
        label, model = selected
        grid = load_grid()
        try:
            positions = parse_indices(self._trial_positions.value, len(grid))
        except ValueError as e:
            self._trial_status.content = f"**Positions**: {e}"
            return
        if not grid:
            self._trial_status.content = "**The grid is empty**: save positions in the Scene tab"
            return
        extra = ["--trials_per_position", str(int(self._trials_per_pos.value))]
        if positions is not None:
            extra += ["--positions", *map(str, positions)]
        n = len(positions if positions is not None else grid) * int(self._trials_per_pos.value)
        self._start_trials(label, model, extra, n)

    def _start_trials(self, label, model, extra: list, n: int):
        cmd = ["scripts/xarm5/run_policy_trials.py", "--checkpoint", model["path"], *self._dataset_args(model),
               "--run_name", label.removesuffix(LOCAL_SUFFIX),
               "--robot_port", str(self.s.zmq_port),
               "--trial_seconds", str(float(self._trial_seconds.value)), *extra]
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
            elif m := re.search(r"\[Trials\] (\d+) frames, error de agarre final ([\d.]+) m, mín ([\d.]+) m", line):
                info["done"] += 1
                info["result"] = (f"last: grasp error final {float(m.group(2)):.3f} m, "
                                  f"min {float(m.group(3)):.3f} m")
            elif m := re.search(r"\[Trials\] (\d+) frames, resultado: (\w+)", line):
                info["done"] += 1
                info["result"] = f"last: {m.group(2)} (marked by the operator)"
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

    # ── vista previa ─────────────────────────────────────────────────────────
    def _build_preview_gui(self):
        # una casilla y una imagen por cámara; en el sim cada vista activa cuesta un render
        # (salvo que un cliente las esté pidiendo: entonces se reusan sus observaciones)
        self._previews = {}
        panel = self.v.gui.add_panel()
        arrangements = load_camera_setup()["arrangements"]
        with panel.add_tab("Cameras", viser.Icon.CAMERA):
            # un arreglo (configs/camera_poses.yaml) marca de una vez sus cámaras
            arrangement = self.v.gui.add_dropdown(
                "Arrangement", ["custom", *arrangements], initial_value="custom")
            checks = {name: self.v.gui.add_checkbox(name, name in PREVIEW_DEFAULT)
                      for name in self.s.preview_cameras}
            for name, show in checks.items():
                image = self.v.gui.add_image(
                    np.zeros((PREVIEW_H, PREVIEW_W, 3), np.uint8), label=name,
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
        images = self.s.preview_images(names, max_age=2 * PREVIEW_PERIOD)
        for name, img in images.items():
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
        if now - self._last_tools > TOOLS_PERIOD:
            self._last_tools = now
            self._update_tools()

    def _update_operator(self):
        """Muestra la pregunta pendiente al operador con un botón por opción (o la oculta)."""
        op = getattr(self.s, "operator", None)
        if op is None:
            return
        state = (op.id, op.text is None)
        if state == self._op_shown:
            return
        self._op_shown = state
        if self._op_buttons is not None:
            self._op_buttons.remove()
            self._op_buttons = None
        if op.text is None:
            self._op_folder.visible = False
            return
        self._op_text.content = f"**{op.text}**"
        with self._op_folder:
            self._op_buttons = self.v.gui.add_button_group("Answer", op.options)
        prompt_id = op.id
        self._op_buttons.on_click(lambda e: op.reply(prompt_id, e.target.value))
        self._op_folder.visible = True

    def _update_status(self):
        self._poll_policy()
        self._update_operator()
        rec = self.s.recorder
        self._rec_status.content = (
            f"**● Recording** — {rec._length} frames in `{rec._folder}`" if rec.active
            else "Idle (also `r`/`s` + Enter in the terminal)")

        pose = self.s.get_current_ee_pose()
        if pose is not None:
            pos, quat = pose
            self._ee.position, self._ee.wxyz = pos, xyzw_to_wxyz(quat)
        self._update_status_extra()

    def _update_status_extra(self):
        """Estado propio de cada backend (se llama cada STATUS_PERIOD)."""
