"""Cola de entrenamiento con panel web: agrega trabajos en cualquier momento y se entrenan uno tras
otro (p.ej. dejar uno de 8 h corriendo y encolar el siguiente sin esperar a que termine).
También encola trabajos de datos (renderizar, espejar y convertir a LeRobot; scripts/xarm5/data_jobs.py),
encadenados: la pestaña Data arma render → mirror → convert de una carpeta de grabaciones, y el
dataset que va a salir se puede elegir en New job antes de que exista (el entrenamiento espera).
Corre a lo más un trabajo de GPU (entrenar, renderizar, espejar) y, si "Run CPU jobs alongside"
está activo, una conversión a la vez a su lado (data_jobs.LANES).

    python scripts/xarm5/train_server.py            # http://localhost:8082

La cola vive en ~/data/train_queues/live/jobs.yaml (configs y logs al lado), así que sobrevive a
reinicios. Cada entrenamiento corre en su propia sesión y escribe directo a su log: si se cierra
este panel, el entrenamiento sigue, y al volver a abrirlo se retoma su seguimiento por PID.

Cada entrenamiento = config base (configs/train/*.yaml) + los campos del formulario. Al terminar
un entrenamiento o una conversión se regenera configs/registry.yaml para que el modelo o el
dataset aparezcan en el panel del simulador. Un trabajo con depends_on espera a que ese termine
bien; si falla, queda esperando hasta que se reintente (o se quite) el anterior.
"""
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
import viser
import yaml

ROOT = Path(__file__).resolve().parents[2]       # raíz del repo
SCRIPTS = Path(__file__).resolve().parent      # scripts/xarm5
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
from splatsim.utils.render.camera_setup import load_camera_setup  # noqa: E402
from splatsim.utils.data.registry import load_registry  # noqa: E402
from splatsim.utils.data.episode_selection import load_discarded  # noqa: E402
from splatsim.utils.viser_page import set_page_title  # noqa: E402
from data_jobs import DATA_KINDS, SCRIPTS as JOB_SCRIPTS, command, kind, lane, plan_pipeline  # noqa: E402
from train_queue import _train_fields, build_runs  # noqa: E402

POLL_PERIOD = 2.0                 # s entre revisiones del trabajo en curso
STOP_TIMEOUT = 20.0               # s de espera tras SIGTERM antes de SIGKILL
ACTIVE = ("running",)
RETRYABLE = ("failed", "stopped", "interrupted")
FORM_FIELDS = ("n_obs_steps", "horizon", "n_action_steps", "training_steps", "batch_size", "save_freq")
DEFAULT_DATASETS_DIR = "~/lerobot/splatsim_test_data"   # donde build_registry.py busca datasets
# corre el trabajo y deja su código de salida al final del log (sirve aunque el panel se cierre)
WRAPPER = '"$@"; c=$?; echo "[exit $c]"; exit $c'


@dataclass
class Args:
    queue_dir: str = "~/data/train_queues/live"
    port: int = 8082
    recordings_root: str = "~/data/bc_data"
    """carpetas de grabaciones que ofrece la pestaña Data"""


# ── cola en disco ────────────────────────────────────────────────────────────
class JobStore:
    def __init__(self, qdir: Path):
        self.qdir = qdir
        (qdir / "configs").mkdir(parents=True, exist_ok=True)
        (qdir / "logs").mkdir(parents=True, exist_ok=True)
        self.path = qdir / "jobs.yaml"
        self.lock = threading.RLock()
        data = yaml.safe_load(self.path.read_text()) if self.path.exists() else None
        data = data or {}
        self.jobs: list[dict] = data.get("jobs") or []
        self.paused: bool = bool(data.get("paused", False))
        self.parallel_cpu: bool = bool(data.get("parallel_cpu", True))  # conversiones junto a la GPU
        self.next_id = max((j["id"] for j in self.jobs), default=0) + 1

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump({"paused": self.paused, "parallel_cpu": self.parallel_cpu,
                                           "jobs": self.jobs},
                                          sort_keys=False, allow_unicode=True))
            os.replace(tmp, self.path)

    def add(self, name, description, base=None, overrides=None, **extra) -> dict:
        """Entrenamiento (base + overrides) o, con kind/args/output en extra, trabajo de datos."""
        with self.lock:
            job = {"id": self.next_id, "name": name, "description": description,
                   **({"base": base, "overrides": overrides or {}} if base else {}),
                   **{k: v for k, v in extra.items() if v is not None}, "status": "pending",
                   "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
            self.next_id += 1
            self.jobs.append(job)
            self.save()
            return job

    def get(self, job_id: int) -> dict | None:
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def running(self) -> list[dict]:
        return [j for j in self.jobs if j["status"] in ACTIVE]

    def ready(self, job: dict) -> bool:
        """Sin depends_on, o el trabajo del que depende terminó bien (o ya no está en la cola)."""
        dep = self.get(job["depends_on"]) if job.get("depends_on") else None
        return dep is None or dep["status"] == "done"

    def move(self, job_id: int, delta: int):
        with self.lock:
            i = next(i for i, j in enumerate(self.jobs) if j["id"] == job_id)
            k = min(max(i + delta, 0), len(self.jobs) - 1)
            self.jobs.insert(k, self.jobs.pop(i))
            self.save()


def _pid_alive(pid: int | None, script: str = "train_diffusion.py") -> bool:
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return Path(script).name.encode() in cmdline


def _exit_code(log: Path) -> int | None:
    """Código que WRAPPER dejó al final del log, o None si el trabajo no terminó por su cuenta."""
    for line in reversed(_progress(log)["lines"][-5:]):
        if m := re.fullmatch(r"\[exit (\d+)\]", line.strip()):
            return int(m.group(1))
    return None


def _progress(log: Path) -> dict:
    """Último 'step: N loss: X' (entrenamiento) o 'n/total' (render, mirror, tqdm del convertidor)
    del log; lee solo el final del archivo."""
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 16384))
            tail = f.read().decode(errors="replace").splitlines()
    except FileNotFoundError:
        return {"lines": []}
    out = {"lines": tail[-200:]}
    for line in reversed(tail):
        if m := re.match(r"step: (\d+) loss: ([\d.eE+-]+)", line):
            out["step"], out["loss"] = int(m.group(1)), float(m.group(2))
            break
    for line in reversed(tail):
        if "Reading camera" in line:             # carga del splat, no es el avance del trabajo
            continue
        if m := re.search(r"(\d+)/(\d+)", line):
            out["done"], out["total"] = int(m.group(1)), int(m.group(2))
            break
    return out


# ── worker ───────────────────────────────────────────────────────────────────
class Worker:
    """Lanza el siguiente trabajo pendiente cuando no hay uno corriendo."""

    def __init__(self, store: JobStore):
        self.s = store
        self.procs: dict[int, subprocess.Popen] = {}   # trabajos lanzados por este proceso
        self._stop_deadline: dict[int, float] = {}
        for job in self.s.jobs:                        # retoma lo que quedó de una sesión anterior
            script = job.get("script") or JOB_SCRIPTS["train"]
            if job["status"] == "running" and not _pid_alive(job.get("pid"), script):
                job["status"] = "interrupted"
        self.s.save()

    def loop(self):
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  el worker no debe morir por un trabajo raro
                print(f"[Train] error en el worker: {type(e).__name__}: {e}")
            time.sleep(POLL_PERIOD)

    def tick(self):
        """Revisa los trabajos en curso y lanza, en orden de la cola, los listos cuyo carril esté
        libre: uno de GPU y (con parallel_cpu) uno de CPU; sin parallel_cpu, uno en total."""
        with self.s.lock:
            for job in self.s.running():
                self._check(job)
            if self.s.paused:
                return
            busy = {lane(j) for j in self.s.running()}
            for job in self.s.jobs:
                if busy and not self.s.parallel_cpu:
                    break
                if job["status"] == "pending" and lane(job) not in busy and self.s.ready(job):
                    self._start(job)
                    busy.add(lane(job))

    def _launch(self, job: dict, args: list[str], log_path: Path) -> subprocess.Popen:
        """Corre `python -u <args>` en su propia sesión con la salida directo al log: el trabajo no
        depende de este proceso."""
        log = open(log_path, "w")
        proc = subprocess.Popen(
            ["bash", "-c", WRAPPER, "job", sys.executable, "-u", *args],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        log.close()
        self.procs[job["id"]] = proc
        return proc

    def _start(self, job: dict):
        if kind(job) in DATA_KINDS:
            self._start_data(job)
        else:
            self._start_train(job)

    def _start_data(self, job: dict):
        args = command(job)
        safe = re.sub(r"[^\w.+-]+", "_", job["name"])
        log_path = self.s.qdir / "logs" / f"{job['id']:03d}_{safe}_{datetime.now():%m%d_%H%M}.log"
        proc = self._launch(job, args, log_path)
        job.update(status="running", pid=proc.pid, script=args[0], log=str(log_path),
                   started=datetime.now().strftime("%Y-%m-%d %H:%M"), started_ts=time.time())
        for k in ("finished", "minutes", "result"):
            job.pop(k, None)
        self.s.save()
        print(f"[Train] ▶ {job['id']} {job['name']} (pid {proc.pid}) → {log_path}")

    def _start_train(self, job: dict):
        with open(ROOT / job["base"]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg.update(job["overrides"])
        cfg["name"], cfg["description"] = job["name"], job["description"]
        cfg["model_name"] = f"{re.sub(r'[^\w.+-]+', '_', job['name'])}_{datetime.now():%m%d_%H%M}"
        cfg_path = self.s.qdir / "configs" / f"{job['id']:03d}_{cfg['model_name']}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        log_path = self.s.qdir / "logs" / f"{job['id']:03d}_{cfg['model_name']}.log"
        proc = self._launch(job, [JOB_SCRIPTS["train"], "--config", str(cfg_path)], log_path)
        job.update(status="running", pid=proc.pid, script=JOB_SCRIPTS["train"], model_name=cfg["model_name"],
                   model_dir=str(Path(cfg["output_root"]).expanduser() / cfg["model_name"]),
                   config=str(cfg_path), log=str(log_path),
                   training_steps=int(cfg.get("training_steps", 0)),
                   started=datetime.now().strftime("%Y-%m-%d %H:%M"), started_ts=time.time())
        for k in ("finished", "minutes", "result"):
            job.pop(k, None)
        self.s.save()
        print(f"[Train] ▶ {job['id']} {job['name']} (pid {proc.pid}) → {log_path}")

    def _check(self, job: dict):
        proc = self.procs.get(job["id"])
        if proc is not None:
            code = proc.poll()
            alive = code is None
        else:                                  # lanzado por una sesión anterior del panel
            alive, code = _pid_alive(job.get("pid"), job.get("script") or JOB_SCRIPTS["train"]), None
        deadline = self._stop_deadline.get(job["id"])
        if alive:
            if deadline is not None and time.monotonic() > deadline:
                os.killpg(job["pid"], signal.SIGKILL)
            return
        if code is None:
            code = _exit_code(Path(job["log"]))
        finished_ok = (code == 0) if code is not None else \
            kind(job) == "train" and (Path(job.get("model_dir", "")) / "model.safetensors").exists()
        job["status"] = "stopped" if deadline is not None else ("done" if finished_ok else "failed")
        job["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        job["minutes"] = round((time.time() - job.get("started_ts", time.time())) / 60, 1)
        self.procs.pop(job["id"], None)
        self._stop_deadline.pop(job["id"], None)
        self.s.save()
        print(f"[Train] ■ {job['id']} {job['name']}: {job['status']} ({job['minutes']} min)")
        if job["status"] == "done" and kind(job) in ("train", "convert"):
            threading.Thread(target=self._update_registry, daemon=True).start()

    def stop(self, job: dict):
        if job["status"] != "running":
            return
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._stop_deadline[job["id"]] = time.monotonic() + STOP_TIMEOUT

    @staticmethod
    def _update_registry():
        subprocess.run([sys.executable, str(SCRIPTS / "build_registry.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[Train] configs/registry.yaml actualizado")


# ── panel ────────────────────────────────────────────────────────────────────
class TrainPanel:
    def __init__(self, store: JobStore, worker: Worker, port: int, recordings_root: str):
        self.s, self.w = store, worker
        self.rec_root = Path(recordings_root).expanduser()
        self.allowed = _train_fields()
        self.v = viser.ViserServer(port=port, label="Training queue", verbose=False)
        set_page_title(self.v, "Training queue")
        tabs = self.v.gui.add_tab_group()
        with tabs.add_tab("Queue", viser.Icon.LIST):
            self._build_queue_tab()
        with tabs.add_tab("New job", viser.Icon.PLUS):
            self._build_form()
        with tabs.add_tab("Data", viser.Icon.DATABASE):
            self._build_data_tab()
        self.v.gui.main_panel.dock_right()
        self.v.gui.main_panel.set_width(520)
        print(f"[Train] Panel en http://localhost:{port}  (cola en {store.path})")

    # ── pestaña Queue ────────────────────────────────────────────────────────
    def _build_queue_tab(self):
        gui = self.v.gui
        with gui.add_folder("Current"):
            # un espacio por carril: el de GPU siempre, el de CPU solo con una conversión en curso
            self.slots = {name: {"title": gui.add_markdown(""), "bar": gui.add_progress_bar(0.0),
                                 "log": gui.add_markdown("")} for name in ("gpu", "cpu")}
            self.pause_box = gui.add_checkbox("Pause queue", self.s.paused,
                                              hint="Finish the running jobs but don't start more")
            self.parallel_box = gui.add_checkbox(
                "Run CPU jobs alongside", self.s.parallel_cpu,
                hint="Convert datasets while a GPU job (training, render, mirror) runs; "
                     "both use RAM, turn it off if the machine runs short")
            stop = gui.add_button("Stop running jobs", color="red",
                                  hint="Stops them (trainings keep their checkpoints) and pauses the queue")
        with gui.add_folder("Jobs"):
            self.table = gui.add_markdown("")
            self.sel = gui.add_dropdown("Job", ["-"])
            self.sel_info = gui.add_markdown("")
            row = gui.add_button_group("Order", ["▲ Up", "▼ Down"])
            actions = gui.add_button_group("Action", ["Retry", "Duplicate", "Remove"])
        with gui.add_folder("Import queue file", expand_by_default=False):
            self.import_path = gui.add_text("Queue YAML", "configs/train/queue_example.yaml")
            imp = gui.add_button("Add its runs to the queue")
            self.import_msg = gui.add_markdown("")

        def on_pause(e):
            if e.client is not None:
                with self.s.lock:
                    self.s.paused = self.pause_box.value
                    self.s.save()
        self.pause_box.on_update(on_pause)

        def on_parallel(e):
            if e.client is not None:
                with self.s.lock:
                    self.s.parallel_cpu = self.parallel_box.value
                    self.s.save()
        self.parallel_box.on_update(on_parallel)
        stop.on_click(lambda _: self._stop_current())
        self.sel.on_update(lambda _: self._show_selected())
        row.on_click(lambda e: self._move(-1 if "Up" in e.target.value else 1))
        actions.on_click(lambda e: self._action(e.target.value))
        imp.on_click(lambda _: self._import())

    def _selected(self) -> dict | None:
        m = re.match(r"#(\d+) ", self.sel.value or "")
        return self.s.get(int(m.group(1))) if m else None

    def _stop_current(self):
        with self.s.lock:
            self.s.paused = True
            self.pause_box.value = True
            self.s.save()
            for job in self.s.running():
                self.w.stop(job)

    def _move(self, delta):
        job = self._selected()
        if job:
            self.s.move(job["id"], delta)

    def _action(self, action):
        job = self._selected()
        if job is None:
            return
        with self.s.lock:
            if action == "Remove":
                if job["status"] in ACTIVE:
                    self.sel_info.content = "**Stop it first** (Stop current job)"
                    return
                self.s.jobs.remove(job)
            elif action == "Retry":
                if job["status"] not in RETRYABLE:
                    self.sel_info.content = f"Only {', '.join(RETRYABLE)} jobs can be retried"
                    return
                job["status"] = "pending"
            elif action == "Duplicate":
                if kind(job) == "train":
                    self.s.add(job["name"], job["description"], job["base"], dict(job["overrides"]),
                               depends_on=job.get("depends_on"))
                else:
                    self.s.add(job["name"], job["description"], kind=job["kind"], args=dict(job["args"]),
                               output=job["output"], depends_on=job.get("depends_on"))
            self.s.save()

    def _import(self):
        path = ROOT / self.import_path.value
        try:
            queue = yaml.safe_load(path.read_text()) or {}
            runs = build_runs(queue)
        except (OSError, SystemExit, yaml.YAMLError) as e:
            self.import_msg.content = f"**Error**: {e}"
            return
        base = queue.get("base", "configs/train/diffusion.yaml")
        for r in runs:
            overrides = {**(queue.get("common") or {}),
                         **{k: v for k, v in r.items() if k not in ("name", "description")}}
            self.s.add(r["name"], r.get("description", ""), base, overrides)
        self.import_msg.content = f"Added {len(runs)} jobs from `{self.import_path.value}`"

    def _show_selected(self):
        job = self._selected()
        if job is None:
            self.sel_info.content = ""
            return
        lines = [f"*{job['description']}*" if job.get("description") else ""]
        if job.get("depends_on"):
            dep = self.s.get(job["depends_on"])
            lines.append(f"after #{job['depends_on']}" + (f" ({dep['status']})" if dep else " (removed)"))
        if kind(job) == "train":
            lines += [f"base `{job['base']}`",
                      "overrides: " + (", ".join(f"`{k}={v}`" for k, v in job["overrides"].items()) or "—")]
        else:
            lines += [f"`{' '.join(command(job))}`", f"output `{job['output']}`"]
        if job.get("model_dir"):
            lines.append(f"model `{job['model_dir']}`")
        if job.get("log"):
            lines.append(f"log `{job['log']}`")
        self.sel_info.content = "  \n".join(line for line in lines if line)

    def refresh(self):
        """Llamar periódicamente: tabla, trabajo en curso y lista de selección."""
        with self.s.lock:
            jobs = [dict(j) for j in self.s.jobs]
            paused = self.s.paused
        icons = {"pending": "⏳", "running": "▶", "done": "✓", "failed": "✗", "stopped": "■",
                 "interrupted": "⚠"}
        rows = ["| # | name | status | steps | added | time |", "|---|---|---|---|---|---|"]
        by_id = {j["id"]: j for j in jobs}
        for j in jobs:
            steps = (j["overrides"].get("training_steps", j.get("training_steps", ""))
                     if kind(j) == "train" else kind(j))
            t = f"{j['minutes']} min" if j.get("minutes") is not None else (j.get("started") or "")
            status = f"{icons.get(j['status'], '')} {j['status']}"
            dep = by_id.get(j.get("depends_on"))
            if j["status"] == "pending" and dep is not None and dep["status"] != "done":
                status += f" (after #{dep['id']})"
            rows.append(f"| {j['id']} | {j['name']} | {status} | {steps} | {j['added'][5:]} | {t} |")
        pending = sum(j["status"] == "pending" for j in jobs)
        self.table.content = "\n".join(rows) + (f"\n\n{pending} pending" + (" · **queue paused**" if paused else ""))
        options = [f"#{j['id']} {j['name']} ({j['status']})" for j in jobs] or ["-"]
        current = self.sel.value
        if tuple(options) != tuple(self.sel.options):
            self.sel.options = options
            keep = next((o for o in options if o.split(" ")[0] == (current or "").split(" ")[0]), None)
            if keep:
                self.sel.value = keep
        if self.pause_box.value != paused:
            self.pause_box.value = paused
        if self.parallel_box.value != self.s.parallel_cpu:
            self.parallel_box.value = self.s.parallel_cpu
        running = {lane(j): j for j in jobs if j["status"] in ACTIVE}
        self._show_current(running.get("gpu"), self.slots["gpu"])
        cpu = self.slots["cpu"]
        for h in cpu.values():
            h.visible = "cpu" in running
        if "cpu" in running:
            self._show_current(running["cpu"], cpu)
        queued = tuple((j["id"], j["status"]) for j in jobs if kind(j) == "convert")
        if queued != self._queued_key:            # datasets por venir o recién convertidos
            self._queued_key = queued
            self._load_datasets()

    def _show_current(self, job, slot):
        if job is None:
            slot["title"].content = "No GPU job running" + (" (queue paused)" if self.s.paused else "")
            slot["bar"].value = 0.0
            slot["log"].content = ""
            return
        p = _progress(Path(job["log"]))
        if kind(job) != "train":
            self._show_current_data(job, p, slot)
            return
        total = job.get("training_steps") or 1
        step = p.get("step", 0)
        elapsed = time.time() - job.get("started_ts", time.time())
        rate = step / elapsed if step and elapsed > 0 else 0
        eta = f", ETA {_fmt_dur((total - step) / rate)}" if rate else ""
        loss = f" · loss {p['loss']:.4f}" if "loss" in p else ""
        slot["title"].content = (f"**#{job['id']} {job['name']}** — step {step}/{total}{loss}  \n"
                                  f"{_fmt_dur(elapsed)} elapsed{eta}" + (f" · {rate:.1f} it/s" if rate else ""))
        slot["bar"].value = 100.0 * step / total
        # últimas líneas del log sin la avalancha de 'step:' (solo la última)
        lines = [line for line in p["lines"] if not line.startswith("step: ")][-8:]
        slot["log"].content = "```\n" + "\n".join(lines) + "\n```" if lines else ""

    def _show_current_data(self, job, p, slot):
        elapsed = time.time() - job.get("started_ts", time.time())
        done, total = p.get("done", 0), p.get("total", 0)
        frac = done / total if total else 0.0
        eta = f", ETA {_fmt_dur(elapsed * (1 - frac) / frac)}" if frac > 0 else ""
        slot["title"].content = (f"**#{job['id']} {job['name']}** ({kind(job)})"
                                  + (f" — {done}/{total}" if total else "")
                                  + f"  \n{_fmt_dur(elapsed)} elapsed{eta}")
        slot["bar"].value = 100.0 * frac
        lines = [line for line in p["lines"] if "Reading camera" not in line][-8:]
        slot["log"].content = "```\n" + "\n".join(lines) + "\n```" if lines else ""

    # ── pestaña New job ──────────────────────────────────────────────────────
    def _build_form(self):
        gui = self.v.gui
        bases = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "train").glob("*.yaml")
                       if not ({"runs", "sweep"} & set(yaml.safe_load(p.read_text()) or {})))
        self._queued_key = None
        self.datasets = {}
        arrangements = load_camera_setup()["arrangements"]
        self.arrangements = arrangements

        self.f_name = gui.add_text("Name", "", hint="Short name, e.g. 3cam-h32")
        self.f_desc = gui.add_text("Description", "", hint="What this model tries to improve")
        self.f_base = gui.add_dropdown("Base config", bases, initial_value="configs/train/diffusion.yaml"
                                       if "configs/train/diffusion.yaml" in bases else bases[0])
        self.f_dataset = gui.add_dropdown("Dataset", ["(base config)"],
                                          hint="⏳ = queued conversion: the job waits for it")
        self.f_arr = gui.add_dropdown("Arrangement", ["(base config)", *arrangements])
        self.f_cams = gui.add_text("Cameras", "", hint="Instead of an arrangement: base_rgb, wrist_rgb")
        self.f_nums = {k: gui.add_number(k, 0, step=1) for k in FORM_FIELDS}
        self.f_aug = gui.add_checkbox("image_augmentation", True)
        self.f_hub = gui.add_checkbox("push_to_hub", False)
        self.f_wandb = gui.add_checkbox("wandb", True, hint="Log to Weights & Biases (offline without `wandb login`)")
        self.f_extra = gui.add_text("Extra (YAML)", "", multiline=True,
                                    hint="Any other training field, e.g.  vision_backbone: resnet34")
        self.f_info = gui.add_markdown("")
        add = gui.add_button("Add to queue", color="green", icon=viser.Icon.PLUS)

        self.f_base.on_update(lambda _: self._load_base())
        self.f_dataset.on_update(lambda _: self._show_dataset())
        add.on_click(lambda _: self._add_job())
        self._load_datasets()
        self._load_base()

    def _load_datasets(self):
        """Datasets del registro y los que dejarán las conversiones aún no terminadas (⏳ #id)."""
        registry = load_registry()
        datasets = {k: d for k, d in (registry.get("datasets") or {}).items() if d.get("episodes")}
        options = sorted(datasets, key=lambda k: datasets[k]["created"], reverse=True)
        with self.s.lock:
            pending = [dict(j) for j in self.s.jobs if kind(j) == "convert" and j["status"] != "done"]
        for j in pending:
            label = f"⏳ #{j['id']} {Path(j['output']).name}"
            datasets[label] = {"path": j["output"], "cameras": j["args"].get("cameras"), "queued": j["id"]}
            options.insert(0, label)
        self.datasets = datasets
        current = self.f_dataset.value
        self.f_dataset.options = ["(base config)", *options]
        if current in datasets:
            self.f_dataset.value = current

    def _base_cfg(self) -> dict:
        return yaml.safe_load((ROOT / self.f_base.value).read_text()) or {}

    def _load_base(self):
        cfg = self._base_cfg()
        for k, h in self.f_nums.items():
            h.value = int(cfg.get(k, 0) or 0)
        self.f_aug.value = bool(cfg.get("image_augmentation", True))
        self.f_hub.value = bool(cfg.get("push_to_hub", False))
        self.f_wandb.value = bool(cfg.get("wandb", True))
        self._show_dataset()

    def _show_dataset(self):
        ds = self.datasets.get(self.f_dataset.value)
        if ds is None:
            root = self._base_cfg().get("dataset_root", "")
            self.f_info.content = f"dataset `{Path(root).name}`"
        elif ds.get("queued"):
            self.f_info.content = (f"dataset `{Path(ds['path']).name}`, made by queued job #{ds['queued']}: "
                                   "this job will wait for it")
        else:
            src = ", ".join(f"{s['recording']}[{s['recording_episodes']}]" for s in ds.get("sources", [])) or "?"
            self.f_info.content = (f"dataset `{self.f_dataset.value}`: {ds['episodes']} episodes, "
                                   f"cameras {', '.join(ds['cameras'])}, from {src}")

    def _add_job(self):
        try:
            name, overrides = self._form_overrides()
        except ValueError as e:
            self.f_info.content = f"**Not added**: {e}"
            return
        ds = self.datasets.get(self.f_dataset.value) or {}
        job = self.s.add(name, self.f_desc.value.strip(), self.f_base.value, overrides,
                         depends_on=ds.get("queued"))
        self.f_info.content = (f"Added **#{job['id']} {name}**"
                               + (" — the queue is paused" if self.s.paused else ""))
        self.f_name.value = ""
        self.f_desc.value = ""

    def _form_overrides(self) -> tuple[str, dict]:
        """Campos que difieren de la config base, validados."""
        name = self.f_name.value.strip()
        if not name:
            raise ValueError("give the job a name")
        base = self._base_cfg()
        values = {k: int(h.value) for k, h in self.f_nums.items()}
        values["image_augmentation"] = bool(self.f_aug.value)
        values["push_to_hub"] = bool(self.f_hub.value)
        values["wandb"] = bool(self.f_wandb.value)
        cams = [c.strip() for c in self.f_cams.value.split(",") if c.strip()]
        if cams and self.f_arr.value != "(base config)":
            raise ValueError("use an arrangement or cameras, not both")
        if self.f_arr.value != "(base config)":
            values["arrangement"], values["cameras"] = self.f_arr.value, None
        elif cams:
            values["cameras"], values["arrangement"] = cams, None
        ds = self.datasets.get(self.f_dataset.value)
        if ds is not None:
            values["dataset_root"] = ds["path"]
            values["dataset_id"] = f"local/{Path(ds['path']).name}"
        if self.f_extra.value.strip():
            try:
                extra = yaml.safe_load(self.f_extra.value) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Extra is not valid YAML: {e}") from e
            if not isinstance(extra, dict):
                raise ValueError("Extra must be `key: value` lines")
            values.update(extra)
        unknown = set(values) - self.allowed
        if unknown:
            raise ValueError(f"unknown training fields {sorted(unknown)}")

        merged = {**base, **values}
        h, o, a = merged["horizon"], merged["n_obs_steps"], merged["n_action_steps"]
        if h % 8:
            raise ValueError(f"horizon ({h}) must be a multiple of 8")
        if a > h - o + 1:
            raise ValueError(f"n_action_steps ({a}) must be ≤ horizon − n_obs_steps + 1 = {h - o + 1}")
        wanted = (self.arrangements.get(merged["arrangement"]) if merged.get("arrangement")
                  else merged.get("cameras"))
        ds_cams = ds["cameras"] if ds is not None else self._base_dataset_cameras(merged)
        if wanted and ds_cams is not None and set(wanted) - set(ds_cams):
            raise ValueError(f"the dataset has no {sorted(set(wanted) - set(ds_cams))} "
                             f"(it has {', '.join(ds_cams)})")
        return name, {k: v for k, v in values.items() if base.get(k) != v}

    # ── pestaña Data: render → mirror → convert ──────────────────────────────
    def _build_data_tab(self):
        gui = self.v.gui
        arrangements = list(self.arrangements)
        with gui.add_folder("Generate (planner)", expand_by_default=False):
            self.d_gen = gui.add_checkbox("Generate", False,
                                          hint="plan_trajectories.py: synthetic demos instead of a recordings folder")
            self.d_gen_name = gui.add_text("Folder", f"planned_{datetime.now():%y%m%d}",
                                           hint=f"In {self.rec_root}; an existing one gets more episodes")
            self.d_gen_n = gui.add_number("Episodes", 100, min=1, max=10000, step=1)
            self.d_gen_pos = gui.add_dropdown("Positions", ["random", "grid"], initial_value="random",
                                              hint="random: the saved range (hemisphere or box), away from the grid")
            self.d_gen_seed = gui.add_number("Seed", -1, step=1, hint="-1 = random")
            self.d_gen_perturb = gui.add_number("Perturb (rad)", 0.0, min=0.0, max=0.5, step=0.01,
                                                hint="Smooth deviation on the way, corrected before arriving")
        self.d_source = gui.add_dropdown("Recordings", self._recording_folders() or ["-"],
                                         hint=f"Folders of {self.rec_root}")
        self.d_source_info = gui.add_markdown("")
        with gui.add_folder("Render (state-only recordings)"):
            self.d_render = gui.add_checkbox("Render", False, hint="render_recordings.py: resample to a fixed rate")
            self.d_render_fps = gui.add_number("Render fps", 10, min=1, max=60, step=1)
            self.d_render_cams = gui.add_dropdown("Render cameras", ["all", *arrangements], initial_value="all")
            self.d_render_workers = gui.add_number("Render workers", 1, min=1, max=8, step=1,
                                                   hint="Parallel processes, ~8 GB of GPU each: keep 1 on a 16 GB GPU")
        with gui.add_folder("Mirror"):
            self.d_mirror = gui.add_checkbox("Mirror", False,
                                             hint="mirror_recordings.py: add the mirror image of every episode")
            self.d_mirror_orig = gui.add_checkbox("Keep originals", True,
                                                  hint="Originals re-rendered with the same cameras, then the mirrors")
            self.d_mirror_cams = gui.add_dropdown("Mirror cameras", ["as recorded", *arrangements],
                                                  initial_value="as recorded")
            self.d_mirror_workers = gui.add_number("Mirror workers", 1, min=1, max=8, step=1,
                                                   hint="Parallel processes, ~8 GB of GPU each: keep 1 on a 16 GB GPU")
        with gui.add_folder("Convert to LeRobot"):
            self.d_convert = gui.add_checkbox("Convert", True)
            self.d_out = gui.add_text("Output folder", DEFAULT_DATASETS_DIR)
            self.d_fps = gui.add_number("fps", 10, min=1, max=60, step=1,
                                        hint="The recordings' rate (the render fps when rendering)")
            self.d_cams = gui.add_text("Cameras", "", hint="Space-separated; empty = all")
            self.d_suffix = gui.add_text("Name suffix", "", hint="Dataset name ends in it; empty = date")
            self.d_discarded = gui.add_checkbox("Include discarded", False)
            self.d_push = gui.add_checkbox("Push to Hub", False)
        self.d_plan = gui.add_markdown("")
        refresh = gui.add_button("Refresh folders")
        add = gui.add_button("Add to queue", color="green", icon=viser.Icon.PLUS)
        for h in (self.d_gen, self.d_gen_name, self.d_gen_n, self.d_gen_pos, self.d_source, self.d_render, self.d_render_fps, self.d_render_cams, self.d_mirror,
                  self.d_mirror_orig, self.d_mirror_cams, self.d_convert, self.d_out, self.d_fps,
                  self.d_cams, self.d_suffix):
            h.on_update(lambda _: self._show_plan())
        self.d_source.on_update(lambda _: self._show_source())
        self.d_gen.on_update(lambda _: self._show_source())
        refresh.on_click(lambda _: self._refresh_folders())
        add.on_click(lambda _: self._add_pipeline())
        self._show_source()

    def _recording_folders(self) -> list[str]:
        if not self.rec_root.is_dir():
            return []
        return sorted(p.name for p in self.rec_root.iterdir()
                      if p.is_dir() and any(c.is_dir() and c.name.isdigit() for c in p.iterdir()))

    def _refresh_folders(self):
        current = self.d_source.value
        self.d_source.options = self._recording_folders() or ["-"]
        if current in self.d_source.options:
            self.d_source.value = current

    def _source_kind(self, folder: Path) -> tuple[str, int | None]:
        """("state" | "images" | "empty", fps de render_info.yaml) según el primer episodio."""
        episodes = sorted(c for c in folder.iterdir() if c.is_dir() and c.name.isdigit())
        if not episodes:
            return "empty", None
        info = folder / "render_info.yaml"
        fps = (yaml.safe_load(info.read_text()) or {}).get("fps") if info.exists() else None
        return ("images" if (episodes[0] / "image_1").is_dir() else "state"), fps

    def _show_source(self):
        self.d_source.disabled = self.d_gen.value
        if self.d_gen.value:     # las demos del planificador son de solo estado: hay que renderizarlas
            self.d_source_info.content = "Source: the planner's folder (state only: rendered next)"
            self.d_render.value = True
            self._show_plan()
            return
        folder = self.rec_root / self.d_source.value
        if not folder.is_dir():
            self.d_source_info.content = ""
            return
        kind_, fps = self._source_kind(folder)
        n = sum(c.is_dir() and c.name.isdigit() for c in folder.iterdir())
        discarded = len(load_discarded(folder))
        self.d_source_info.content = (
            f"{n} episodes" + (f" ({discarded} discarded)" if discarded else "") + ", "
            + ("**state only**: render it" if kind_ == "state" else "with images")
            + (f", rendered at {fps} fps" if fps else ""))
        self.d_render.value = kind_ == "state"
        if fps:
            self.d_fps.value = fps
        self._show_plan()

    def _pipeline(self) -> list[dict]:
        """Trabajos de la pestaña Data, validados (ValueError si algo no cuadra)."""
        plan = None
        if self.d_gen.value:
            name = self.d_gen_name.value.strip()
            if not re.fullmatch(r"[\w.-]+", name):
                raise ValueError("planner folder: letters, digits, `_`, `-` or `.`")
            folder = self.rec_root / name
            seed = int(self.d_gen_seed.value)
            plan = {"episodes": int(self.d_gen_n.value), "positions": self.d_gen_pos.value,
                    "seed": seed if seed >= 0 else None, "perturb": float(self.d_gen_perturb.value)}
            kind_ = "state"
        else:
            folder = self.rec_root / self.d_source.value
            if not folder.is_dir():
                raise ValueError("choose a recordings folder")
            kind_, _ = self._source_kind(folder)
        if kind_ == "empty":
            raise ValueError("the folder has no episodes")
        if not (self.d_render.value or self.d_mirror.value or self.d_convert.value):
            raise ValueError("enable at least one step")
        if kind_ == "state" and not self.d_render.value and (self.d_mirror.value or self.d_convert.value):
            raise ValueError("state-only recordings: enable Render first (fixed rate)")
        render = mirror = convert = None
        if self.d_render.value:
            render = {"fps": int(self.d_render_fps.value),
                      "arrangement": None if self.d_render_cams.value == "all" else self.d_render_cams.value,
                      "workers": int(self.d_render_workers.value)}
        if self.d_mirror.value:
            mirror = {"with_original": bool(self.d_mirror_orig.value),
                      "arrangement": None if self.d_mirror_cams.value == "as recorded" else self.d_mirror_cams.value,
                      "workers": int(self.d_mirror_workers.value)}
        if self.d_convert.value:
            suffix = self.d_suffix.value.strip() or datetime.now().strftime("%y%m%d_%H%M")
            if not re.fullmatch(r"[\w.-]+", suffix):
                raise ValueError("name suffix: letters, digits, `_`, `-` or `.`")
            convert = {"output_dir": self.d_out.value.strip() or DEFAULT_DATASETS_DIR,
                       "fps": int(self.d_render_fps.value) if render else int(self.d_fps.value),
                       "cameras": self.d_cams.value.split() or None, "suffix": suffix,
                       "ignore_selection": bool(self.d_discarded.value), "push": bool(self.d_push.value)}
        jobs = plan_pipeline(str(folder), render, mirror, convert, plan)
        with self.s.lock:
            queued = {j.get("output") for j in self.s.jobs if j["status"] != "done"}
        last = jobs[-1]
        if last["kind"] == "convert" and (Path(last["output"]).exists() or last["output"] in queued):
            raise ValueError(f"dataset `{Path(last['output']).name}` already exists or is queued: "
                             "change the name suffix")
        return jobs

    def _show_plan(self):
        try:
            jobs = self._pipeline()
        except ValueError as e:
            self.d_plan.content = f"**Can't add**: {e}"
            return
        exists = {"plan": " (exists: episodes are added until it has the requested number)",
                  "render": " (exists: finished episodes are skipped)",
                  "mirror": " (exists: finished episodes are skipped)"}
        lines = [f"{i + 1}. **{j['kind']}** → `{j['output']}`"
                 + (exists.get(j["kind"], "") if Path(j["output"]).exists() else "")
                 for i, j in enumerate(jobs)]
        if jobs[-1]["kind"] == "convert":
            lines.append("Then pick it in **New job** → Dataset (⏳) to train on it after it is made.")
        self.d_plan.content = "  \n".join(lines)

    def _add_pipeline(self):
        try:
            jobs = self._pipeline()
        except ValueError as e:
            self.d_plan.content = f"**Not added**: {e}"
            return
        ids = []
        for j in jobs:
            dep = ids[j["depends_on"]] if j["depends_on"] is not None else None
            ids.append(self.s.add(j["name"], "", kind=j["kind"], args=j["args"], output=j["output"],
                                  depends_on=dep)["id"])
        self._load_datasets()
        self.d_suffix.value = ""
        self.d_plan.content = (f"Added {', '.join(f'#{i}' for i in ids)}"
                               + (" — the queue is paused" if self.s.paused else ""))

    def _base_dataset_cameras(self, cfg: dict) -> list | None:
        root = os.path.realpath(os.path.expanduser(cfg.get("dataset_root", "")))
        for d in self.datasets.values():
            if os.path.realpath(d["path"]) == root:
                return d["cameras"]
        return None


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{seconds % 60:02d}s"


def main(args: Args):
    store = JobStore(Path(os.path.expanduser(args.queue_dir)))
    worker = Worker(store)
    panel = TrainPanel(store, worker, args.port, args.recordings_root)
    threading.Thread(target=worker.loop, daemon=True).start()
    try:
        while True:
            panel.refresh()
            time.sleep(1.0)
    except KeyboardInterrupt:
        for job in store.running():
            print(f"\n[Train] Panel cerrado; #{job['id']} {job['name']} sigue corriendo "
                  f"(pid {job['pid']}). Vuelve a abrir el panel para seguirlo.")


if __name__ == "__main__":
    main(tyro.cli(Args))
