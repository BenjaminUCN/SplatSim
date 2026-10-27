"""Cola de entrenamiento con panel web: agrega trabajos en cualquier momento y se entrenan uno tras
otro (p.ej. dejar uno de 8 h corriendo y encolar el siguiente sin esperar a que termine).

    python scripts/train_server.py            # http://localhost:8082

La cola vive en ~/data/train_queues/live/jobs.yaml (configs y logs al lado), así que sobrevive a
reinicios. Cada entrenamiento corre en su propia sesión y escribe directo a su log: si se cierra
este panel, el entrenamiento sigue, y al volver a abrirlo se retoma su seguimiento por PID.

Cada trabajo = config base (configs/train/*.yaml) + los campos del formulario. Al terminar un
trabajo se regenera configs/registry.yaml para que el modelo aparezca en el panel del simulador.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from splatsim.utils.camera_setup import load_camera_setup  # noqa: E402
from splatsim.utils.registry import load_registry  # noqa: E402
from train_queue import _train_fields, build_runs  # noqa: E402

POLL_PERIOD = 2.0                 # s entre revisiones del trabajo en curso
STOP_TIMEOUT = 20.0               # s de espera tras SIGTERM antes de SIGKILL
ACTIVE = ("running",)
RETRYABLE = ("failed", "stopped", "interrupted")
FORM_FIELDS = ("n_obs_steps", "horizon", "n_action_steps", "training_steps", "batch_size", "save_freq")


@dataclass
class Args:
    queue_dir: str = "~/data/train_queues/live"
    port: int = 8082


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
        self.next_id = max((j["id"] for j in self.jobs), default=0) + 1

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump({"paused": self.paused, "jobs": self.jobs},
                                          sort_keys=False, allow_unicode=True))
            os.replace(tmp, self.path)

    def add(self, name, description, base, overrides) -> dict:
        with self.lock:
            job = {"id": self.next_id, "name": name, "description": description, "base": base,
                   "overrides": overrides, "status": "pending",
                   "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
            self.next_id += 1
            self.jobs.append(job)
            self.save()
            return job

    def get(self, job_id: int) -> dict | None:
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def running(self) -> dict | None:
        return next((j for j in self.jobs if j["status"] in ACTIVE), None)

    def move(self, job_id: int, delta: int):
        with self.lock:
            i = next(i for i, j in enumerate(self.jobs) if j["id"] == job_id)
            k = min(max(i + delta, 0), len(self.jobs) - 1)
            self.jobs.insert(k, self.jobs.pop(i))
            self.save()


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"train_diffusion.py" in cmdline


def _progress(log: Path) -> dict:
    """Último 'step: N loss: X' del log (lee solo el final del archivo)."""
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
    return out


# ── worker ───────────────────────────────────────────────────────────────────
class Worker:
    """Lanza el siguiente trabajo pendiente cuando no hay uno corriendo."""

    def __init__(self, store: JobStore):
        self.s = store
        self.procs: dict[int, subprocess.Popen] = {}   # trabajos lanzados por este proceso
        self._stop_deadline: dict[int, float] = {}
        for job in self.s.jobs:                        # retoma lo que quedó de una sesión anterior
            if job["status"] == "running" and not _pid_alive(job.get("pid")):
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
        with self.s.lock:
            job = self.s.running()
            if job is not None:
                self._check(job)
            elif not self.s.paused:
                nxt = next((j for j in self.s.jobs if j["status"] == "pending"), None)
                if nxt is not None:
                    self._start(nxt)

    def _start(self, job: dict):
        with open(ROOT / job["base"]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg.update(job["overrides"])
        cfg["name"], cfg["description"] = job["name"], job["description"]
        cfg["model_name"] = f"{re.sub(r'[^\w.+-]+', '_', job['name'])}_{datetime.now():%m%d_%H%M}"
        cfg_path = self.s.qdir / "configs" / f"{job['id']:03d}_{cfg['model_name']}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        log_path = self.s.qdir / "logs" / f"{job['id']:03d}_{cfg['model_name']}.log"
        log = open(log_path, "w")
        # sesión propia y salida directa al log: el entrenamiento no depende de este proceso
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "train_diffusion.py"), "--config", str(cfg_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        log.close()
        self.procs[job["id"]] = proc
        job.update(status="running", pid=proc.pid, model_name=cfg["model_name"],
                   model_dir=str(Path(cfg["output_root"]) / cfg["model_name"]),
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
            alive, code = _pid_alive(job.get("pid")), None
        deadline = self._stop_deadline.get(job["id"])
        if alive:
            if deadline is not None and time.monotonic() > deadline:
                os.killpg(job["pid"], signal.SIGKILL)
            return
        finished_ok = (code == 0) if code is not None else \
            (Path(job.get("model_dir", "")) / "model.safetensors").exists()
        job["status"] = "stopped" if deadline is not None else ("done" if finished_ok else "failed")
        job["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        job["minutes"] = round((time.time() - job.get("started_ts", time.time())) / 60, 1)
        self.procs.pop(job["id"], None)
        self._stop_deadline.pop(job["id"], None)
        self.s.save()
        print(f"[Train] ■ {job['id']} {job['name']}: {job['status']} ({job['minutes']} min)")
        if job["status"] == "done":
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
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_registry.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[Train] configs/registry.yaml actualizado")


# ── panel ────────────────────────────────────────────────────────────────────
class TrainPanel:
    def __init__(self, store: JobStore, worker: Worker, port: int):
        self.s, self.w = store, worker
        self.allowed = _train_fields()
        self.v = viser.ViserServer(port=port, label="Training queue", verbose=False)
        tabs = self.v.gui.add_tab_group()
        with tabs.add_tab("Queue", viser.Icon.LIST):
            self._build_queue_tab()
        with tabs.add_tab("New job", viser.Icon.PLUS):
            self._build_form()
        self.v.gui.main_panel.dock_right()
        self.v.gui.main_panel.set_width(520)
        print(f"[Train] Panel en http://localhost:{port}  (cola en {store.path})")

    # ── pestaña Queue ────────────────────────────────────────────────────────
    def _build_queue_tab(self):
        gui = self.v.gui
        with gui.add_folder("Current"):
            self.cur_title = gui.add_markdown("")
            self.cur_bar = gui.add_progress_bar(0.0)
            self.cur_log = gui.add_markdown("")
            self.pause_box = gui.add_checkbox("Pause queue", self.s.paused,
                                              hint="Finish the current job but don't start the next")
            stop = gui.add_button("Stop current job", color="red",
                                  hint="Stops it (keeps its checkpoints) and pauses the queue")
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
            job = self.s.running()
            self.s.paused = True
            self.pause_box.value = True
            self.s.save()
            if job:
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
                self.s.add(job["name"], job["description"], job["base"], dict(job["overrides"]))
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
        lines = [f"*{job['description']}*" if job.get("description") else "",
                 f"base `{job['base']}`",
                 "overrides: " + (", ".join(f"`{k}={v}`" for k, v in job["overrides"].items()) or "—")]
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
        for j in jobs:
            steps = j["overrides"].get("training_steps", j.get("training_steps", ""))
            t = f"{j['minutes']} min" if j.get("minutes") is not None else (j.get("started") or "")
            rows.append(f"| {j['id']} | {j['name']} | {icons.get(j['status'], '')} {j['status']} "
                        f"| {steps} | {j['added'][5:]} | {t} |")
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
        self._show_current(next((j for j in jobs if j["status"] in ACTIVE), None))

    def _show_current(self, job):
        if job is None:
            self.cur_title.content = "No job running" + (" (queue paused)" if self.s.paused else "")
            self.cur_bar.value = 0.0
            self.cur_log.content = ""
            return
        p = _progress(Path(job["log"]))
        total = job.get("training_steps") or 1
        step = p.get("step", 0)
        elapsed = time.time() - job.get("started_ts", time.time())
        rate = step / elapsed if step and elapsed > 0 else 0
        eta = f", ETA {_fmt_dur((total - step) / rate)}" if rate else ""
        loss = f" · loss {p['loss']:.4f}" if "loss" in p else ""
        self.cur_title.content = (f"**#{job['id']} {job['name']}** — step {step}/{total}{loss}  \n"
                                  f"{_fmt_dur(elapsed)} elapsed{eta}" + (f" · {rate:.1f} it/s" if rate else ""))
        self.cur_bar.value = 100.0 * step / total
        # últimas líneas del log sin la avalancha de 'step:' (solo la última)
        lines = [line for line in p["lines"] if not line.startswith("step: ")][-8:]
        self.cur_log.content = "```\n" + "\n".join(lines) + "\n```" if lines else ""

    # ── pestaña New job ──────────────────────────────────────────────────────
    def _build_form(self):
        gui = self.v.gui
        bases = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "train").glob("*.yaml")
                       if not ({"runs", "sweep"} & set(yaml.safe_load(p.read_text()) or {})))
        registry = load_registry()
        self.datasets = {k: d for k, d in (registry.get("datasets") or {}).items() if d.get("episodes")}
        arrangements = load_camera_setup()["arrangements"]
        self.arrangements = arrangements

        self.f_name = gui.add_text("Name", "", hint="Short name, e.g. 3cam-h32")
        self.f_desc = gui.add_text("Description", "", hint="What this model tries to improve")
        self.f_base = gui.add_dropdown("Base config", bases, initial_value="configs/train/diffusion.yaml"
                                       if "configs/train/diffusion.yaml" in bases else bases[0])
        self.f_dataset = gui.add_dropdown(
            "Dataset", ["(base config)", *sorted(self.datasets, key=lambda k: self.datasets[k]["created"], reverse=True)])
        self.f_arr = gui.add_dropdown("Arrangement", ["(base config)", *arrangements])
        self.f_cams = gui.add_text("Cameras", "", hint="Instead of an arrangement: base_rgb, wrist_rgb")
        self.f_nums = {k: gui.add_number(k, 0, step=1) for k in FORM_FIELDS}
        self.f_aug = gui.add_checkbox("image_augmentation", True)
        self.f_hub = gui.add_checkbox("push_to_hub", False)
        self.f_extra = gui.add_text("Extra (YAML)", "", multiline=True,
                                    hint="Any other training field, e.g.  vision_backbone: resnet34")
        self.f_info = gui.add_markdown("")
        add = gui.add_button("Add to queue", color="green", icon=viser.Icon.PLUS)

        self.f_base.on_update(lambda _: self._load_base())
        self.f_dataset.on_update(lambda _: self._show_dataset())
        add.on_click(lambda _: self._add_job())
        self._load_base()

    def _base_cfg(self) -> dict:
        return yaml.safe_load((ROOT / self.f_base.value).read_text()) or {}

    def _load_base(self):
        cfg = self._base_cfg()
        for k, h in self.f_nums.items():
            h.value = int(cfg.get(k, 0) or 0)
        self.f_aug.value = bool(cfg.get("image_augmentation", True))
        self.f_hub.value = bool(cfg.get("push_to_hub", False))
        self._show_dataset()

    def _show_dataset(self):
        ds = self.datasets.get(self.f_dataset.value)
        if ds is None:
            root = self._base_cfg().get("dataset_root", "")
            self.f_info.content = f"dataset `{Path(root).name}`"
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
        job = self.s.add(name, self.f_desc.value.strip(), self.f_base.value, overrides)
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
        ds_cams = (ds or {}).get("cameras") or self._base_dataset_cameras(merged)
        if wanted and ds_cams is not None and set(wanted) - set(ds_cams):
            raise ValueError(f"the dataset has no {sorted(set(wanted) - set(ds_cams))} "
                             f"(it has {', '.join(ds_cams)})")
        return name, {k: v for k, v in values.items() if base.get(k) != v}

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
    panel = TrainPanel(store, worker, args.port)
    threading.Thread(target=worker.loop, daemon=True).start()
    try:
        while True:
            panel.refresh()
            time.sleep(1.0)
    except KeyboardInterrupt:
        running = store.running()
        if running:
            print(f"\n[Train] Panel cerrado; #{running['id']} {running['name']} sigue entrenando "
                  f"(pid {running['pid']}). Vuelve a abrir el panel para seguirlo.")


if __name__ == "__main__":
    main(tyro.cli(Args))
