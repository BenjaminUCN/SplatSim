"""Cola de trabajos en la nube: un pod por trabajo, en paralelo hasta max_pods. Un trabajo puede
esperar a otro (depends_on), p.ej. entrenar con el dataset que deja una conversión. Lo propio de cada
tipo de trabajo está en stages.py.

    pending → preparing (sube entradas, crea el pod) → starting → running → finishing → done
                                                                                      ↘ failed / stopped
Un splat baja al lab y queda en aligning hasta que approve() comprueba la alineación en CloudCompare;
lo que dependa de él espera hasta entonces.

El estado vive en <qdir>/jobs.yaml, así que el panel se puede cerrar: los pods siguen, y al volver
se retoman. Un pod nunca depende de este proceso para detenerse (ver runner.py).
"""
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from scripts.xarm5.data_jobs import dataset_path
from splatsim.cloud.estimate import (SEED_HISTORY, Estimate, Workload, convert_workload, prefer_machines, rank,
                                     splat_workload, train_workload)
from splatsim.cloud.providers import Offer, Requirements
from splatsim.cloud.runner import MAX_BOOTS, sha256
from splatsim.cloud.stages import (DEFAULT_SCRIPTS, STAGES, alignment_problems, describe_pipeline, probe_video,
                                   scan_recordings)
from splatsim.cloud.storage import Storage, item_label
from splatsim.utils.paths import DATASETS_DIR, contract

ACTIVE = ("preparing", "starting", "running", "finishing")
RETRYABLE = ("failed", "stopped")
MAX_ATTEMPTS = 3
STILL_S = 180                 # sin cambios por más de esto, el panel lo avisa; tres latidos perdidos no bastan
BOOT_TIMEOUT_S = 25 * 60        # sin señales del pod (imagen lenta o máquina rota): otra máquina
STALE_TIMEOUT_S = 10 * 60       # sin latidos con el pod ya corriendo
EXIT_GRACE_S = 120              # el pod terminó, pero result.json puede tardar en verse
WAIT_RETRY_S = 120              # sin ofertas o sin saldo: cuánto esperar antes de volver a mirar
DISK_MARGIN_GB = 20
OFFERS_TO_TRY = 5
MACHINE_MEMORY_S = 14 * 24 * 3600   # después, el host pudo borrar la imagen o haberse arreglado
RESET_ON_RETRY = ("offer", "estimate", "phase", "progress", "run_ts", "started_ts", "billing_ts", "boot_s", "result",
                  "settled", "phase_since", "moved_ts", "detail")
POD_PYTHON = "/opt/venv/bin/python"
SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)   # jobs.yaml guarda los sha256 de cada salida: pesa
GPU_CHECK = [POD_PYTHON, "-c",     # 1 GB en la GPU: detecta hosts como el del "out of memory" en RunPod
             "import torch; x = torch.ones(1024, 1024, 256, device='cuda'); assert (x * 2).sum() > 0"]
DEFAULT_IMAGE = "ghcr.io/benjaminucn/splatsim-train:latest"
DEFAULT_RENDER_IMAGE = "ghcr.io/benjaminucn/splatsim-render:latest"
POD_ROOT = "/workspace"
DATASETS_ROOT = DATASETS_DIR   # nombre con que lo usa cloud_server.py
SCENES_ROOT = "~/data/Colmaps"
ALIGNING = "aligning"


def train_config(base: dict, name: str, dataset: str, overrides: dict, has_wandb_key: bool) -> dict:
    """Config final de un entrenamiento. Conserva las rutas con ~: valen en el lab y en el pod."""
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError(f"campos desconocidos: {sorted(unknown)}")
    dataset_name = Path(dataset).name
    safe = re.sub(r"[^\w.+-]+", "_", name)
    return {**base, "push_to_hub": False,    # el pod no tiene token de Hugging Face
            "wandb_mode": "online" if has_wandb_key else "offline", **overrides,
            "name": name, "model_name": f"{safe}_{datetime.now():%m%d_%H%M}",
            "dataset_id": f"local/{dataset_name}", "dataset_root": f"{DATASETS_ROOT}/{dataset_name}"}


@dataclass
class Settings:
    max_pods: int = 2
    budget_usd: float = 5.0       # gasto máximo de toda la cola, comprometido incluido
    paused: bool = False


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 60 else f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def describe_progress(job: dict, now: float | None = None) -> str:
    """Para el panel: en qué va, desde cuándo, y si lleva rato sin moverse."""
    if job["status"] not in ACTIVE:
        return job.get("wait") or "" if job["status"] == "pending" else job.get("phase") or ""
    now = time.time() if now is None else now
    p = job.get("progress") or {}
    parts = [job.get("phase") or job.get("wait") or ""]
    if "done" in p:
        total = p.get("total") or (job.get("config") or {}).get("training_steps")
        parts.append(f"{p['done']:,}/{total:,}" if isinstance(total, int) else f"{p['done']:,}/?")
        if p.get("rate"):
            parts.append(f"{p['rate']:.2f}/s")
    if job.get("detail"):
        parts.append(job["detail"])
    if "phase_since" in job:
        parts.append(_duration(now - job["phase_since"]))
    if ("done" in p or job.get("detail")) and now - job.get("moved_ts", now) > STILL_S:
        parts.append(f"⚠️ no change for {_duration(now - job['moved_ts'])}")
    return " · ".join(parts)


def folder_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def disk_for(work: Workload) -> int:
    return math.ceil(work.down_gb - work.image_gb + work.image_disk_gb + work.up_gb + DISK_MARGIN_GB)


class Orchestrator:
    def __init__(self, qdir: str | Path, provider, storage: Storage, pod_env: dict[str, str] | None = None,
                 image: str = DEFAULT_IMAGE, render_image: str = DEFAULT_RENDER_IMAGE, health: list[str] | None = GPU_CHECK,
                 on_ready=None, python: str = POD_PYTHON, scripts: dict[str, str] | None = None,
                 pod_root: str = POD_ROOT):
        self.qdir = Path(qdir).expanduser()
        self.qdir.mkdir(parents=True, exist_ok=True)
        self.python, self.pod_root = python, pod_root
        self.scripts = {**DEFAULT_SCRIPTS, **(scripts or {})}
        # cada cola solo responde por sus pods: otra cola o una prueba aparte no son huérfanos suyos. Con ~,
        # la misma cola compartida por R2 (splatsim/cloud/sync.py) tiene el mismo prefijo en cada PC.
        self.label_prefix = f"splatsim-{hashlib.sha1(contract(str(self.qdir)).encode()).hexdigest()[:6]}-"
        self.provider, self.storage, self.health = provider, storage, health
        self.images = {"train": image, "render": render_image}
        self.pod_env = pod_env or {}
        self.on_ready = on_ready
        self.lock = threading.RLock()
        self.read_only = False          # otro panel avanza la cola (sync.py): este no la escribe
        self.busy: set[int] = set()      # trabajos con un hilo de preparación o descarga en curso
        self.warnings: list[str] = []
        self.load()

    # ── estado ───────────────────────────────────────────────────────────────
    def load(self, resume: bool = True) -> None:
        """Lee la cola de qdir. resume: este panel la va a avanzar, así que lo que hacía un hilo de otra
        sesión vuelve a empezar; un panel de solo lectura la muestra tal cual."""
        path = self.qdir / "jobs.yaml"
        data = (yaml.load(path.read_text(), Loader=SafeLoader) if path.exists() else None) or {}
        path = self.qdir / "history.yaml"
        history = (yaml.safe_load(path.read_text()) if path.exists() else None) or []
        path = self.qdir / "machines.yaml"
        machines = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
        with self.lock:
            self.jobs: list[dict] = data.get("jobs") or []
            self.settings = Settings(**(data.get("settings") or {}))
            self.next_id = max((j["id"] for j in self.jobs), default=0) + 1
            self.history: list[dict] = history
            self.machines: dict[int, dict] = machines
            self._orphans_checked = False
            if resume:
                for job in self.jobs:
                    if job["status"] == "preparing":
                        job["status"] = "pending"

    def save(self) -> None:
        with self.lock:
            if self.read_only:
                return
            tmp = self.qdir / "jobs.tmp"
            tmp.write_text(yaml.safe_dump({"settings": asdict(self.settings), "jobs": self.jobs},
                                          sort_keys=False, allow_unicode=True))
            os.replace(tmp, self.qdir / "jobs.yaml")

    def get(self, job_id: int) -> dict | None:
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def known_history(self) -> list[dict]:
        return SEED_HISTORY + self.history

    def spent(self) -> float:
        return sum(j.get("cost") or 0.0 for j in self.jobs)

    def committed(self) -> float:
        """Lo que aún pueden gastar los trabajos activos, según el máximo de cada uno."""
        return sum(max(j["max_cost"] - (j.get("cost") or 0.0), 0.0) for j in self.jobs if j["status"] in ACTIVE)

    # ── encolar ──────────────────────────────────────────────────────────────
    @staticmethod
    def workload(job: dict) -> Workload:
        return STAGES[job["kind"]].workload(job)

    def rank_offers(self, work: Workload, req: Requirements) -> list[Estimate]:
        """Ofertas disponibles ahora para este trabajo, de la mejor a la peor."""
        return self._ranked(work, self.provider.search(req, disk_for(work)))

    def _ranked(self, work: Workload, offers: list[Offer]) -> list[Estimate]:
        since = time.time() - MACHINE_MEMORY_S
        recent = {machine: r["ok"] for machine, r in self.machines.items() if r["ts"] > since}
        return prefer_machines(rank(self.known_history(), work, offers),
                               {m for m, ok in recent.items() if ok}, {m for m, ok in recent.items() if not ok})

    def _note_machine(self, machine: int, ok: bool, boot_s: int | None = None) -> None:
        """Solo el último resultado de cada máquina. El arranque se guarda para ver si la preferencia sirve."""
        self.machines[machine] = {"ok": ok, "ts": time.time(), "boot_s": boot_s,
                                  "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
        tmp = self.qdir / "machines.tmp"
        tmp.write_text(yaml.safe_dump(self.machines, sort_keys=False))
        os.replace(tmp, self.qdir / "machines.yaml")

    def _note_boot(self, job: dict) -> None:
        """La primera señal del pod: la imagen ya está en esa máquina."""
        if job["status"] == "starting":
            job["boot_s"] = round(time.time() - job["started_ts"])
            self._note_machine(job["offer"]["machine_id"], True, job["boot_s"])

    def estimate(self, cfg: dict, dataset: str, req: Requirements, depends_on: int | None = None) -> list[Estimate]:
        return self.rank_offers(train_workload(cfg, self._dataset_gb(dataset, depends_on)), req)

    def estimate_convert(self, args: dict, req: Requirements) -> tuple[dict, list[Estimate]]:
        scan = scan_recordings(args["input"], args.get("cameras"), args.get("ignore_selection", False))
        work = convert_workload(scan["frames"], len(scan["cameras"]), scan["gb"])
        return scan, self.rank_offers(work, req)

    def estimate_splat(self, args: dict, req: Requirements) -> tuple[dict, list[Estimate]]:
        probe = probe_video(args["source"], args.get("fps", 0.0))
        work = splat_workload(probe["frames"], args.get("matcher", "exhaustive"), args["iterations"], probe["gb"])
        return probe, self.rank_offers(work, req)

    def submit_splat(self, name: str, scene: str, args: dict, req: Requirements, max_cost: float,
                     offer_id: int | None = None, scenes_root: str = SCENES_ROOT) -> dict:
        """args: source (video o carpeta de fotos), iterations y opcionalmente fps y matcher."""
        if not re.fullmatch(r"[\w.+-]+", scene):
            raise ValueError(f"Nombre de escena inválido: {scene!r}")
        output = str(Path(scenes_root).expanduser() / scene)
        taken = any(j.get("output") == output and j["status"] not in RETRYABLE for j in self.jobs)
        if taken or Path(output).exists():
            raise ValueError(f"Ya existe o está en cola la escena {output}")
        probe = probe_video(args["source"], args.get("fps", 0.0))
        return self._add({"name": name, "description": "", "kind": "splat", "args": dict(args), "output": output,
                          "frames": probe["frames"], "source_gb": probe["gb"], "depends_on": None},
                         req, max_cost, offer_id)

    def estimate_pipeline(self, plan: list[dict], gpu: Requirements, cpu: Requirements) -> list[tuple[dict, list[Estimate]]]:
        """Por cada etapa de plan_pipeline (data_jobs.py), lo que hará y sus ofertas."""
        return [(s, self.rank_offers(self.workload(s), gpu if STAGES[s["kind"]].needs_gpu else cpu))
                for s in describe_pipeline(plan)]

    def submit_pipeline(self, plan: list[dict], gpu: Requirements, cpu: Requirements, max_cost: float,
                        keep_output: bool = False) -> list[dict]:
        """Encola las etapas encadenadas. Lo intermedio queda solo en el bucket: al lab baja lo último, salvo
        con keep_output, que lo deja en el bucket para bajarlo después (download)."""
        stages = describe_pipeline(plan)
        for stage in stages:
            if Path(stage["output"]).expanduser().exists() or any(
                    j.get("output") == stage["output"] and j["status"] not in RETRYABLE for j in self.jobs):
                raise ValueError(f"Ya existe o está en cola {stage['output']}")
        jobs: list[dict] = []
        for i, stage in enumerate(stages):
            req = gpu if STAGES[stage["kind"]].needs_gpu else cpu
            jobs.append(self._add({**stage, "description": "", "depends_on": jobs[-1]["id"] if jobs else None,
                                   **(_last_stage(keep_output) if i == len(stages) - 1 else {"download": False})},
                                  req, max_cost, None))
        return jobs

    def _dataset_gb(self, dataset: str, depends_on: int | None) -> float:
        """El dataset de una cadena aún no existe: se usa el tamaño previsto por la conversión."""
        if depends_on is None:
            path = Path(dataset).expanduser()
            if not path.is_dir():
                raise ValueError(f"No existe el dataset {path}")
            return folder_gb(path)
        parent = self.get(depends_on)
        if parent is None or parent["kind"] != "convert" or Path(parent["output"]) != Path(dataset).expanduser():
            raise ValueError(f"El trabajo #{depends_on} no deja el dataset {dataset}")
        return self.workload(parent).up_gb

    def _add(self, fields: dict, req: Requirements, max_cost: float, offer_id: int | None) -> dict:
        if max_cost <= 0:
            raise ValueError("max_cost debe ser mayor que 0")
        with self.lock:
            job = {"id": self.next_id, "status": "pending", **fields, "requirements": asdict(req),
                   "max_cost": float(max_cost), "offer_id": offer_id, "attempts": 0, "excluded_machines": [],
                   "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
            self.next_id += 1
            self.jobs.append(job)
            self.save()
            return job

    def submit(self, name: str, cfg: dict, dataset: str, req: Requirements, max_cost: float,
               offer_id: int | None = None, description: str = "", depends_on: int | None = None,
               keep_output: bool = False) -> dict:
        gb = round(self._dataset_gb(dataset, depends_on), 2)
        return self._add({"name": name, "description": description, "kind": "train", "config": cfg,
                          "dataset": str(dataset), "dataset_gb": gb, "depends_on": depends_on,
                          **_last_stage(keep_output)}, req, max_cost, offer_id)

    def submit_convert(self, name: str, args: dict, req: Requirements, max_cost: float,
                       offer_id: int | None = None, output_dir: str = DATASETS_ROOT) -> dict:
        """args: input, fps, suffix y opcionalmente cameras e ignore_selection."""
        output = dataset_path(output_dir, args["fps"], args["suffix"])
        taken = any(j.get("output") == output and j["status"] not in RETRYABLE for j in self.jobs)
        if taken or Path(output).exists():
            raise ValueError(f"Ya existe o está en cola el dataset {output}: cambia el sufijo")
        scan = scan_recordings(args["input"], args.get("cameras"), args.get("ignore_selection", False))
        return self._add({"name": name, "description": "", "kind": "convert", "args": dict(args), "output": output,
                          "frames": scan["frames"], "cameras": scan["cameras"], "recordings_gb": scan["gb"],
                          "exclude": scan["exclude"], "depends_on": None}, req, max_cost, offer_id)

    # ── ciclo ────────────────────────────────────────────────────────────────
    def tick(self) -> None:
        with self.lock:
            instances = None
            # con un hilo trabajando, un pod recién creado todavía no tiene su trabajo: no es huérfano
            check_orphans = not self._orphans_checked and not self.busy
            if any(j["status"] in ("starting", "running") for j in self.jobs) or check_orphans:
                instances = {i.id: i for i in self.provider.instances()}
            if check_orphans:
                self._destroy_orphans(instances)
            for job in self.jobs:
                if job["id"] in self.busy:
                    continue
                if job["status"] in ("starting", "running") and instances is not None:
                    self._check(job, instances.get(job.get("instance_id")))
                elif job["status"] == "finishing":
                    self._spawn(job, self._finish)
            if not self.settings.paused:
                self._launch_ready()

    def _spawn(self, job: dict, target) -> None:
        self.busy.add(job["id"])

        def work():
            try:
                target(job)
            except Exception as e:  # noqa: BLE001  un trabajo roto no debe parar la cola
                self._fail(job, f"{type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.busy.discard(job["id"])
                    self.save()

        threading.Thread(target=work, daemon=True).start()

    def _destroy_orphans(self, instances: dict) -> None:
        """Pods nuestros que ningún trabajo conoce (p.ej. se cerró el panel al crearlos): cobran."""
        known = {j.get("instance_id") for j in self.jobs if j["status"] in ACTIVE}
        self._orphans_checked = True
        for instance in instances.values():
            if instance.label.startswith(self.label_prefix) and instance.id not in known:
                try:
                    self.provider.destroy(instance.id)
                    self.warnings.append(f"Eliminado el pod huérfano {instance.id} ({instance.label})")
                except RuntimeError as e:
                    self._orphans_checked = False     # se reintenta en el siguiente ciclo
                    self.warnings.append(f"No se pudo eliminar el pod {instance.id}: {e}")

    def _launch_ready(self) -> None:
        active = sum(j["status"] in ACTIVE for j in self.jobs)
        for job in self.jobs:
            if active >= self.settings.max_pods:
                return
            if job["status"] != "pending" or job["id"] in self.busy or job.get("retry_at", 0) > time.time():
                continue
            if waiting_for := self._unfinished_parent(job):
                job["wait"] = f"espera al trabajo #{waiting_for['id']} ({waiting_for['status']})"
                continue
            spent = self.spent() + self.committed()
            if spent + job["max_cost"] > self.settings.budget_usd:
                job["wait"] = (f"presupuesto: comprometidos ${spent:.2f} + máximo ${job['max_cost']:.2f} "
                               f"> ${self.settings.budget_usd:.2f}")
                continue
            job["status"] = "preparing"
            active += 1
            self._spawn(job, self._prepare)
        self.save()

    def _unfinished_parent(self, job: dict) -> dict | None:
        """Si el padre falla, el hijo espera: el padre todavía se puede reintentar."""
        parent = self.get(job.get("depends_on")) if job.get("depends_on") is not None else None
        return parent if parent and parent["status"] != "done" else None

    # ── preparar y lanzar ────────────────────────────────────────────────────
    def _run_key(self, job: dict) -> str:
        """Siempre una carpeta sin usar: Retry reinicia los intentos, y un pod nuevo que encuentra el estado
        o el resultado de otro anterior parece haber arrancado o terminado."""
        safe = re.sub(r"[^\w.+-]+", "_", job["name"])
        attempt = job["attempts"]
        while self.storage.read(key := f"runs/{job['id']:04d}_{attempt}_{safe}/spec.json") is not None:
            attempt += 1
        return key.removesuffix("/spec.json")

    def _pick_offers(self, job: dict) -> list[Estimate]:
        req, work = Requirements(**job["requirements"]), self.workload(job)
        offers = [o for o in self.provider.search(req, disk_for(work))
                  if o.machine_id not in job["excluded_machines"]]
        ranked = self._ranked(work, offers)
        ranked.sort(key=lambda e: e.offer.id != job.get("offer_id"))     # la elegida, si sigue disponible
        left = self._budget_left(job)     # sin estimación se sabe al menos lo que costará la transferencia
        return [e for e in ranked if (e.total if e.total is not None else e.transfer_cost) <= left]

    @staticmethod
    def _budget_left(job: dict) -> float:
        return job["max_cost"] - job.get("cost_before", 0.0)

    def _spec(self, job: dict, estimate: Estimate, run_key: str) -> dict:
        stage = STAGES[job["kind"]]
        pod = stage.pod(job, run_key, self.pod_root)
        deadline_s = max(self._budget_left(job) - estimate.transfer_cost, 0.0) / estimate.offer.dph * 3600
        return {"command": [self.python, "-u", self.scripts[job["kind"]], *pod.pop("args")],
                "health": self.health if stage.needs_gpu else None, **pod, "deadline_s": deadline_s}

    def _prepare(self, job: dict) -> None:
        with self.lock:
            job["attempts"] += 1
            job["run_key"] = run_key = self._run_key(job)
        stage = STAGES[job["kind"]]
        # la cola pudo llegar de otro PC (sync.py): sin sus archivos, espera en vez de fallar
        missing = [item["local"] for item in stage.uploads(job) if not Path(item["local"]).expanduser().exists()]
        if missing:
            self._wait(job, f"falta en este PC: {contract(str(Path(missing[0]).expanduser()))}"
                            + (f" y {len(missing) - 1} más" if len(missing) > 1 else ""))
            return
        stage.refresh(job, folder_gb)
        self._enter_phase(job, "uploading inputs")
        uploads = stage.uploads(job)
        for i, item in enumerate(uploads, 1):  # lo que ya está en el bucket no se vuelve a subir
            local, key = Path(item["local"]).expanduser(), item["key"]
            self._note_detail(job, item_label(i, len(uploads), key))
            if local.is_file():
                self.storage.upload_file(local, key)
            else:
                self.storage.upload(local, key, item.get("exclude"), packed=item.get("packed", False),
                                    progress=lambda done, total, i=i, key=key: self._note_detail(
                                        job, item_label(i, len(uploads), key, done, total)))

        candidates = self._pick_offers(job)
        if not candidates:
            self._wait(job, "sin ofertas dentro del máximo y los requisitos")
            return
        balance = self.provider.balance()
        if balance < job["max_cost"]:
            self._wait(job, f"saldo ${balance:.2f} < máximo ${job['max_cost']:.2f}")
            return
        env = {**self.pod_env, "SPLATSIM_REMOTE": self.storage.remote}
        label = self.label_prefix + run_key.removeprefix("runs/")
        error = None
        for candidate in candidates[:OFFERS_TO_TRY]:
            self.storage.write(f"{run_key}/spec.json", json.dumps(self._spec(job, candidate, run_key)))
            try:
                instance = self.provider.create(candidate.offer, self.images[stage.image], [run_key], env,
                                                disk_for(self.workload(job)), label)
            except RuntimeError as e:          # la oferta ya no está: la siguiente
                error = e
                print(f"[Cloud] oferta {candidate.offer.id} ({candidate.offer.cpu_name}): {e}", flush=True)
                continue
            with self.lock:
                job.update(status="starting", instance_id=instance, offer=asdict(candidate.offer),
                           estimate=_estimate_fields(candidate), started_ts=time.time(), seen_ts=time.time(),
                           seq=-1, started=datetime.now().strftime("%Y-%m-%d %H:%M"))
                self._enter_phase(job, "waiting for the pod")
                job.pop("wait", None)
                job.pop("retry_at", None)
                self.save()
            return
        raise RuntimeError(f"no se pudo crear el pod: {error}")

    @staticmethod
    def _enter_phase(job: dict, name: str) -> None:
        now = time.time()
        job.update(phase=name, phase_since=now, moved_ts=now)
        job.pop("detail", None)

    @staticmethod
    def _note_detail(job: dict, text: str) -> None:
        job.update(detail=text, moved_ts=time.time())

    def _wait(self, job: dict, reason: str) -> None:
        """Vuelve a la cola sin gastar un intento; se reintenta más tarde, no en cada ciclo."""
        with self.lock:
            job.update(status="pending", wait=reason, attempts=job["attempts"] - 1,
                       retry_at=time.time() + WAIT_RETRY_S)
            for key in ("phase", "phase_since", "moved_ts", "detail"):
                job.pop(key, None)      # el panel muestra la fase antes que el motivo de la espera

    # ── seguimiento ──────────────────────────────────────────────────────────
    def _check(self, job: dict, instance) -> None:
        if "billing_ts" not in job and instance is not None and instance.status in ("running", "exited"):
            job["billing_ts"] = time.time()
        result = self.storage.read(f"{job['run_key']}/result.json")
        if result:
            job["result"] = json.loads(result)    # antes del costo: dice cuánto corrió de verdad
            self._update_cost(job)
            self._note_boot(job)
            self._on_result(job, job["result"])
            return
        self._update_cost(job)
        now = time.time()
        status = self.storage.read(f"{job['run_key']}/status.json")
        if status:
            status = json.loads(status)
            if status["seq"] != job.get("seq"):
                progress = {k: status[k] for k in ("done", "total", "rate") if k in status}
                if status["phase"] != job.get("phase"):
                    self._enter_phase(job, status["phase"])
                elif (progress.get("done"), status.get("detail")) != ((job.get("progress") or {}).get("done"),
                                                                      job.get("detail")):
                    job["moved_ts"] = now
                job.update(seq=status["seq"], seen_ts=now, progress=progress)
                job.pop("detail", None)
                if status.get("detail"):
                    job["detail"] = status["detail"]
                self._note_boot(job)
                job["status"] = "running"
                job.setdefault("billing_ts", now)
                if status["phase"] == "run" and "run_ts" not in job:
                    job["run_ts"] = now
            if status.get("boot", 1) > MAX_BOOTS + 1:     # el runner debió cortarlo: no seguir pagando
                self._retry_or_fail(job, "el pod se reinicia en bucle")
                self.save()
                return
        if job.get("cost", 0.0) > job["max_cost"]:
            self._release(job)
            self._fail(job, f"superó el máximo de ${job['max_cost']:.2f}")
        elif instance is None:
            self._retry_or_fail(job, "el pod desapareció")
        elif instance.status == "exited" and now - job["seen_ts"] > EXIT_GRACE_S:
            self._retry_or_fail(job, "el pod terminó sin dejar resultado")
        elif job["status"] == "starting" and now - job["started_ts"] > BOOT_TIMEOUT_S:
            self._retry_or_fail(job, "el pod no arrancó a tiempo")
        elif job["status"] == "running" and now - job["seen_ts"] > STALE_TIMEOUT_S:
            self._retry_or_fail(job, "el pod dejó de dar señales")
        self.save()

    def _on_result(self, job: dict, result: dict) -> None:
        job["result"] = result
        if result["ok"]:
            job["status"] = "finishing"
            self._spawn(job, self._finish)
        elif result["reason"] == "host_bad":
            self._retry_or_fail(job, f"host defectuoso: {result.get('error', '')}")
        else:
            self._release(job)
            self._fail(job, f"{result['reason']}: {result.get('error') or 'ver run.log'}")

    def _release(self, job: dict) -> None:
        """Un pod que no se deja eliminar no debe tumbar el trabajo: queda como huérfano y se reintenta."""
        if job.get("instance_id") is not None:
            try:
                self.provider.destroy(job["instance_id"])
            except RuntimeError as e:
                self._orphans_checked = False
                self.warnings.append(f"No se pudo eliminar el pod {job['instance_id']}: {e}")
            self._update_cost(job)
            job["instance_id"] = None

    def _retry_or_fail(self, job: dict, reason: str) -> None:
        machine = (job.get("offer") or {}).get("machine_id")
        self._release(job)
        self._settle(job)
        job["cost_before"] = job.get("cost") or 0.0
        if machine is not None:
            job["excluded_machines"].append(machine)
            self._note_machine(machine, False)
        if job["attempts"] >= MAX_ATTEMPTS:
            self._fail(job, f"{reason} ({job['attempts']} intentos)")
            return
        job.update(status="pending", wait=f"reintento: {reason}")
        for key in RESET_ON_RETRY:
            job.pop(key, None)
        self.save()

    def _fail(self, job: dict, reason: str) -> None:
        with self.lock:
            self._settle(job)
            job.update(status="failed", error=reason, finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
            for key in ("phase", "phase_since", "moved_ts", "detail"):
                job.pop(key, None)      # si no, el panel sigue mostrando la fase en la que falló
            self.save()

    def _update_cost(self, job: dict) -> None:
        """Horas corriendo por su precio más la transferencia prevista: la descarga de la imagen no se
        cobra por hora. El valor exacto sale en la factura."""
        if "offer" not in job:
            return
        job["cost"] = round(job.get("cost_before", 0.0) + self._hours(job) * job["offer"]["dph"]
                            + job["estimate"]["transfer_cost"], 4)

    @staticmethod
    def _hours(job: dict) -> float:
        """Lo que el pod dice haber corrido, si ya terminó: el panel puede enterarse horas después."""
        ran = (job.get("result") or {}).get("total_s")
        if ran is not None:
            return ran / 3600
        return (time.time() - job["billing_ts"]) / 3600 if "billing_ts" in job else 0.0

    def _settle(self, job: dict) -> None:
        """Un pod que no terminó no subió sus salidas: se queda lo que corrió y lo que bajó, no lo previsto."""
        if "offer" not in job or job.get("settled") or (job.get("result") or {}).get("ok"):
            return
        job["cost"] = round(job.get("cost_before", 0.0) + self._hours(job) * job["offer"]["dph"]
                            + self.workload(job).down_gb * job["offer"]["down_cost_gb"], 4)
        job["settled"] = True

    # ── descargar ────────────────────────────────────────────────────────────
    def _finish(self, job: dict) -> None:
        """Se puede repetir: el bucket sigue teniendo las salidas si el lab falla o se llena."""
        self._enter_phase(job, "downloading")
        self._release(job)
        self.save()
        stage = STAGES[job["kind"]]
        key, target = stage.output_key(job), stage.target(job)
        if job.get("download", True):
            need = self.workload(job).up_gb
            free = shutil.disk_usage(Path.home()).free / 1e9
            if free < need + 5:
                raise RuntimeError(f"solo hay {free:.0f} GB libres en el lab; la salida sigue en el bucket: {key}")
            self.storage.download(key, target, progress=lambda done, total: self._note_detail(
                job, item_label(1, 1, key, done, total)))
            outputs = job["result"]["outputs"][key]
            missing = [name for name in outputs if not (target / name).is_file()]
            if missing:
                raise RuntimeError(f"no llegaron {missing[:3]}; la salida sigue en el bucket: {key}")
            bad = [name for name, digest in outputs.items() if sha256(target / name) != digest]
            if bad:
                raise RuntimeError(f"sha256 distinto en {bad[:3]}; la salida sigue en el bucket: {key}")
        if publish := stage.publish_key(job):
            self.storage.copy(key, publish)
        # con una etapa esperándola, la salida es su entrada: la borra esa etapa al terminar
        has_child = stage.output_feeds_children and any(j.get("depends_on") == job["id"] for j in self.jobs)
        keep = job.get("keep_output", False)
        self._clean_bucket(job, None if has_child or keep else key)
        with self.lock:
            self._record_history(job)
            if job.get("download", True):
                job["output_dir"] = str(target)
            elif keep:
                job["remote_output"] = key
            if getattr(stage, "needs_approval", False):
                self._enter_phase(job, "align it in CloudCompare, then Approve")
                job.update(status=ALIGNING, downloaded=datetime.now().strftime("%Y-%m-%d %H:%M"))
                self.save()
                return
            job.update(status="done", phase="done",
                       finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.save()
        if self.on_ready:
            self.on_ready(job)

    def approve(self, job_id: int, robot: str) -> list[str]:
        """Da por alineada la escena de un splat si objects.yaml y las etiquetas están al día; si no,
        devuelve lo que falta y el trabajo sigue esperando."""
        job = self.get(job_id)
        if job is None or job["status"] != ALIGNING:
            return [f"el trabajo #{job_id} no espera una alineación"]
        problems = alignment_problems(robot, job["output"])
        if problems:
            return problems
        with self.lock:
            job.update(status="done", phase="done", robot_name=robot,
                       finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.save()
        if self.on_ready:
            self.on_ready(job)
        return []

    def _clean_bucket(self, job: dict, output_key: str | None) -> None:
        """Lo que ya llegó verificado no sigue cobrando. Si falla, solo queda el aviso."""
        stage = STAGES[job["kind"]]
        others = {key for j in self.jobs if j is not job and j["status"] not in ("done", *RETRYABLE)
                  for key in STAGES[j["kind"]].temporary_inputs(j)}
        for key in [*([output_key] if output_key else []), *(k for k in stage.temporary_inputs(job) if k not in others)]:
            try:
                self.storage.delete(key)
            except RuntimeError as e:
                self.warnings.append(f"No se pudo borrar {key} del bucket: {e}")

    def _record_history(self, job: dict) -> None:
        rate = STAGES[job["kind"]].rate(job)
        if not rate or job.get("recorded"):      # download() repite _finish: el pod corrió una vez
            return
        job["recorded"] = True
        self.history.append({"kind": job["kind"], "profile": self.workload(job).profile,
                             "gpu": job["offer"]["gpu_name"],
                             "cpu": job["offer"]["cpu_name"], "rate": rate, "job": job["id"],
                             "seconds": job["result"].get("seconds"), "boot_s": job.get("boot_s"),
                             "machine": job["offer"]["machine_id"], "dph": job["offer"]["dph"],
                             "cost": job.get("cost"), "estimated_total": job["estimate"].get("total"),
                             "date": datetime.now().strftime("%Y-%m-%d")})
        (self.qdir / "history.yaml").write_text(yaml.safe_dump(self.history, sort_keys=False, allow_unicode=True))

    # ── acciones del panel ───────────────────────────────────────────────────
    def stop(self, job_id: int) -> None:
        with self.lock:
            job = self.get(job_id)
            if job and job["status"] in ACTIVE and job["id"] not in self.busy:
                self._release(job)
                self._settle(job)
                job.update(status="stopped", finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
                self.save()
            elif job and job["status"] == "pending":
                job["status"] = "stopped"
                self.save()

    def set_max_cost(self, job_id: int, value: float) -> bool:
        """Solo sin pod: el que está corriendo ya recibió su límite y se detiene solo al alcanzarlo."""
        with self.lock:
            job = self.get(job_id)
            if not job or value <= 0 or job["status"] in ACTIVE or job["status"] == "done":
                return False
            job["max_cost"] = float(value)
            job.pop("retry_at", None)
            self.save()
            return True

    def retry(self, job_id: int) -> None:
        with self.lock:
            job = self.get(job_id)
            if not job or job["status"] not in RETRYABLE:
                return
            job.pop("error", None)
            if (job.get("result") or {}).get("ok"):
                job["status"] = "finishing"     # solo falló la descarga: no se entrena de nuevo
            else:
                job.update(status="pending", attempts=0, excluded_machines=[], cost_before=job.get("cost") or 0.0)
                for key in RESET_ON_RETRY:
                    job.pop(key, None)
            self.save()

    def download(self, job_id: int) -> bool:
        """Baja a este PC una salida que se dejó en el bucket; la verificación y la limpieza son las de siempre."""
        with self.lock:
            job = self.get(job_id)
            if not job or job["status"] != "done" or not job.get("remote_output") or job["id"] in self.busy:
                return False
            job.update(status="finishing", download=True, keep_output=False)
            job.pop("remote_output", None)
            self.save()
            return True

    def remove(self, job_id: int) -> None:
        with self.lock:
            job = self.get(job_id)
            waiting = [j["id"] for j in self.jobs if j.get("depends_on") == job_id and j["status"] == "pending"]
            if job and job["status"] != "done" and waiting:
                self.warnings.append(f"El trabajo #{job_id} no se quitó: {waiting} esperan su resultado")
            elif job and job["status"] not in ACTIVE:
                self.jobs.remove(job)
                self.save()


def _last_stage(keep_output: bool) -> dict:
    """Lo que entrega un trabajo: baja a este PC, o queda en el bucket hasta pedirlo."""
    return {"download": not keep_output, "keep_output": keep_output}


def _estimate_fields(e: Estimate) -> dict:
    return {"rate": e.rate, "confidence": e.confidence, "source": e.source, "hours": e.hours,
            "compute_cost": e.compute_cost, "transfer_cost": e.transfer_cost, "total": e.total}
