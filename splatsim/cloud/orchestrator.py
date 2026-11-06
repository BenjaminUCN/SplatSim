"""Cola de trabajos en la nube: un pod por trabajo, en paralelo hasta max_pods.

    pending → preparing (sube entradas, crea el pod) → starting → running → finishing → done
                                                                                      ↘ failed / stopped

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

from splatsim.cloud.estimate import IMAGE_DISK_GB, IMAGE_GB, SEED_HISTORY, Estimate, Workload, rank, train_workload
from splatsim.cloud.providers import Offer, Requirements
from splatsim.cloud.runner import MAX_BOOTS, sha256
from splatsim.cloud.storage import Storage

ACTIVE = ("preparing", "starting", "running", "finishing")
RETRYABLE = ("failed", "stopped")
MAX_ATTEMPTS = 3
BOOT_TIMEOUT_S = 25 * 60        # sin señales del pod (imagen lenta o máquina rota): otra máquina
STALE_TIMEOUT_S = 10 * 60       # sin latidos con el pod ya corriendo
EXIT_GRACE_S = 120              # el pod terminó, pero result.json puede tardar en verse
WAIT_RETRY_S = 120              # sin ofertas o sin saldo: cuánto esperar antes de volver a mirar
DISK_MARGIN_GB = 20
OFFERS_TO_TRY = 5
POD_PYTHON = "/opt/venv/bin/python"
GPU_CHECK = [POD_PYTHON, "-c",     # 1 GB en la GPU: detecta hosts como el del "out of memory" en RunPod
             "import torch; x = torch.ones(1024, 1024, 256, device='cuda'); assert (x * 2).sum() > 0"]
DEFAULT_IMAGE = "ghcr.io/benjaminucn/splatsim-train:latest"
TRAIN_SCRIPT = "scripts/xarm5/train_diffusion.py"
DATASETS_ROOT = "~/lerobot/splatsim_test_data"


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


def folder_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def disk_for(work: Workload) -> int:
    return math.ceil(work.down_gb - IMAGE_GB + IMAGE_DISK_GB + work.up_gb + DISK_MARGIN_GB)


class Orchestrator:
    def __init__(self, qdir: str | Path, provider, storage: Storage, pod_env: dict[str, str] | None = None,
                 image: str = DEFAULT_IMAGE, health: list[str] | None = GPU_CHECK,
                 on_model_ready=None, python: str = POD_PYTHON, train_script: str = TRAIN_SCRIPT):
        self.qdir = Path(qdir).expanduser()
        self.qdir.mkdir(parents=True, exist_ok=True)
        self.python, self.train_script = python, train_script
        # cada cola solo responde por sus pods: otra cola o una prueba aparte no son huérfanos suyos
        self.label_prefix = f"splatsim-{hashlib.sha1(str(self.qdir).encode()).hexdigest()[:6]}-"
        self.provider, self.storage, self.image, self.health = provider, storage, image, health
        self.pod_env = pod_env or {}
        self.on_model_ready = on_model_ready
        self.lock = threading.RLock()
        self.busy: set[int] = set()      # trabajos con un hilo de preparación o descarga en curso
        self.warnings: list[str] = []
        data = yaml.safe_load((self.qdir / "jobs.yaml").read_text()) if (self.qdir / "jobs.yaml").exists() else {}
        data = data or {}
        self.jobs: list[dict] = data.get("jobs") or []
        self.settings = Settings(**(data.get("settings") or {}))
        self.next_id = max((j["id"] for j in self.jobs), default=0) + 1
        path = self.qdir / "history.yaml"
        self.history: list[dict] = (yaml.safe_load(path.read_text()) if path.exists() else None) or []
        self._orphans_checked = False
        for job in self.jobs:            # lo que hacía un hilo de la sesión anterior
            if job["status"] == "preparing":
                job["status"] = "pending"

    # ── estado ───────────────────────────────────────────────────────────────
    def save(self) -> None:
        with self.lock:
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
        return train_workload(job["config"], job["dataset_gb"])

    def estimate(self, cfg: dict, dataset: str, req: Requirements) -> list[Estimate]:
        """Ofertas disponibles ahora para este trabajo, de la mejor a la peor."""
        work = train_workload(cfg, folder_gb(Path(dataset).expanduser()))
        return rank(self.known_history(), work, self.provider.search(req, disk_for(work)))

    def submit(self, name: str, cfg: dict, dataset: str, req: Requirements, max_cost: float,
               offer_id: int | None = None, description: str = "") -> dict:
        path = Path(dataset).expanduser()
        if not path.is_dir():
            raise ValueError(f"No existe el dataset {path}")
        if max_cost <= 0:
            raise ValueError("max_cost debe ser mayor que 0")
        with self.lock:
            job = {"id": self.next_id, "name": name, "description": description, "kind": "train",
                   "status": "pending", "config": cfg, "dataset": str(dataset), "dataset_gb": round(folder_gb(path), 2),
                   "requirements": asdict(req), "max_cost": float(max_cost), "offer_id": offer_id,
                   "attempts": 0, "excluded_machines": [],
                   "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
            self.next_id += 1
            self.jobs.append(job)
            self.save()
            return job

    # ── ciclo ────────────────────────────────────────────────────────────────
    def tick(self) -> None:
        with self.lock:
            instances = None
            if any(j["status"] in ("starting", "running") for j in self.jobs) or not self._orphans_checked:
                instances = {i.id: i for i in self.provider.instances()}
            if not self._orphans_checked and instances is not None:
                self._destroy_orphans(instances)
                self._orphans_checked = True
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
        for instance in instances.values():
            if instance.label.startswith(self.label_prefix) and instance.id not in known:
                self.provider.destroy(instance.id)
                self.warnings.append(f"Eliminado el pod huérfano {instance.id} ({instance.label})")

    def _launch_ready(self) -> None:
        active = sum(j["status"] in ACTIVE for j in self.jobs)
        for job in self.jobs:
            if active >= self.settings.max_pods:
                return
            if job["status"] != "pending" or job["id"] in self.busy or job.get("retry_at", 0) > time.time():
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

    # ── preparar y lanzar ────────────────────────────────────────────────────
    def _run_key(self, job: dict) -> str:
        safe = re.sub(r"[^\w.+-]+", "_", job["name"])
        return f"runs/{job['id']:04d}_{job['attempts']}_{safe}"

    def _pick_offers(self, job: dict) -> list[Estimate]:
        req, work = Requirements(**job["requirements"]), self.workload(job)
        offers = [o for o in self.provider.search(req, disk_for(work))
                  if o.machine_id not in job["excluded_machines"]]
        ranked = rank(self.known_history(), work, offers)
        ranked.sort(key=lambda e: e.offer.id != job.get("offer_id"))     # la elegida, si sigue disponible
        return [e for e in ranked if e.total is None or e.total <= job["max_cost"]]

    def _spec(self, job: dict, estimate: Estimate, run_key: str) -> dict:
        cfg = dict(job["config"])
        model_dir = f"{cfg['output_root']}/{cfg['model_name']}"
        config_path = f"/tmp/{cfg['model_name']}.yaml"
        deadline_s = max(job["max_cost"] - estimate.transfer_cost, 0.0) / estimate.offer.dph * 3600
        return {"command": [self.python, "-u", self.train_script, "--config", config_path],
                "health": self.health,
                "files": {config_path: yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)},
                "inputs": [{"key": f"datasets/{Path(job['dataset']).name}", "path": cfg["dataset_root"]}],
                "outputs": [{"key": f"{run_key}/outputs/model", "path": model_dir}],
                "deadline_s": max(deadline_s, 0.0)}

    def _prepare(self, job: dict) -> None:
        with self.lock:
            job["attempts"] += 1
            job["run_key"] = run_key = self._run_key(job)
        name = Path(job["dataset"]).name
        job["phase"] = "uploading dataset"
        self.storage.upload(Path(job["dataset"]).expanduser(), f"datasets/{name}")

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
                instance = self.provider.create(candidate.offer, self.image, [run_key], env,
                                                disk_for(self.workload(job)), label)
            except RuntimeError as e:          # la oferta ya no está: la siguiente
                error = e
                print(f"[Cloud] oferta {candidate.offer.id} ({candidate.offer.cpu_name}): {e}", flush=True)
                continue
            with self.lock:
                job.update(status="starting", instance_id=instance, offer=asdict(candidate.offer),
                           estimate=_estimate_fields(candidate), started_ts=time.time(), seen_ts=time.time(),
                           seq=-1, phase="waiting for the pod", started=datetime.now().strftime("%Y-%m-%d %H:%M"))
                job.pop("wait", None)
                job.pop("retry_at", None)
                self.save()
            return
        raise RuntimeError(f"no se pudo crear el pod: {error}")

    def _wait(self, job: dict, reason: str) -> None:
        """Vuelve a la cola sin gastar un intento; se reintenta más tarde, no en cada ciclo."""
        with self.lock:
            job.update(status="pending", wait=reason, attempts=job["attempts"] - 1,
                       retry_at=time.time() + WAIT_RETRY_S)

    # ── seguimiento ──────────────────────────────────────────────────────────
    def _check(self, job: dict, instance) -> None:
        if "billing_ts" not in job and instance is not None and instance.status in ("running", "exited"):
            job["billing_ts"] = time.time()
        self._update_cost(job)
        result = self.storage.read(f"{job['run_key']}/result.json")
        if result:
            self._on_result(job, json.loads(result))
            return
        now = time.time()
        status = self.storage.read(f"{job['run_key']}/status.json")
        if status:
            status = json.loads(status)
            if status["seq"] != job.get("seq"):
                job.update(seq=status["seq"], seen_ts=now, phase=status["phase"],
                           progress={k: status[k] for k in ("done", "total", "rate") if k in status})
                job["status"] = "running"
                job.setdefault("billing_ts", now)
                if status["phase"] == "run" and "run_ts" not in job:
                    job["run_ts"] = now
            if status.get("boot", 1) > MAX_BOOTS + 1:     # el runner debió cortarlo: no seguir pagando
                self._retry_or_fail(job, "el pod se reinicia en bucle")
                self.save()
                return
        if job.get("cost", 0.0) > job["max_cost"]:
            self.provider.destroy(job["instance_id"])
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
        if job.get("instance_id") is not None:
            self.provider.destroy(job["instance_id"])
            self._update_cost(job)
            job["instance_id"] = None

    def _retry_or_fail(self, job: dict, reason: str) -> None:
        machine = (job.get("offer") or {}).get("machine_id")
        self._release(job)
        job["cost_before"] = job.get("cost") or 0.0
        if machine is not None:
            job["excluded_machines"].append(machine)
        if job["attempts"] >= MAX_ATTEMPTS:
            self._fail(job, f"{reason} ({job['attempts']} intentos)")
            return
        job.update(status="pending", wait=f"reintento: {reason}")
        for key in ("offer", "estimate", "phase", "progress", "run_ts", "started_ts", "billing_ts", "result"):
            job.pop(key, None)
        self.save()

    def _fail(self, job: dict, reason: str) -> None:
        with self.lock:
            job.update(status="failed", error=reason, finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.save()

    def _update_cost(self, job: dict) -> None:
        """Horas corriendo por su precio más la transferencia prevista: la descarga de la imagen no se
        cobra por hora. El valor exacto sale en la factura."""
        if "offer" not in job:
            return
        hours = (time.time() - job["billing_ts"]) / 3600 if "billing_ts" in job else 0.0
        job["cost"] = round(job.get("cost_before", 0.0) + hours * job["offer"]["dph"]
                            + job["estimate"]["transfer_cost"], 4)

    # ── descargar ────────────────────────────────────────────────────────────
    def _finish(self, job: dict) -> None:
        """Se puede repetir: el bucket sigue teniendo las salidas si el lab falla o se llena."""
        job["phase"] = "downloading"
        self._release(job)
        self.save()
        cfg, key = job["config"], f"{job['run_key']}/outputs/model"
        target = Path(cfg["output_root"]).expanduser() / cfg["model_name"]
        need = self.workload(job).up_gb
        free = shutil.disk_usage(Path.home()).free / 1e9
        if free < need + 5:
            raise RuntimeError(f"solo hay {free:.0f} GB libres en el lab; el modelo sigue en el bucket: {key}")
        self.storage.download(key, target)
        bad = [name for name, digest in job["result"]["outputs"][key].items() if sha256(target / name) != digest]
        if bad:
            raise RuntimeError(f"sha256 distinto en {bad[:3]}; el modelo sigue en el bucket: {key}")
        with self.lock:
            self._record_history(job)
            job.update(status="done", phase="done", model_dir=str(target),
                       finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.save()
        if self.on_model_ready:
            self.on_model_ready(job)

    def _record_history(self, job: dict) -> None:
        rate = job["result"].get("rate")
        if not rate:
            return
        self.history.append({"kind": job["kind"], "profile": self.workload(job).profile,
                             "gpu": job["offer"]["gpu_name"],
                             "cpu": job["offer"]["cpu_name"], "rate": rate, "job": job["id"],
                             "seconds": job["result"].get("seconds"), "dph": job["offer"]["dph"],
                             "cost": job.get("cost"), "estimated_total": job["estimate"].get("total"),
                             "date": datetime.now().strftime("%Y-%m-%d")})
        (self.qdir / "history.yaml").write_text(yaml.safe_dump(self.history, sort_keys=False, allow_unicode=True))

    # ── acciones del panel ───────────────────────────────────────────────────
    def stop(self, job_id: int) -> None:
        with self.lock:
            job = self.get(job_id)
            if job and job["status"] in ACTIVE and job["id"] not in self.busy:
                self._release(job)
                job.update(status="stopped", finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
                self.save()
            elif job and job["status"] == "pending":
                job["status"] = "stopped"
                self.save()

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
                for key in ("offer", "estimate", "phase", "progress", "run_ts", "started_ts", "billing_ts", "result"):
                    job.pop(key, None)
            self.save()

    def remove(self, job_id: int) -> None:
        with self.lock:
            job = self.get(job_id)
            if job and job["status"] not in ACTIVE:
                self.jobs.remove(job)
                self.save()


def _estimate_fields(e: Estimate) -> dict:
    return {"rate": e.rate, "confidence": e.confidence, "source": e.source, "hours": e.hours,
            "compute_cost": e.compute_cost, "transfer_cost": e.transfer_cost, "total": e.total}
