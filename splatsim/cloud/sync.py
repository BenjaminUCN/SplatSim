"""Una sola cola para varios PCs: la copia de verdad vive en R2 y solo un panel la avanza a la vez.

Dos paneles avanzando la misma cola lanzarían dos pods por trabajo y se borrarían los huérfanos uno al
otro, así que el que tiene el control deja en R2 un permiso que renueva en cada ciclo y que vence solo
si el panel muere. Los demás muestran la cola sin escribirla. Para pasar el control, el otro panel lo
pide; el que lo tiene termina sus subidas y descargas en curso, sube la cola y lo suelta.

La cola en R2 lleva las rutas con ~ (cada PC tiene su home) y sin los sha256 de los trabajos ya
terminados, que solo sirven para verificar una descarga y son casi todo el peso de jobs.yaml.
"""
import copy
import functools
import hashlib
import os
import socket
import threading
import time
from dataclasses import asdict

import yaml

from splatsim.utils.paths import contract_tree, expand_tree

LEASE_KEY = "queue/lease.yaml"
STATE_KEY = "queue/state.yaml"
HANDOVER_KEY = "queue/handover.yaml"
LEASE_TTL_S = 90                # sin renovarse en este tiempo, el panel que lo tenía se da por muerto
SETTLE_S = 3.0                  # tras escribir el permiso, se relee: si dos lo tomaron a la vez, gana uno
FILES = ("jobs.yaml", "history.yaml", "machines.yaml")

CONTROL, VIEW, RELEASING = "control", "view", "releasing"


def panel_name() -> str:
    return os.environ.get("SPLATSIM_PANEL_NAME") or socket.gethostname()


def _locked(method):
    """El botón del panel y el hilo del orquestador no pueden tomar o soltar el control a la vez."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


def _slim(job: dict) -> dict:
    """Un trabajo terminado no vuelve a verificar su descarga: sin los sha256, salvo si sigue en el bucket."""
    if job["status"] != "done" or job.get("remote_output") or "outputs" not in (job.get("result") or {}):
        return job
    result = {k: v for k, v in job["result"].items() if k != "outputs"}
    return {**job, "result": result}


class QueueSync:
    def __init__(self, orch, name: str | None = None, ttl_s: float = LEASE_TTL_S, settle_s: float = SETTLE_S):
        self.orch, self.storage = orch, orch.storage
        self.name = name or panel_name()
        self.ttl_s, self.settle_s = ttl_s, settle_s
        self.role = VIEW
        self.lease: dict = {}
        self.wants_control = False      # este panel pidió el control y espera que el otro lo suelte
        self.handover_to: str | None = None
        self.message = ""
        self.lock = threading.RLock()
        path = orch.qdir / "sync.yaml"
        local = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
        self.rev, self.hash = local.get("rev", 0), local.get("hash")

    # ── lo que ve el panel ───────────────────────────────────────────────────
    @property
    def in_control(self) -> bool:
        return self.role == CONTROL

    def holder(self) -> str | None:
        return self.lease.get("holder") if self._alive(self.lease) else None

    def status(self) -> str:
        if self.role == CONTROL:
            return f"🟢 **{self.name}** controls the queue"
        if self.role == RELEASING:
            busy = len(self.orch.busy)
            to = f" to **{self.handover_to}**" if self.handover_to else ""
            return f"⏳ Handing over{to}" + (f": waiting for {busy} transfer(s) to finish" if busy else "")
        holder = self.holder()
        text = f"👀 Read only: **{holder}** controls the queue" if holder else "⚪ Nobody controls the queue"
        if self.wants_control:
            text += " · control requested, waiting for it to be released"
        return text + (f" · {self.message}" if self.message else "")

    # ── ciclo (hilo del orquestador) ─────────────────────────────────────────
    @_locked
    def start(self) -> None:
        """Al abrir el panel: toma el control si está libre o era suyo (el mismo PC, reabierto)."""
        self.lease = self._read(LEASE_KEY)
        if not self._alive(self.lease) or self.lease.get("holder") == self.name or self._first_with_jobs():
            self.take()
        else:
            self._view()

    def _first_with_jobs(self) -> bool:
        """La primera vez, la cola que ya tiene trabajos manda sobre una vacía: si el otro PC abrió antes, solo
        subió una cola sin nada, y adoptarla dejaría sin seguimiento los pods de este."""
        return not self.rev and bool(self.orch.jobs) and not self._read(STATE_KEY).get("jobs")

    @_locked
    def cycle(self) -> None:
        """Lo llama el hilo del orquestador cada TICK_PERIOD, en lugar de orch.tick()."""
        if self.role == VIEW:
            self._view()
            return
        self.lease = self._read(LEASE_KEY)
        if self.lease.get("holder") != self.name:      # su permiso venció (sin red) y otro lo tomó
            self.message = "control was taken by another panel"
            self._view()
            return
        if self.role == CONTROL:
            request = self._read(HANDOVER_KEY)
            if request.get("to") and request["to"] != self.name and request.get("ts", 0) > time.time() - 2 * self.ttl_s:
                self.release(to=request["to"])
        if self.role == CONTROL:
            self.orch.tick()
            self.push()
            self._write_lease(self.name)
        elif not self.orch.busy:                         # RELEASING: lo que estaba subiendo o bajando ya terminó
            self.push()
            self._write_lease(None)
            self._view()
        else:
            self._write_lease(self.name)                 # sigue entregando: que no lo den por muerto

    @_locked
    def request_control(self) -> None:
        """Botón del panel: si nadie lo tiene, lo toma ya; si no, lo pide y espera a que lo suelten."""
        self.lease = self._read(LEASE_KEY)
        if self.role == CONTROL:
            return
        if not self._alive(self.lease) or self.lease.get("holder") == self.name:
            self.take()
            return
        self.wants_control = True
        self.storage.write(HANDOVER_KEY, yaml.safe_dump({"to": self.name, "ts": time.time()}))

    @_locked
    def release(self, to: str | None = None) -> None:
        """Deja de lanzar trabajos; cuando terminan sus transferencias, sube la cola y suelta el permiso."""
        if self.role == CONTROL:
            self.role, self.handover_to = RELEASING, to

    @_locked
    def shutdown(self) -> None:
        """Al cerrar el panel: lo que estaba transfiriendo lo retoma el siguiente (preparing vuelve a pending
        y finishing se repite), así que basta con subir la cola y soltar el permiso."""
        if self.role in (CONTROL, RELEASING):
            self.push()
            self._write_lease(None)
            self.role = VIEW

    def take(self) -> None:
        self._write_lease(self.name)
        if self.settle_s:
            time.sleep(self.settle_s)
        self.lease = self._read(LEASE_KEY)
        if self.lease.get("holder") != self.name:
            self.message = "another panel took control at the same time"
            self._view()
            return
        state = self._read(STATE_KEY)
        if state.get("rev", 0) > self.rev and not (not self.rev and self.orch.jobs and not state.get("jobs")):
            self._apply(state, resume=True)
        else:
            self.orch.load(resume=True)
            if state.get("rev", 0) > self.rev:
                self.rev = state["rev"]          # la próxima subida reemplaza a la cola vacía
        self.orch.read_only = False
        self.role, self.wants_control, self.message = CONTROL, False, ""
        if self._read(HANDOVER_KEY).get("to") == self.name:
            self.storage.write(HANDOVER_KEY, "{}")
        self.push()
        self._write_lease(self.name)

    # ── R2 ───────────────────────────────────────────────────────────────────
    def push(self) -> None:
        """Sube la cola si cambió desde la última vez; cada subida es una revisión nueva."""
        bundle = self._bundle()
        text = yaml.safe_dump(bundle, sort_keys=False, allow_unicode=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest == self.hash and self.rev:
            return
        rev = self.rev + 1
        self.storage.write(STATE_KEY, yaml.safe_dump({"rev": rev, "by": self.name, "saved": time.time(), **bundle},
                                                     sort_keys=False, allow_unicode=True))
        self._remember(rev, digest)

    def _view(self) -> None:
        self.role = VIEW
        self.orch.read_only = True
        self.lease = self._read(LEASE_KEY)
        if self.wants_control and not self._alive(self.lease):
            self.take()
            return
        if self.wants_control:       # renueva el pedido: uno viejo, de un panel que se cerró, no cuenta
            self.storage.write(HANDOVER_KEY, yaml.safe_dump({"to": self.name, "ts": time.time()}))
        if self.lease.get("rev", 0) != self.rev:
            state = self._read(STATE_KEY)
            if state:
                self._apply(state, resume=False)

    def _apply(self, state: dict, resume: bool) -> None:
        """Escribe la cola de R2 en este PC y la carga. Lo que este PC cambió sin subir queda en *.bak."""
        if self.orch.jobs and self._bundle_hash() != self.hash:
            stamp = time.strftime("%y%m%d%H%M%S")
            for name in FILES:
                if (self.orch.qdir / name).exists():
                    os.replace(self.orch.qdir / name, self.orch.qdir / f"{name}.bak{stamp}")
            self.orch.warnings.append(f"La cola de este PC tenía cambios sin subir: quedó en {self.orch.qdir}/*.bak{stamp}")
        data = expand_tree({k: state[k] for k in ("settings", "jobs", "history", "machines") if k in state})
        files = {"jobs.yaml": {"settings": data.get("settings") or {}, "jobs": data.get("jobs") or []},
                 "history.yaml": data.get("history") or [], "machines.yaml": data.get("machines") or {}}
        for name, content in files.items():
            tmp = self.orch.qdir / f"{name}.sync"
            tmp.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
            os.replace(tmp, self.orch.qdir / name)
        self.orch.load(resume=resume)
        self._remember(state.get("rev", 0), self._bundle_hash())

    def _bundle(self) -> dict:
        orch = self.orch
        with orch.lock:              # copia: los hilos de subida y descarga siguen escribiendo en los trabajos
            bundle = copy.deepcopy({"settings": asdict(orch.settings), "jobs": [_slim(j) for j in orch.jobs],
                                    "history": orch.history, "machines": orch.machines})
        return contract_tree(bundle)

    def _bundle_hash(self) -> str:
        return hashlib.sha256(yaml.safe_dump(self._bundle(), sort_keys=False, allow_unicode=True).encode()).hexdigest()

    def _remember(self, rev: int, digest: str) -> None:
        self.rev, self.hash = rev, digest
        (self.orch.qdir / "sync.yaml").write_text(yaml.safe_dump({"rev": rev, "hash": digest}))

    def _write_lease(self, holder: str | None) -> None:
        self.lease = {"holder": holder, "expires": time.time() + self.ttl_s if holder else 0, "rev": self.rev,
                      "renewed": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.storage.write(LEASE_KEY, yaml.safe_dump(self.lease))

    def _read(self, key: str) -> dict:
        text = self.storage.read(key)
        return (yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) if text else None) or {}

    @staticmethod
    def _alive(lease: dict) -> bool:
        return bool(lease.get("holder")) and lease.get("expires", 0) > time.time()
