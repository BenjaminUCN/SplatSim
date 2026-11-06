"""Punto de entrada de la imagen: corre una etapa y deja sus resultados en el bucket.

    python -m splatsim.cloud.runner runs/0007_1_mi-modelo

Lee <run>/spec.json, comprueba el host, baja las entradas, corre el comando, sube las salidas y
escribe <run>/result.json, siempre lo último: si existe, las salidas ya están completas. Cada
HEARTBEAT_S actualiza <run>/status.json y <run>/run.log, que es todo lo que necesita el panel.

Vast reinicia el contenedor cuando termina, así que salir no deja de cobrar: al final el pod detiene
su propia instancia con la API de Vast. <run>/boot.json guarda el primer arranque y cuántos van, para
que el plazo (spec["deadline_s"]) no vuelva a cero en cada reinicio y un bucle de reinicios se corte.
Con eso el gasto queda acotado aunque el PC del panel esté apagado.

Solo usa la biblioteca estándar y rclone.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

from splatsim.cloud.storage import Storage, item_label

HEARTBEAT_S = float(os.environ.get("SPLATSIM_HEARTBEAT_S", 30))
HEALTH_TIMEOUT_S = 180
STOP_TIMEOUT_S = 30
MAX_BOOTS = 3
WARMUP_S = float(os.environ.get("SPLATSIM_WARMUP_S", 180))   # torch.compile: no es la velocidad de crucero
RESULT_ATTEMPTS = 20          # result.json es lo único imprescindible: unos minutos de reintentos
RESULT_RETRY_S = float(os.environ.get("SPLATSIM_RESULT_RETRY_S", 15))
STEP = re.compile(r"step: (\d+)\b")
# la barra de lerobot_conversion.py (no las de LeRobot) y el [n/total] de render y mirror
EPISODES = re.compile(r"(?:episodes:\s+\d+%.*?|\[)(\d+)/(\d+)")


def expand(path: str) -> Path:
    return Path(path).expanduser()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(folder: Path, progress=None) -> dict[str, str]:
    """sha256 de cada archivo: el panel lo compara con lo que descarga."""
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    digests = {}
    for i, path in enumerate(files, 1):
        digests[str(path.relative_to(folder))] = sha256(path)
        if progress:
            progress(i, len(files))
    return digests


class Run:
    def __init__(self, storage: Storage, key: str):
        self.storage, self.key = storage, key
        self.t0 = time.monotonic()
        self.first_boot = time.time()
        self.status = {"phase": "boot", "seq": 0}
        self.seconds: dict[str, float] = {}
        self.log_path = Path(tempfile.mkstemp(suffix=".log")[1])
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.timed_out = False
        self.first_step: tuple[int, float] | None = None
        self.started_steps = 0.0
        self.last_step: tuple[int, float] | None = None

    def publish(self, **fields) -> None:
        with self.lock:
            self.status.update(fields, seq=self.status["seq"] + 1)
            text = json.dumps(self.status)
        try:
            self.storage.write(f"{self.key}/status.json", text)
        except Exception as e:  # noqa: BLE001  informar el avance no debe matar el trabajo
            print(f"[runner] no se pudo publicar el estado: {e}", flush=True)

    def heartbeat(self) -> None:
        """Un corte de red no debe matar el trabajo: el siguiente latido lo vuelve a intentar."""
        while not self.finished.wait(HEARTBEAT_S):
            self.publish()
            try:
                if self.log_path.exists():
                    self.storage.upload_file(self.log_path, f"{self.key}/run.log")
            except Exception as e:  # noqa: BLE001
                print(f"[runner] latido fallido: {e}", flush=True)

    def phase(self, name: str) -> None:
        print(f"[runner] {name} ({time.monotonic() - self.t0:.0f} s)", flush=True)
        with self.lock:                     # el avance de la fase anterior ya no dice nada
            for field in ("done", "total", "rate", "detail"):
                self.status.pop(field, None)
        self.publish(phase=name)

    def detail(self, text: str) -> None:
        """Lo publica el siguiente latido: una escritura por archivo costaría más que el archivo."""
        with self.lock:
            self.status["detail"] = text

    def timed(self, name: str, action) -> None:
        self.phase(name)
        start = time.monotonic()
        action()
        self.seconds[name] = round(time.monotonic() - start, 1)

    def rate(self) -> float | None:
        """Unidades de trabajo por segundo (pasos o frames) durante la ejecución."""
        if not self.first_step or not self.last_step or self.last_step[1] <= self.first_step[1]:
            return None
        return round((self.last_step[0] - self.first_step[0]) / (self.last_step[1] - self.first_step[1]), 3)

    def _track(self, line: str) -> None:
        now = time.monotonic()
        if m := STEP.match(line):
            done, total = int(m.group(1)), None
        elif m := EPISODES.match(line):
            done, total = int(m.group(1)), int(m.group(2))
        else:
            return
        if self.first_step is None:
            self.started_steps = now
        if self.first_step is None or now - self.started_steps < WARMUP_S:
            self.first_step = (done, now)
        self.last_step = (done, now)
        with self.lock:
            self.status.update(done=done, rate=self.rate(), **({"total": total} if total else {}))

    def execute(self, command: list[str], deadline_at: float) -> int:
        """Corre el comando con la salida al log y lo termina si se acaba el tiempo."""
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     errors="replace", env=env, start_new_session=True)

        def watchdog():
            while self.proc.poll() is None:
                if time.time() > deadline_at:
                    self.timed_out = True
                    self.terminate()
                    return
                time.sleep(1)

        threading.Thread(target=watchdog, daemon=True).start()
        with open(self.log_path, "a") as log:
            for line in self.proc.stdout:
                log.write(line)
                log.flush()
                self._track(line)
        self.proc.stdout.close()
        return self.proc.wait()

    def terminate(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)

    def finish(self, **result) -> None:
        self.finished.set()
        try:
            if self.log_path.exists():
                self.storage.upload_file(self.log_path, f"{self.key}/run.log")
        except Exception as e:  # noqa: BLE001
            print(f"[runner] no se pudo subir el log: {e}", flush=True)
        self.publish(phase="finished")
        text = json.dumps({**result, "rate": self.rate(), "seconds": self.seconds,
                           "total_s": round(time.monotonic() - self.t0, 1)})
        for attempt in range(RESULT_ATTEMPTS):
            try:
                self.storage.write(f"{self.key}/result.json", text)
                return
            except Exception as e:  # noqa: BLE001
                print(f"[runner] result.json, intento {attempt + 1}: {e}", flush=True)
                time.sleep(RESULT_RETRY_S)
        raise RuntimeError("no se pudo escribir result.json")


def host_is_healthy(command: list[str] | None) -> str | None:
    """None si el host sirve; si no, el motivo. Un host defectuoso se descarta sin gastar más."""
    if not command:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=HEALTH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "la comprobación del host no terminó"
    return None if done.returncode == 0 else (done.stdout + done.stderr).strip()[-500:]


def stop_own_instance() -> str:
    """Detiene esta instancia de Vast (CONTAINER_API_KEY solo sirve para ella). Devuelve qué pasó."""
    key, instance = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not key or not instance:
        return "sin CONTAINER_API_KEY o CONTAINER_ID"
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance}/", data=json.dumps({"state": "stopped"}).encode(),
        method="PUT", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return f"http {response.status}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def run_stage(run: Run, spec: dict) -> None:
    deadline_at = run.first_boot + spec["deadline_s"]
    if time.time() > deadline_at:
        run.finish(ok=False, reason="deadline")
        return
    run.publish(phase="health")
    if problem := host_is_healthy(spec.get("health")):
        run.finish(ok=False, reason="host_bad", error=problem)
        return
    for path, text in spec.get("files", {}).items():
        expand(path).parent.mkdir(parents=True, exist_ok=True)
        expand(path).write_text(text)

    def download():
        inputs = spec.get("inputs", [])
        for i, item in enumerate(inputs, 1):
            key = item["key"]
            run.detail(item_label(i, len(inputs), key))
            run.storage.download(key, expand(item["path"]), progress=lambda done, total, i=i, key=key: run.detail(
                item_label(i, len(inputs), key, done, total)))

    run.timed("download", download)
    for path in spec.get("clean", []):
        shutil.rmtree(expand(path), ignore_errors=True)
    run.phase("run")
    start = time.monotonic()
    code = run.execute(spec["command"], deadline_at)
    run.seconds["run"] = round(time.monotonic() - start, 1)
    if run.timed_out:
        run.finish(ok=False, reason="deadline", exit_code=code)
        return
    if code != 0:
        run.finish(ok=False, reason="command_failed", exit_code=code)
        return

    outputs: dict[str, dict[str, str]] = {}

    def upload():
        items = spec.get("outputs", [])
        for i, item in enumerate(items, 1):
            folder, key = expand(item["path"]), item["key"]
            outputs[key] = manifest(folder, lambda done, total: run.detail(f"hashing {done}/{total} files"))
            run.detail(item_label(i, len(items), key))
            run.storage.upload(folder, key, packed=item.get("packed", False),
                               progress=lambda done, total, i=i, key=key: run.detail(
                                   item_label(i, len(items), key, done, total)))

    run.timed("upload", upload)
    run.finish(ok=True, reason="done", exit_code=0, outputs=outputs)


def main(key: str) -> int:
    storage = Storage(os.environ["SPLATSIM_REMOTE"])
    run = Run(storage, key)
    code = run_once(run, storage, key)
    outcome = stop_own_instance()
    print(f"[runner] detener la instancia: {outcome}", flush=True)
    try:
        storage.write(f"{key}/stop.txt", outcome)
    except Exception as e:  # noqa: BLE001
        print(f"[runner] {e}", flush=True)
    return code


def run_once(run: Run, storage: Storage, key: str) -> int:
    if storage.read(f"{key}/result.json"):       # reinicio de un trabajo ya terminado: solo detenerse
        return 0
    boot = json.loads(storage.read(f"{key}/boot.json") or "{}")
    boot = {"first": boot.get("first", run.first_boot), "boots": boot.get("boots", 0) + 1}
    storage.write(f"{key}/boot.json", json.dumps(boot))
    run.first_boot = boot["first"]
    run.status["boot"] = boot["boots"]
    signal.signal(signal.SIGTERM, lambda *_: (run.terminate(), sys.exit(143)))
    threading.Thread(target=run.heartbeat, daemon=True).start()
    if boot["boots"] > MAX_BOOTS:
        run.finish(ok=False, reason="restart_loop", error=f"{boot['boots']} arranques sin terminar")
        return 1
    try:
        run_stage(run, json.loads(storage.read(f"{key}/spec.json")))
    except Exception:  # noqa: BLE001  el panel debe enterarse de cualquier fallo
        traceback.print_exc()
        run.finish(ok=False, reason="runner_error", error=traceback.format_exc()[-1500:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
