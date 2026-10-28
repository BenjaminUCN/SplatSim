"""Entrena una cola de modelos uno tras otro (p.ej. barrer horizon, n_obs_steps o cámaras).

    python scripts/xarm5/train_queue.py configs/train/queue_example.yaml --dry_run   # muestra las corridas
    python scripts/xarm5/train_queue.py configs/train/queue_example.yaml             # entrena
    python scripts/xarm5/train_queue.py configs/train/queue_example.yaml --only 2cam-h32

Archivo de cola:
    base: configs/train/diffusion.yaml   # config de partida
    common: {push_to_hub: false}         # se aplica a todas las corridas
    runs:                                # corridas explícitas (name obligatorio)
      - {name: 2cam-h64, description: base, arrangement: 2cam}
    sweep:                               # opcional: producto cartesiano de estos valores
      horizon: [32, 64]
      arrangement: [2cam, 3cam]

Cada corrida combina base + common + sus campos; las del sweep se llaman
<valor>-<valor>... Por corrida se guardan en ~/data/train_queues/<cola>/: la config final
(configs/<name>.yaml), el log (logs/<name>.log) y el estado en state.yaml. Al relanzar, las
corridas terminadas se saltan (--rerun para repetirlas). Al final se regenera
configs/registry.yaml para que los modelos aparezcan en el panel con su name/description.
"""
import itertools
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import tyro
import yaml

ROOT = Path(__file__).resolve().parents[2]       # raíz del repo
SCRIPTS = Path(__file__).resolve().parent      # scripts/xarm5
PRINT_EVERY = 1000


@dataclass
class Args:
    queue: tyro.conf.Positional[str]    # archivo YAML de la cola
    out: str = "~/data/train_queues"    # carpeta base de configs, logs y estado de cada cola
    dry_run: bool = False               # solo muestra las corridas
    only: tuple[str, ...] = ()          # nombres de corridas a ejecutar (por defecto todas)
    rerun: bool = False                 # repetir también las terminadas
    stop_on_error: bool = False         # por defecto una corrida fallida no detiene la cola
    update_registry: bool = True


def _train_fields() -> set[str]:
    sys.path.insert(0, str(SCRIPTS))
    from train_diffusion import TrainConfig
    return {f.name for f in fields(TrainConfig)}


# prefijos cortos para nombrar las corridas del sweep (h32-a16-2cam)
_ABBR = {"horizon": "h", "n_action_steps": "a", "n_obs_steps": "o", "training_steps": "steps",
         "batch_size": "bs"}


def _sweep_name(combo: dict) -> str:
    return "-".join(f"{_ABBR.get(k, k)}{v}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else str(v) for k, v in combo.items())


def build_runs(queue: dict) -> list[dict]:
    """Lista de corridas {name, description, ...overrides} (sin base ni common)."""
    runs = [dict(r) for r in queue.get("runs") or []]
    sweep = queue.get("sweep") or {}
    if sweep:
        keys = list(sweep)
        for values in itertools.product(*(sweep[k] for k in keys)):
            combo = dict(zip(keys, values))
            runs.append({"name": _sweep_name(combo),
                         "description": ", ".join(f"{k}={v}" for k, v in combo.items()), **combo})
    names = [r.get("name") for r in runs]
    if None in names or "" in names:
        raise SystemExit("Cada corrida en runs necesita name")
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"Nombres de corrida repetidos: {sorted(dup)}")
    return runs


def run_config(queue: dict, run: dict, stamp: str) -> dict:
    with open(ROOT / queue.get("base", "configs/train/diffusion.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(queue.get("common") or {})
    cfg.update(run)
    # carpeta del modelo: <name>_<MMDD_HHMM> (el registro lee la fecha del sufijo)
    safe = re.sub(r"[^\w.+-]+", "_", run["name"])
    cfg["model_name"] = f"{safe}_{stamp}"
    return cfg


def _load_state(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return {}


def main(args: Args) -> None:
    queue_path = Path(args.queue)
    queue = yaml.safe_load(queue_path.read_text()) or {}
    runs = build_runs(queue)
    if args.only:
        missing = set(args.only) - {r["name"] for r in runs}
        if missing:
            raise SystemExit(f"--only: corridas desconocidas {sorted(missing)}")
        runs = [r for r in runs if r["name"] in args.only]

    allowed = _train_fields()
    for r in runs:
        unknown = set(r) | set(queue.get("common") or {})
        unknown -= allowed
        if unknown:
            raise SystemExit(f"Corrida {r['name']}: campos desconocidos {sorted(unknown)}")

    qdir = Path(os.path.expanduser(args.out)) / queue_path.stem
    state_path = qdir / "state.yaml"
    state = _load_state(state_path)

    print(f"[Queue] {queue_path} → {qdir}")
    for r in runs:
        status = state.get(r["name"], {}).get("status", "pending")
        overrides = {k: v for k, v in r.items() if k not in ("name", "description")}
        print(f"  {r['name']:24s} {status:8s} {overrides}  {r.get('description', '')}")
    if args.dry_run:
        return

    (qdir / "configs").mkdir(parents=True, exist_ok=True)
    (qdir / "logs").mkdir(parents=True, exist_ok=True)

    def save_state():
        state_path.write_text(yaml.safe_dump(state, sort_keys=False))

    for i, r in enumerate(runs, 1):
        name = r["name"]
        if state.get(name, {}).get("status") == "done" and not args.rerun:
            print(f"\n[Queue] {i}/{len(runs)} {name}: ya terminada, se salta")
            continue
        cfg = run_config(queue, r, datetime.now().strftime("%m%d_%H%M"))
        cfg_path = qdir / "configs" / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        log_path = qdir / "logs" / f"{name}.log"
        print(f"\n[Queue] {i}/{len(runs)} {name} → {cfg['model_name']} (log: {log_path})")

        state[name] = {"status": "running", "model_name": cfg["model_name"],
                       "started": datetime.now().strftime("%Y-%m-%d %H:%M"), "config": str(cfg_path)}
        save_state()
        t0 = time.monotonic()
        with open(log_path, "w") as log:
            # la salida va al log y a la terminal; sin buffer para ver el avance al momento
            proc = subprocess.Popen(
                [sys.executable, str(SCRIPTS / "train_diffusion.py"), "--config", str(cfg_path)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            try:
                for line in proc.stdout:
                    log.write(line)
                    # en la terminal, solo uno de cada PRINT_EVERY pasos (el log los tiene todos)
                    m = re.match(r"step: (\d+) ", line)
                    if m is None or int(m.group(1)) % PRINT_EVERY == 0:
                        sys.stdout.write(f"  {line}")
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                state[name]["status"] = "stopped"
                save_state()
                print("\n[Queue] Detenida. Relanza el mismo comando para continuar.")
                return
        minutes = (time.monotonic() - t0) / 60
        state[name].update(status="done" if code == 0 else "failed", minutes=round(minutes, 1),
                           finished=datetime.now().strftime("%Y-%m-%d %H:%M"))
        save_state()
        print(f"[Queue] {name}: {state[name]['status']} en {minutes:.1f} min")
        if code != 0 and args.stop_on_error:
            break

    done = [n for n, s in state.items() if s.get("status") == "done"]
    failed = [n for n, s in state.items() if s.get("status") == "failed"]
    print(f"\n[Queue] terminadas: {len(done)}  fallidas: {failed or 'ninguna'}")
    if args.update_registry:
        print("[Queue] Actualizando configs/registry.yaml")
        subprocess.run([sys.executable, str(SCRIPTS / "build_registry.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main(tyro.cli(Args))
