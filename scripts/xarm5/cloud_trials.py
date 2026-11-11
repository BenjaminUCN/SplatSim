"""Evaluación de una política en un pod (etapa "trials" de splatsim/cloud): levanta el sim sin
ventana, corre run_policy_trials.py contra él y deja en --out solo lo que se baja al lab.

    python scripts/xarm5/cloud_trials.py --model ~/models/leaves_2cam --dataset_root ~/datasets/x \
        --run_name leaves_2cam --out ~/trials_out --arrangement 2cam --hz 10 \
        --grid_path configs/eval_grid_4x4.yaml --occlusion_targets 0 25 50 75

--model: carpeta del modelo (model.safetensors, config.json, processors y train_config.yaml). Con
--step, un checkpoint intermedio (checkpoints/<step>/{policy,preprocessor,postprocessor}): se junta
en una carpeta con el train_config.yaml del modelo, como espera el agente.
--dataset_root: el dataset con que se entrenó; el agente solo lee su meta/.
Las imágenes de los ensayos se borran salvo las de --keep_images_pos (pesan ~125 MB por ensayo).
"""
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[2]
SERVER_TIMEOUT_S = 900          # cargar la escena, el robot y compilar los kernels del rasterizador


@dataclass
class Args:
    model: str
    dataset_root: str
    run_name: str
    out: str = "~/trials_out"
    step: str | None = None                      # checkpoints/<step> en vez del modelo final
    robot_name: str = "xarm5_lite_plant_v3"
    arrangement: str = "2cam"
    hz: int = 10
    control_mode: str = "velocity"
    grid_path: str = "configs/eval_grid_4x4.yaml"
    occlusion_targets: tuple[float, ...] | None = None
    leaf_dir: str = "~/data/occluders/xarm5_lite_plant_v3"
    leaf_seed: int = 0
    trials_per_position: int = 1
    trial_seconds: float = 15.0
    positions: tuple[int, ...] | None = None
    ik_orientation: str = "off"
    stop_on_success: bool = True
    keep_images_pos: tuple[int, ...] = (5,)
    robot_port: int = 6021
    panel_port: int = 8094


def flatten_step(model: Path, step: str) -> Path:
    """checkpoints/<step>/{policy,preprocessor,postprocessor} -> una carpeta como la del modelo final."""
    out = model.parent / f"{model.name}__{step}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for part in ("policy", "preprocessor", "postprocessor"):
        for f in (model / "checkpoints" / step / part).iterdir():
            shutil.copy2(f, out / f.name)
    for name in ("train_config.yaml", "relative_action_stats.json"):
        if (model / name).is_file():
            shutil.copy2(model / name, out / name)
    return out


def wait_for_port(port: int, server: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"el servidor del sim terminó (código {server.returncode}) antes de abrir el puerto")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise SystemExit(f"el servidor del sim no abrió el puerto {port} en {timeout:.0f} s")


def prune_images(run_dir: Path, keep: tuple[int, ...]) -> None:
    """Deja los .pkl y meta.json de cada ensayo; las imágenes solo de las posiciones en keep."""
    import csv
    with open(run_dir / "trials.csv") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["pos_idx"]) in keep:
            continue
        folder = run_dir / Path(row["folder"]).name
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.suffix.lower() in (".png", ".jpg", ".jpeg", ".mp4"):
                item.unlink()


def main(a: Args):
    model = Path(a.model).expanduser()
    checkpoint = flatten_step(model, a.step) if a.step else model
    out = Path(a.out).expanduser()
    shutil.rmtree(out / a.run_name, ignore_errors=True)     # un reinicio del pod empieza de cero
    env = {**os.environ, "SPLATSIM_HEADLESS": "1"}
    server_cmd = [sys.executable, "-u", "scripts/launch_nodes.py", "--robot", "sim_xarm5",
                  "--robot_name", a.robot_name, "--arrangement", a.arrangement, "--hz", str(a.hz),
                  "--control_mode", a.control_mode, "--gui", "none",
                  "--robot_port", str(a.robot_port), "--panel_port", str(a.panel_port)]
    print("[cloud_trials] servidor:", " ".join(server_cmd), flush=True)
    server = subprocess.Popen(server_cmd, cwd=ROOT, env=env)
    try:
        wait_for_port(a.robot_port, server, SERVER_TIMEOUT_S)
        print("[cloud_trials] servidor listo", flush=True)
        trials_cmd = [sys.executable, "-u", "scripts/xarm5/run_policy_trials.py",
                      "--checkpoint", str(checkpoint), "--dataset_root", str(Path(a.dataset_root).expanduser()),
                      "--run_name", a.run_name, "--output_dir", str(out), "--robot_port", str(a.robot_port),
                      "--grid_path", a.grid_path, "--trials_per_position", str(a.trials_per_position),
                      "--trial_seconds", str(a.trial_seconds), "--ik_orientation", a.ik_orientation,
                      "--leaf_dir", a.leaf_dir, "--leaf_seed", str(a.leaf_seed),
                      "--stop-on-success" if a.stop_on_success else "--no-stop-on-success"]
        if a.occlusion_targets:
            trials_cmd += ["--occlusion_targets", *(f"{t:g}" for t in a.occlusion_targets)]
        if a.positions:
            trials_cmd += ["--positions", *(str(p) for p in a.positions)]
        print("[cloud_trials] ensayos:", " ".join(trials_cmd), flush=True)
        code = subprocess.run(trials_cmd, cwd=ROOT, env=env).returncode
        if code != 0:
            raise SystemExit(f"run_policy_trials.py terminó con código {code}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    runs = sorted((out / a.run_name).glob("*/trials.csv"))
    if not runs:
        raise SystemExit(f"no quedó ningún trials.csv en {out / a.run_name}")
    for csv_path in runs:
        prune_images(csv_path.parent, a.keep_images_pos)
    print(f"[cloud_trials] listo: {[str(p.parent) for p in runs]}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
