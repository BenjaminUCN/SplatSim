"""Trabajos de datos para la cola de train_server.py: generar demos con el planificador,
renderizar, espejar y convertir a LeRobot.

Cada trabajo es un dict de jobs.yaml con kind ("plan" | "render" | "mirror" | "convert"), args y
output (la carpeta que deja, conocida al encolarlo). plan_pipeline arma la cadena plan → render →
mirror → convert: cada paso toma la salida del anterior y espera a que termine (depends_on), y el dataset
resultante se puede elegir en el formulario de entrenamiento antes de que exista.

Cada tipo usa un carril (LANES): entrenar, renderizar y espejar cargan la GPU (y no caben dos a la
vez en su memoria); planificar (PyBullet sin cámaras) y convertir usan solo CPU, así que pueden
correr al lado de uno de GPU.
"""
from pathlib import Path

DATA_KINDS = ("plan", "render", "mirror", "convert")
SCRIPTS = {"train": "scripts/xarm5/train_diffusion.py",
           "plan": "scripts/xarm5/plan_trajectories.py",
           "render": "scripts/xarm5/render_recordings.py",
           "mirror": "scripts/xarm5/mirror_recordings.py",
           "convert": "splatsim/utils/lerobot_conversion.py"}
LANES = {"train": "gpu", "plan": "cpu", "render": "gpu", "mirror": "gpu", "convert": "cpu"}
ROBOT_TYPE = "xarm5_lite"      # --robot_type por defecto del convertidor, va en el nombre del dataset


def kind(job: dict) -> str:
    return job.get("kind", "train")       # los trabajos anteriores a esto son de entrenamiento


def lane(job: dict) -> str:
    return LANES[kind(job)]


def dataset_path(output_dir: str, fps: int, suffix: str) -> str:
    """Carpeta que crea lerobot_conversion.py (root_dir en create_lerobot_dataset)."""
    return str(Path(output_dir).expanduser() / f"splatsim_{ROBOT_TYPE}_{fps}fps_{suffix}")


def command(job: dict) -> list[str]:
    """Script y argumentos (sin el intérprete) de un trabajo de datos."""
    a, k = job["args"], kind(job)
    cmd = [SCRIPTS[k]]
    if k == "plan":
        cmd += ["--out", job["output"], "--episodes", str(a["episodes"]), "--positions", a["positions"],
                "--perturb", str(a.get("perturb", 0.0))]
        if a.get("seed") is not None:
            cmd += ["--seed", str(a["seed"])]
    elif k == "render":
        cmd += [a["input"], "--fps", str(a["fps"]), "--out", job["output"]]
        if a.get("arrangement"):
            cmd += ["--arrangement", a["arrangement"]]
        if a.get("workers", 1) > 1:
            cmd += ["--workers", str(a["workers"])]
        if a.get("leaf_dir"):
            cmd += ["--leaf_dir", a["leaf_dir"], "--seed", str(a.get("seed", 0))]
            if a.get("occlusion_targets"):
                cmd += ["--occlusion_targets", *(str(t) for t in a["occlusion_targets"])]
    elif k == "mirror":
        cmd += [a["input"], "--out", job["output"]]
        if not a.get("with_original", True):
            cmd.append("--no-with-original")
        if a.get("arrangement"):
            cmd += ["--arrangement", a["arrangement"]]
        if a.get("workers", 1) > 1:
            cmd += ["--workers", str(a["workers"])]
    elif k == "convert":
        cmd += ["--input_dir", *a["inputs"], "--output_dir", str(Path(a["output_dir"]).expanduser()),
                "--fps", str(a["fps"]), "--repo_suffix", a["suffix"]]
        if a.get("cameras"):
            cmd += ["--cameras", *a["cameras"]]
        if a.get("ignore_selection"):
            cmd.append("--ignore_selection")
        if not a.get("push"):
            cmd.append("--no_push")
    else:
        raise ValueError(f"tipo de trabajo desconocido: {k}")
    return cmd


def plan_pipeline(source: str, render: dict | None, mirror: dict | None, convert: dict | None,
                  plan: dict | None = None) -> list[dict]:
    """Trabajos encadenados (sin id; depends_on = índice del anterior en la lista).

    plan: {episodes, positions, seed, perturb}: genera las demos en source primero;
    render: {fps, arrangement, workers y, opcional, leaf_dir, occlusion_targets, seed};
    mirror: {with_original, arrangement, workers};
    convert: {output_dir, fps, cameras, suffix, ignore_selection, push}. Los None se omiten."""
    if render and render.get("leaf_dir") and mirror:
        # mirror_recordings.py renderiza sin hojas pero copia el meta.json que dice que las tienen
        raise ValueError("El espejo no conserva las hojas: desactiva Mirror o las oclusiones")
    jobs, current = [], str(Path(source).expanduser())
    if plan:
        jobs.append({"kind": "plan", "name": f"plan {Path(current).name}", "args": dict(plan),
                     "output": current})
    if render:
        out = f"{current}_{render['fps']}hz{'_leaves' if render.get('leaf_dir') else ''}"
        jobs.append({"kind": "render", "name": f"render {Path(current).name}",
                     "args": {"input": current, **render}, "output": out})
        current = out
    if mirror:
        out = f"{current}_mirrored"
        jobs.append({"kind": "mirror", "name": f"mirror {Path(current).name}",
                     "args": {"input": current, **mirror}, "output": out})
        current = out
    if convert:
        out = dataset_path(convert["output_dir"], convert["fps"], convert["suffix"])
        jobs.append({"kind": "convert", "name": f"convert {Path(current).name}",
                     "args": {"inputs": [current], **convert}, "output": out})
    for i, job in enumerate(jobs):
        job["depends_on"] = i - 1 if i else None
    return jobs
