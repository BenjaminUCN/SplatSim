"""Qué hace cada tipo de trabajo: qué sube, cómo corre en el pod y dónde quedan sus salidas.

El orquestador no conoce los detalles de cada etapa; agregar una es agregar una clase a STAGES.
"""
import os
import re
from pathlib import Path

import yaml

from scripts.xarm5.data_jobs import SCRIPTS, command, dataset_path
from splatsim.cloud.estimate import Workload, convert_workload, train_workload
from splatsim.utils.episode_selection import load_discarded

IMAGE_DIRS = ("image_1", "images")     # los mismos que acepta lerobot_conversion.py
CAMERA_FILE = re.compile(r"^(.+_rgb)_\d+\.png$")


def scan_recordings(folder: str | Path, cameras: list[str] | None = None, ignore_selection: bool = False) -> dict:
    """Lo que hay que subir para convertir una carpeta: sin descartados ni cámaras que no se usan."""
    root = Path(folder).expanduser()
    discarded = set() if ignore_selection else load_discarded(root)
    frames = episodes = size = 0
    found: set[str] = set()
    episode_dirs = sorted(p for p in root.iterdir() if p.is_dir() and next(p.glob("*.pkl"), None))
    for episode in episode_dirs:
        if episode.name in discarded:
            continue
        episodes += 1
        for item in os.scandir(episode):
            if item.name.endswith(".pkl"):
                frames += 1
                size += item.stat().st_size
            elif item.is_dir() and item.name in IMAGE_DIRS:
                for image in os.scandir(item.path):
                    match = CAMERA_FILE.match(image.name)
                    if match:
                        found.add(match.group(1))
                    if match is None or not cameras or match.group(1) in cameras:
                        size += image.stat().st_size
    if not episodes:
        raise ValueError(f"No hay episodios que convertir en {root}")
    if not found:
        raise ValueError(f"{root} no tiene imágenes: primero hay que renderizarla")
    if cameras and (missing := set(cameras) - found):
        raise ValueError(f"Cámaras {sorted(missing)} no están en {root} (hay {sorted(found)})")
    used = sorted(cameras or found)
    exclude = [f"/{e.name}/**" for e in episode_dirs if e.name in discarded]
    exclude += [f"**/{cam}_*.png" for cam in sorted(found - set(used))]
    return {"frames": frames, "episodes": episodes, "cameras": used, "gb": round(size / 1e9, 2), "exclude": exclude}


class Train:
    kind = "train"
    needs_gpu = True

    def workload(self, job: dict) -> Workload:
        return train_workload(job["config"], job["dataset_gb"])

    def refresh(self, job: dict, folder_gb) -> None:
        """Con una cadena, el tamaño del dataset se había previsto: ahora ya existe."""
        job["dataset_gb"] = round(folder_gb(Path(job["dataset"]).expanduser()), 2)

    def uploads(self, job: dict) -> list[dict]:
        return [{"local": job["dataset"], "key": f"datasets/{Path(job['dataset']).name}", "exclude": []}]

    def pod(self, job: dict, run_key: str, pod_root: str) -> dict:
        cfg = job["config"]
        config_path = f"/tmp/{cfg['model_name']}.yaml"
        return {"args": ["--config", config_path], "clean": [],
                "files": {config_path: yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)},
                "inputs": [{"key": f"datasets/{Path(job['dataset']).name}", "path": cfg["dataset_root"]}],
                "outputs": [{"key": self.output_key(job), "path": f"{cfg['output_root']}/{cfg['model_name']}"}]}

    def output_key(self, job: dict) -> str:
        return f"{job['run_key']}/outputs/model"

    def target(self, job: dict) -> Path:
        cfg = job["config"]
        return Path(cfg["output_root"]).expanduser() / cfg["model_name"]

    def publish_key(self, job: dict) -> str | None:
        return None

    def temporary_inputs(self, job: dict) -> list[str]:
        return []                # un dataset se vuelve a usar en otros entrenamientos

    def rate(self, job: dict) -> float | None:
        return job["result"].get("rate")


class Convert:
    kind = "convert"
    needs_gpu = False

    def workload(self, job: dict) -> Workload:
        return convert_workload(job["frames"], len(job["cameras"]), job["recordings_gb"])

    def refresh(self, job: dict, folder_gb) -> None:
        pass

    @staticmethod
    def _recordings_key(job: dict) -> str:
        return f"recordings/{Path(job['args']['input']).name}"

    def uploads(self, job: dict) -> list[dict]:
        return [{"local": job["args"]["input"], "key": self._recordings_key(job), "exclude": job["exclude"]}]

    def pod(self, job: dict, run_key: str, pod_root: str) -> dict:
        a = job["args"]
        recordings = f"{pod_root}/recordings/{Path(a['input']).name}"
        datasets = f"{pod_root}/datasets"
        out = dataset_path(datasets, a["fps"], a["suffix"])
        pod_job = {"kind": "convert", "args": {"inputs": [recordings], "output_dir": datasets, "fps": a["fps"],
                                               "suffix": a["suffix"], "cameras": a.get("cameras"),
                                               "ignore_selection": a.get("ignore_selection"), "push": False}}
        return {"args": command(pod_job)[1:], "files": {},
                "clean": [out],     # LeRobot no acepta una carpeta existente, y el contenedor puede reiniciarse
                "inputs": [{"key": self._recordings_key(job), "path": recordings}],
                "outputs": [{"key": self.output_key(job), "path": out}]}

    def output_key(self, job: dict) -> str:
        return f"{job['run_key']}/outputs/dataset"

    def target(self, job: dict) -> Path:
        return Path(job["output"]).expanduser()

    def publish_key(self, job: dict) -> str | None:
        """datasets/ solo guarda datasets completos: el siguiente paso de la cadena lo lee de ahí."""
        return f"datasets/{self.target(job).name}"

    def temporary_inputs(self, job: dict) -> list[str]:
        return [self._recordings_key(job)]

    def rate(self, job: dict) -> float | None:
        """Frames por segundo de toda la ejecución; la barra del conversor cuenta episodios."""
        seconds = (job["result"].get("seconds") or {}).get("run")
        return round(job["frames"] / seconds, 3) if seconds else None


STAGES = {stage.kind: stage for stage in (Train(), Convert())}
DEFAULT_SCRIPTS = {kind: SCRIPTS[kind] for kind in STAGES}
