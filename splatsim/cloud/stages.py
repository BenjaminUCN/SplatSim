"""Qué hace cada tipo de trabajo: qué sube, cómo corre en el pod y dónde quedan sus salidas.

El orquestador no conoce los detalles de cada etapa; agregar una es agregar una clase a STAGES.
"""
import os
import pickle
import re
from pathlib import Path

import yaml

from scripts.xarm5.data_jobs import SCRIPTS, command, dataset_path
from splatsim.cloud.estimate import (PNG_GB, Workload, convert_workload, render_workload, splat_workload,
                                     train_workload)
from splatsim.utils.data.episode_selection import load_discarded
from splatsim.robots.xarm.config import APPLE_NAME as APPLE
from splatsim.utils.paths import OBJECTS_CONFIG

IMAGE_DIRS = ("image_1", "images")     # los mismos que acepta lerobot_conversion.py
CAMERA_FILE = re.compile(r"^(.+_rgb)_\d+\.png$")
ROBOT = "xarm5_lite_plant_v3"          # el que usan por defecto render_recordings y mirror_recordings
# fotos COLMAP que cargaba el servidor del xArm cuando heredaba de la base del paper (Scene). El de
# xarm/sim_base.py lee solo el .ply; se siguen subiendo mientras haya imágenes del pod anteriores
SCENE_CAMERAS = (0, 347)
ALL_CAMERAS = ["base_rgb", "base_left_rgb", "base_right_rgb", "wrist_rgb"]   # --cameras por defecto
OBJECTS_FILE = OBJECTS_CONFIG
LABELS_DIR = "data/labels_path"         # etiquetas por link de articulated_robot_pipeline.py
LAB_CONFIGS = (OBJECTS_FILE, "configs/camera_poses.yaml", "configs/apple_grid.yaml", "configs/grasp_point.yaml",
               "configs/folder_configs.yaml")


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


def scan_states(folder: str | Path, fps: int) -> dict:
    """Lo que dejará renderizar una carpeta a fps: cada episodio dura lo que dicen sus sim_time."""
    root = Path(folder).expanduser()
    frames = episodes = size = 0
    for episode in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        pkls = sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem))
        if not pkls:
            continue
        with open(pkls[0], "rb") as f:
            first = pickle.load(f)
        if "sim_joint_positions" not in first:
            raise ValueError(f"{episode} no es una grabación con sim_clock=realtime: no se puede renderizar")
        with open(pkls[-1], "rb") as f:
            last = pickle.load(f)
        frames += int((last["sim_time"] - first["sim_time"]) * fps + 1e-9) + 1
        episodes += 1
        size += sum(p.stat().st_size for p in episode.iterdir() if p.is_file())
    if not episodes:
        raise ValueError(f"No hay episodios en {root}")
    return {"frames": frames, "episodes": episodes, "gb": round(size / 1e9, 2)}


def scene_assets(robot: str = ROBOT) -> tuple[list[dict], list[dict]]:
    """(qué subir, dónde lo espera el código en el pod). Lo mínimo para renderizar: sin las fotos de
    COLMAP, que pesan más que todo lo demás."""
    with open(OBJECTS_FILE) as f:
        objects = yaml.safe_load(f)
    data = Path.home() / "data"
    scene = Path(objects[robot]["source_path"]).expanduser()
    urdf = next(p for p in Path(objects[robot]["urdf_path"][0]).expanduser().parents if p.parent == data)
    apple = Path(objects[APPLE]["ply_path"]).expanduser().parents[2]
    photos = sorted(p.name for p in (scene / "images").iterdir())
    key = {path: f"assets/{path.relative_to(data)}" for path in (scene, urdf, apple)}
    uploads = [{"local": str(scene / "sparse"), "key": f"{key[scene]}/sparse"},
               {"local": objects[robot]["model_path"], "key": f"{key[scene]}/output"},
               *({"local": str(scene / "images" / photos[i]), "key": f"{key[scene]}/images/{photos[i]}"}
                 for i in SCENE_CAMERAS),
               {"local": str(urdf), "key": key[urdf]},
               {"local": str(apple), "key": key[apple]},
               {"local": "data/labels_path", "key": "assets/labels_path"}]
    inputs = [*({"key": k, "path": f"~/data/{path.relative_to(data)}"} for path, k in key.items()),
              {"key": "assets/labels_path", "path": "data/labels_path"}]
    return uploads, inputs


def pod_files() -> dict[str, str]:
    """Las configs del lab: el pod renderiza con las poses de cámara de hoy, no con las de la imagen.
    La carpeta de grabación solo tiene que existir."""
    return {**{path: Path(path).read_text() for path in LAB_CONFIGS}, "~/data/bc_data/gello/.keep": ""}


def describe_pipeline(plan: list[dict]) -> list[dict]:
    """Agrega a cada etapa de plan_pipeline lo que hace falta para estimarla, antes de que exista su entrada."""
    from splatsim.utils.render.camera_setup import arrangement_cameras
    stages, frames, cameras, gb = [], 0, [], 0.0
    for i, item in enumerate(plan):
        a, kind = dict(item["args"]), item["kind"]
        if kind == "convert":
            a["input"] = a["inputs"][0]
            exclude = []
            if i == 0:
                scan = scan_recordings(a["input"], a.get("cameras"), a.get("ignore_selection", False))
                frames, cameras, gb, exclude = scan["frames"], scan["cameras"], scan["gb"], scan["exclude"]
            stages.append({**item, "args": a, "frames": frames, "cameras": a.get("cameras") or cameras,
                           "recordings_gb": gb, "exclude": exclude})
            continue
        chosen = arrangement_cameras(a["arrangement"]) if a.get("arrangement") else None
        if kind == "render":
            scan = scan_states(a["input"], a["fps"])
            frames, cameras, gb = scan["frames"], chosen or ALL_CAMERAS, scan["gb"]
            item = {**item, "episodes": scan["episodes"]}
        elif i == 0:
            scan = scan_recordings(a["input"], chosen)
            frames, cameras, gb = scan["frames"], scan["cameras"], scan["gb"]
        input_gb = gb
        if kind == "mirror":
            cameras = chosen or cameras
            frames *= 2 if a.get("with_original", True) else 1
        stages.append({**item, "args": a, "frames": frames, "cameras": cameras, "input_gb": input_gb,
                       "workers": a.get("workers", 1)})
        gb = frames * len(cameras) * PNG_GB
    return stages


class Train:
    kind = "train"
    needs_gpu = True
    image = "train"
    output_feeds_children = False

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
    image = "train"                   # tiene LeRobot y FFmpeg
    output_feeds_children = False     # el entrenamiento lee datasets/, no la salida de esta corrida

    def workload(self, job: dict) -> Workload:
        return convert_workload(job["frames"], len(job["cameras"]), job["recordings_gb"])

    def refresh(self, job: dict, folder_gb) -> None:
        pass

    @staticmethod
    def _recordings_key(job: dict) -> str:
        return f"recordings/{Path(job['args']['input']).name}"

    def uploads(self, job: dict) -> list[dict]:
        if job.get("depends_on") is not None:
            return []            # la etapa anterior ya las dejó en el bucket
        return [{"local": job["args"]["input"], "key": self._recordings_key(job), "exclude": job["exclude"],
                 "packed": True}]

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


class Render:
    """render_recordings.py. Lo que deja queda directo en recordings/, en un paquete por episodio.
    La siguiente etapa solo parte cuando esta terminó bien."""
    kind = "render"
    needs_gpu = True
    image = "render"
    output_feeds_children = True

    def workload(self, job: dict) -> Workload:
        leaf_episodes = job.get("episodes", 0) if job["args"].get("leaf_dir") else 0
        return render_workload(self.kind, job["frames"], len(job["cameras"]), job["workers"], job["input_gb"],
                               leaf_episodes)

    def refresh(self, job: dict, folder_gb) -> None:
        pass

    @staticmethod
    def _input_key(job: dict) -> str:
        return f"recordings/{Path(job['args']['input']).name}"

    @staticmethod
    def _leaves_key(job: dict) -> str:
        return f"assets/occluders/{Path(job['args']['leaf_dir']).name}"

    def uploads(self, job: dict) -> list[dict]:
        own = [] if job.get("depends_on") is not None else [{"local": job["args"]["input"], "key": self._input_key(job),
                                                      "packed": True}]
        leaves = [{"local": job["args"]["leaf_dir"], "key": self._leaves_key(job)}] if job["args"].get("leaf_dir") else []
        return [*own, *leaves, *scene_assets()[0]]

    def pod(self, job: dict, run_key: str, pod_root: str) -> dict:
        a = job["args"]
        source = f"{pod_root}/recordings/{Path(a['input']).name}"
        out = f"{pod_root}/recordings/{Path(job['output']).name}"
        pod_args, leaves = {**a, "input": source}, []
        if a.get("leaf_dir"):
            pod_args["leaf_dir"] = f"~/data/occluders/{Path(a['leaf_dir']).name}"
            leaves = [{"key": self._leaves_key(job), "path": pod_args["leaf_dir"]}]
        pod_job = {"kind": self.kind, "args": pod_args, "output": out}
        return {"args": command(pod_job)[1:], "files": pod_files(), "clean": [],    # lo renderizado no se repite
                "inputs": [{"key": self._input_key(job), "path": source}, *leaves, *scene_assets()[1]],
                "outputs": [{"key": self.output_key(job), "path": out, "packed": True}]}

    def output_key(self, job: dict) -> str:
        return f"recordings/{Path(job['output']).name}"

    def target(self, job: dict) -> Path:
        return Path(job["output"]).expanduser()

    def publish_key(self, job: dict) -> str | None:
        return None

    def temporary_inputs(self, job: dict) -> list[str]:
        return [self._input_key(job)]

    def rate(self, job: dict) -> float | None:
        if job["args"].get("leaf_dir"):
            return None          # incluye el ajuste de las hojas: no es comparable con un render normal
        seconds = (job["result"].get("seconds") or {}).get("run")
        return round(job["frames"] / seconds, 3) if seconds else None


class Mirror(Render):
    kind = "mirror"


def probe_video(source: str | Path, fps: float = 0.0) -> dict:
    """Cuántas fotos saldrán del video (o hay en la carpeta) y cuánto pesa lo que se sube."""
    path = Path(source).expanduser()
    if path.is_dir():
        photos = [p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not photos:
            raise ValueError(f"{path} no tiene fotos")
        return {"frames": len(photos), "gb": round(sum(p.stat().st_size for p in photos) / 1e9, 2)}
    if not path.is_file():
        raise ValueError(f"No existe {path}")
    import json
    import subprocess
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=nb_frames,avg_frame_rate,duration:format=duration", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise ValueError(f"ffprobe no pudo leer {path.name}: {done.stderr.strip()[-200:]}")
    info = json.loads(done.stdout)
    stream = (info.get("streams") or [{}])[0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    if fps > 0:
        frames = int(duration * fps) + 1
    elif str(stream.get("nb_frames", "")).isdigit():
        frames = int(stream["nb_frames"])
    else:
        num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
        frames = int(duration * float(num) / float(den or 1))
    if frames < 3:
        raise ValueError(f"{path.name} da solo {frames} fotos")
    return {"frames": frames, "gb": round(path.stat().st_size / 1e9, 2)}


IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class _FlowLists(yaml.SafeDumper):
    """Las listas de números en una línea, como el resto de objects.yaml."""


_FlowLists.add_representer(list, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=all(isinstance(x, (int, float)) for x in data)))


def scene_entry(robot: str, scene: str | Path, base: str = ROBOT) -> str:
    """Una entrada nueva de objects.yaml para la escena, copiada de base: falta la matriz de CloudCompare
    (queda la identidad) y quizá joint_states, la postura del brazo al grabar el video."""
    with open(OBJECTS_FILE) as f:
        objects = yaml.safe_load(f)
    if base not in objects:
        raise ValueError(f"{base} no está en {OBJECTS_FILE}")
    folder = Path(scene).expanduser()
    shown = f"~/{folder.relative_to(Path.home())}" if folder.is_relative_to(Path.home()) else str(folder)
    entry = {**objects[base], "source_path": shown, "model_path": f"{shown}/output",
             "transformation": {"matrix": IDENTITY},
             "aabb": {**objects[base].get("aabb", {}), "bounding_box": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}}
    entry.pop("object_overrides", None)       # pertenece a la escena base, no a esta
    return yaml.dump({robot: entry}, Dumper=_FlowLists, sort_keys=False, allow_unicode=True)


def latest_splat(model_path: Path) -> Path | None:
    """El que carga Scene(load_iteration=-1): la iteración más alta."""
    plys = list(model_path.glob("point_cloud/iteration_*/point_cloud.ply"))
    return max(plys, key=lambda p: int(p.parent.name.split("_")[1])) if plys else None


def alignment_problems(robot: str, scene: str | Path) -> list[str]:
    """Lo que falta para dar por alineada una escena; vacío si está lista. Las etiquetas por link tienen
    que haber salido de la matriz y la caja de ahora: articulated_robot_pipeline.py etiqueta los puntos
    del splat que caen dentro de la caja después de transformarlos, así que se cuentan esos mismos."""
    import numpy as np
    from plyfile import PlyData

    with open(OBJECTS_FILE) as f:
        objects = yaml.safe_load(f)
    folder = Path(scene).expanduser().resolve()
    entry = objects.get(robot)
    if entry is None:
        return [f"{robot} no está en {OBJECTS_FILE}"]
    problems = []
    if Path(entry.get("source_path", "")).expanduser().resolve() != folder:
        problems.append(f"source_path de {robot} es {entry.get('source_path')}, no {folder}")
    model = Path(entry.get("model_path", "")).expanduser().resolve()
    if model != folder / "output":
        problems.append(f"model_path de {robot} es {entry.get('model_path')}, no {folder / 'output'}")
    try:
        matrix = np.array((entry.get("transformation") or {})["matrix"], dtype=float)
        assert matrix.shape == (4, 4)
    except (KeyError, TypeError, ValueError, AssertionError):
        return [*problems, f"transformation.matrix de {robot} no es una matriz de 4×4"]
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        problems.append(f"la última fila de la matriz es {matrix[3].tolist()}, no [0, 0, 0, 1]")
    if np.allclose(matrix, np.eye(4)):
        problems.append("la matriz sigue siendo la identidad: pega la de CloudCompare (Transformation History)")
    labels = Path(LABELS_DIR) / f"{robot}_labels.npy"
    if not labels.is_file():
        return [*problems, f"falta {labels}: corre articulated_robot_pipeline.py --robot_name {robot}"]
    if problems:
        return problems
    ply = latest_splat(model)
    if ply is None:
        return [f"no hay splat en {model}/point_cloud"]
    vertex = PlyData.read(str(ply))["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    moved = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    low, high = (np.array(corner, dtype=float) for corner in entry["aabb"]["bounding_box"])
    inside = int(np.all((moved > low) & (moved < high), axis=1).sum())
    count = len(np.load(labels))
    if inside == 0:
        return ["ningún punto del splat cae en la caja del robot: corre articulated_robot_pipeline.py"]
    if abs(count - inside) > max(10, inside // 1000):
        return [f"{labels} tiene {count} etiquetas, pero con la matriz y la caja de ahora caen {inside} puntos: "
                f"vuelve a correr articulated_robot_pipeline.py --robot_name {robot}"]
    return []


class Splat:
    """splat_from_video.py: video → COLMAP → splat. Baja al lab y espera la alineación en CloudCompare
    (needs_approval): mientras, lo que dependa de ella sigue en cola."""
    kind = "splat"
    needs_gpu = True
    image = "render"             # tiene el rasterizador; pycolmap-cuda12 trae COLMAP con CUDA
    script = "scripts/xarm5/splat_from_video.py"
    output_feeds_children = False
    needs_approval = True

    def workload(self, job: dict) -> Workload:
        a = job["args"]
        return splat_workload(job["frames"], a.get("matcher", "exhaustive"), a["iterations"], job["source_gb"])

    def refresh(self, job: dict, folder_gb) -> None:
        pass

    @staticmethod
    def _scene(job: dict) -> str:
        return Path(job["output"]).name

    def _video_key(self, job: dict) -> str:
        return f"videos/{self._scene(job)}"

    def uploads(self, job: dict) -> list[dict]:
        source = Path(job["args"]["source"]).expanduser()
        return [{"local": str(source), "key": f"{self._video_key(job)}/{source.name}"}]

    def pod(self, job: dict, run_key: str, pod_root: str) -> dict:
        a = job["args"]
        videos, out = f"{pod_root}/videos/{self._scene(job)}", f"{pod_root}/scenes/{self._scene(job)}"
        args = [f"{videos}/{Path(a['source']).name}", "--out", out, "--fps", str(a.get("fps", 0.0)),
                "--matcher", a.get("matcher", "exhaustive"), "--iterations", str(a["iterations"]), "--clean"]
        return {"args": args, "files": {}, "clean": [out],
                "inputs": [{"key": self._video_key(job), "path": videos}],
                # images/ son ~1300 fotos: en un paquete, en vez de un objeto del bucket cada una
                "outputs": [{"key": self.output_key(job), "path": out, "packed": True}]}

    def output_key(self, job: dict) -> str:
        return f"{job['run_key']}/outputs/scene"

    def target(self, job: dict) -> Path:
        return Path(job["output"]).expanduser()

    def publish_key(self, job: dict) -> str | None:
        return None

    def temporary_inputs(self, job: dict) -> list[str]:
        return [self._video_key(job)]

    def rate(self, job: dict) -> float | None:
        seconds = (job["result"].get("seconds") or {}).get("run")
        return round(job["frames"] / seconds, 4) if seconds else None


STAGES = {stage.kind: stage for stage in (Train(), Convert(), Render(), Mirror(), Splat())}
DEFAULT_SCRIPTS = {kind: getattr(stage, "script", None) or SCRIPTS[kind] for kind, stage in STAGES.items()}
