"""Lectura de configs/registry.yaml (lo genera scripts/xarm5/build_registry.py)."""
import json
import os
import re

import yaml

from splatsim.utils.paths import load_yaml

REGISTRY_PATH = "configs/registry.yaml"


def load_registry(path: str = REGISTRY_PATH) -> dict:
    try:
        return load_yaml(path)
    except FileNotFoundError:
        return {}


def model_label(key: str, model: dict) -> str:
    """Nombre para mostrar: name manual, o suggested_name, o la carpeta."""
    return model.get("name") or model.get("suggested_name") or key


def loadable_models(registry: dict) -> dict:
    """Modelos con pesos finales (los que tienen solo checkpoints intermedios se omiten)."""
    return {k: m for k, m in (registry.get("models") or {}).items()
            if m.get("finished") and os.path.isdir(m["path"])}


# ── modelos y datasets en el Hugging Face Hub ────────────────────────────────
# train_diffusion.py sube cada modelo a <hub_user>/<clave del registro con _ en vez de />
# (p.ej. robot_learning_tutorial/diffusion_1021_1640 -> robot_learning_tutorial_diffusion_1021_1640)
# y lerobot_conversion.py cada dataset a <hub_user>/<clave del dataset>, así se pueden usar en una
# máquina que no tiene las carpetas.
_HUB_URL = re.compile(r"^https?://huggingface\.co/(?:datasets/)?")
_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def hub_repo_id(text: str | None) -> str | None:
    """`user/repo` si `text` es un repo del Hub (id o URL de huggingface.co), None si es una ruta local."""
    if not text:
        return None
    text = text.strip()
    if _HUB_URL.match(text):
        text = "/".join(_HUB_URL.sub("", text).split("/")[:2])   # sin /tree/main/...
    elif text.startswith(("/", "~", ".")) or os.path.exists(text):
        return None
    return text if _REPO_ID.match(text) else None


def _hf():
    try:
        import huggingface_hub
    except ImportError:
        raise ValueError("huggingface_hub is not installed in this environment "
                         "(pip install huggingface_hub)") from None
    return huggingface_hub


def model_for_checkpoint(checkpoint: str, registry: dict | None = None) -> tuple[str, dict] | None:
    """(clave, entrada) del modelo de la carpeta o repo del Hub `checkpoint`, o None si no está."""
    registry = registry if registry is not None else load_registry()
    models = (registry.get("models") or {}).items()
    repo = hub_repo_id(checkpoint)
    if repo:
        name = repo.split("/")[1]
        return next(((k, m) for k, m in models if k.replace("/", "_") == name), None)
    target = os.path.realpath(os.path.expanduser(checkpoint))
    return next(((k, m) for k, m in models if os.path.realpath(m["path"]) == target), None)


def dataset_for_checkpoint(checkpoint: str, registry: dict | None = None) -> dict | None:
    """Entrada del dataset con que se entrenó el modelo de la carpeta `checkpoint`."""
    registry = registry if registry is not None else load_registry()
    found = model_for_checkpoint(checkpoint, registry)
    if found is None or not found[1].get("dataset"):
        return None
    return (registry.get("datasets") or {}).get(found[1]["dataset"])


def dataset_source_for_checkpoint(checkpoint: str, registry: dict | None = None) -> str | None:
    """Carpeta del dataset del modelo si existe en esta máquina; si no, y el modelo es del Hub,
    el repo del dataset con el mismo usuario (los datasets dentro de la carpeta de su grabación,
    clave <grabación>/<dataset>, se suben como <dataset>). None si el modelo no está en el registro."""
    registry = registry if registry is not None else load_registry()
    found = model_for_checkpoint(checkpoint, registry)
    dataset = dataset_for_checkpoint(checkpoint, registry)
    if dataset is None:
        return None
    if os.path.isdir(dataset["path"]):
        return dataset["path"]
    repo = hub_repo_id(checkpoint)
    return f"{repo.split('/')[0]}/{found[1]['dataset'].split('/')[-1]}" if repo else None


# ── modelos cargados a mano en el panel (rutas de esta máquina) ──────────────
LOCAL_MODELS_PATH = "configs/local_models.yaml"   # no va a git: las rutas son de cada máquina
_LOCAL_HEADER = """\
# Models loaded with the control panel's "Load model" (Policy folder). Paths are of this
# machine, so this file is not in git. Delete an entry to forget it.
"""


def describe_model(path: str) -> dict:
    """Entrada de registro de la carpeta (o repo del Hub) de un modelo LeRobot entrenado;
    ValueError si no sirve."""
    repo = hub_repo_id(path)
    if repo:
        hf = _hf()
        try:
            files = hf.list_repo_files(repo)
            missing = [f for f in ("config.json", "model.safetensors") if f not in files]
            if missing:
                raise ValueError(f"Hub model `{repo}` has no {' / '.join(missing)}")
            with open(hf.hf_hub_download(repo, "config.json")) as f:
                return _model_entry(repo, json.load(f), hub=True)
        except hf.errors.HfHubHTTPError as e:
            raise ValueError(f"cannot read Hub model `{repo}`: {e}") from None
    path = os.path.realpath(os.path.expanduser(path.strip()))
    if not os.path.isdir(path):
        raise ValueError(f"`{path}` is not a folder")
    missing = [f for f in ("config.json", "model.safetensors") if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise ValueError(f"`{path}` has no {' / '.join(missing)}: pick the model's final folder "
                         "(the one train_diffusion.py saves at the end)")
    with open(os.path.join(path, "config.json")) as f:
        return _model_entry(path, json.load(f))


def _model_entry(path: str, cfg: dict, hub: bool = False) -> dict:
    return {
        "path": path,
        **({"hub": True} if hub else {}),
        "finished": True,
        "policy": cfg.get("type"),
        "cameras": [k.removeprefix("observation.images.")
                    for k in cfg.get("input_features", {}) if k.startswith("observation.images.")],
        "n_obs_steps": cfg.get("n_obs_steps"),
        "horizon": cfg.get("horizon"),
        "n_action_steps": cfg.get("n_action_steps"),
    }


def describe_dataset(path: str) -> dict:
    """Dataset LeRobot local o del Hub (meta/info.json); ValueError si no sirve."""
    repo = hub_repo_id(path)
    if repo:
        hf = _hf()
        try:
            info_path = hf.hf_hub_download(repo, "meta/info.json", repo_type="dataset")
        except hf.errors.HfHubHTTPError as e:
            raise ValueError(f"cannot read Hub dataset `{repo}` (meta/info.json), it may not be on "
                             f"the Hub: set Dataset ({type(e).__name__})") from None
        path = repo
    else:
        path = os.path.realpath(os.path.expanduser(path.strip()))
        info_path = os.path.join(path, "meta", "info.json")
        if not os.path.exists(info_path):
            raise ValueError(f"`{path}` has no meta/info.json: pick the LeRobot dataset folder")
    with open(info_path) as f:
        info = json.load(f)
    return {
        "path": path,
        "episodes": info.get("total_episodes"),
        "fps": info.get("fps"),
        "cameras": [k.removeprefix("observation.images.")
                    for k in info.get("features", {}) if k.startswith("observation.images.")],
    }


def load_local_models(path: str = LOCAL_MODELS_PATH) -> dict:
    """Etiqueta -> entrada, solo las que siguen existiendo en esta máquina (o están en el Hub)."""
    try:
        with open(path) as f:
            models = (yaml.safe_load(f) or {}).get("models") or {}
    except FileNotFoundError:
        return {}
    return {k: m for k, m in models.items() if m.get("hub") or os.path.isdir(m.get("path", ""))}


def save_local_model(label: str, entry: dict, path: str = LOCAL_MODELS_PATH) -> None:
    try:
        with open(path) as f:
            models = (yaml.safe_load(f) or {}).get("models") or {}
    except FileNotFoundError:
        models = {}
    models[label] = entry
    with open(path, "w") as f:
        f.write(_LOCAL_HEADER)
        yaml.safe_dump({"models": models}, f, sort_keys=False, default_flow_style=None)
