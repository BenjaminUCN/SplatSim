"""Lectura de configs/registry.yaml (lo genera scripts/xarm5/build_registry.py)."""
import json
import os

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


def model_for_checkpoint(checkpoint: str, registry: dict | None = None) -> tuple[str, dict] | None:
    """(clave, entrada) del modelo de la carpeta `checkpoint`, o None si no está registrado."""
    registry = registry if registry is not None else load_registry()
    target = os.path.realpath(os.path.expanduser(checkpoint))
    for key, model in (registry.get("models") or {}).items():
        if os.path.realpath(model["path"]) == target:
            return key, model
    return None


def dataset_for_checkpoint(checkpoint: str, registry: dict | None = None) -> dict | None:
    """Entrada del dataset con que se entrenó el modelo de la carpeta `checkpoint`."""
    registry = registry if registry is not None else load_registry()
    found = model_for_checkpoint(checkpoint, registry)
    if found is None or not found[1].get("dataset"):
        return None
    return (registry.get("datasets") or {}).get(found[1]["dataset"])


# ── modelos cargados a mano en el panel (rutas de esta máquina) ──────────────
LOCAL_MODELS_PATH = "configs/local_models.yaml"   # no va a git: las rutas son de cada máquina
_LOCAL_HEADER = """\
# Models loaded with the control panel's "Load model" (Policy folder). Paths are of this
# machine, so this file is not in git. Delete an entry to forget it.
"""


def describe_model(path: str) -> dict:
    """Entrada de registro de la carpeta de un modelo LeRobot entrenado; ValueError si no sirve."""
    path = os.path.realpath(os.path.expanduser(path.strip()))
    if not os.path.isdir(path):
        raise ValueError(f"`{path}` is not a folder")
    missing = [f for f in ("config.json", "model.safetensors") if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise ValueError(f"`{path}` has no {' / '.join(missing)}: pick the model's final folder "
                         "(the one train_diffusion.py saves at the end)")
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return {
        "path": path,
        "finished": True,
        "policy": cfg.get("type"),
        "cameras": [k.removeprefix("observation.images.")
                    for k in cfg.get("input_features", {}) if k.startswith("observation.images.")],
        "n_obs_steps": cfg.get("n_obs_steps"),
        "horizon": cfg.get("horizon"),
        "n_action_steps": cfg.get("n_action_steps"),
    }


def describe_dataset(path: str) -> dict:
    """Dataset LeRobot local (meta/info.json); ValueError si no sirve."""
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
    """Etiqueta -> entrada, solo las que siguen existiendo en esta máquina."""
    try:
        with open(path) as f:
            models = (yaml.safe_load(f) or {}).get("models") or {}
    except FileNotFoundError:
        return {}
    return {k: m for k, m in models.items() if os.path.isdir(m.get("path", ""))}


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
