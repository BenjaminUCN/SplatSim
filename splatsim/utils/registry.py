"""Lectura de configs/registry.yaml (lo genera scripts/build_registry.py)."""
import os

import yaml

REGISTRY_PATH = "configs/registry.yaml"


def load_registry(path: str = REGISTRY_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
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
