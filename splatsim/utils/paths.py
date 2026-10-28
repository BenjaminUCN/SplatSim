"""Rutas portables entre máquinas: en configs y en el código se escriben con ~ (el home de
cada usuario), así el mismo repo sirve en /home/magister y en /home/doctoradoia."""
import os

import yaml


def expand(path: str) -> str:
    """~ -> el home de este usuario (y $VAR -> su valor)."""
    return os.path.expanduser(os.path.expandvars(path))


def contract(path: str) -> str:
    """El home de este usuario -> ~ (para escribir rutas que sirvan en otra máquina)."""
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _map_strings(obj, fn):
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    return fn(obj) if isinstance(obj, str) else obj


def expand_tree(obj):
    """expand() en cada texto que empieza con ~, en dicts y listas anidados."""
    return _map_strings(obj, lambda s: expand(s) if s.startswith("~") else s)


def contract_tree(obj):
    """contract() en cada texto de dicts y listas anidados."""
    return _map_strings(obj, contract)


def load_yaml(path) -> dict:
    """yaml.safe_load con las rutas ~/... ya expandidas; {} si el archivo está vacío."""
    with open(path) as f:
        return expand_tree(yaml.safe_load(f) or {})
