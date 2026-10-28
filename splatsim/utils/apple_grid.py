"""Grilla de posiciones del tomate para los ensayos (configs/apple_grid.yaml).

En el sim, run_policy_trials.py pone el tomate en cada posición; en el robot real el panel indica
la posición y el operador lo pone a mano.
"""
import yaml

APPLE_GRID_PATH = "configs/apple_grid.yaml"


def load_grid(path: str = APPLE_GRID_PATH) -> list:
    """Posiciones guardadas (lista vacía si no hay archivo)."""
    try:
        with open(path) as f:
            return (yaml.safe_load(f) or {}).get("positions") or []
    except FileNotFoundError:
        return []
