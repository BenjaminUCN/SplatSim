"""Configuración de la estación del xArm5 real (configs/xarm5_real.yaml)."""
import yaml

REAL_CONFIG_PATH = "configs/xarm5_real.yaml"


def load_real_config(path: str = REAL_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}
