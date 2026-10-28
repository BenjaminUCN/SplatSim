"""Configuración de la estación del xArm5 real (configs/xarm5_real.yaml)."""
from splatsim.utils.paths import load_yaml

REAL_CONFIG_PATH = "configs/xarm5_real.yaml"


def load_real_config(path: str = REAL_CONFIG_PATH) -> dict:
    return load_yaml(path)
