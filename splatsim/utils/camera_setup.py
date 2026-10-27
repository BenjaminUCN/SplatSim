"""Cámaras base y arreglos de cámaras (configs/camera_poses.yaml).

Las cámaras base son fijas en el mundo y se graban todas; un arreglo es solo el
subconjunto de cámaras con que se entrena y evalúa (p.ej. 2cam = centro + muñeca,
3cam = izquierda + derecha + muñeca). Así todos los arreglos comparten las mismas demos.

Solo depende de yaml para poder usarlo desde launch_nodes, el servidor y el entrenamiento.
"""
import yaml

CAMERA_POSES_PATH = "configs/camera_poses.yaml"

_HEADER = """\
# Fixed base cameras (pose in PyBullet world coordinates: xyz in meters, rpy in radians)
# and camera arrangements (subsets used to train and evaluate; every camera is recorded).
# Poses are written by the "Save Camera Poses" button of the xArm5 control panel;
# arrangements are edited by hand.
"""


def load_camera_setup(path: str = CAMERA_POSES_PATH) -> dict:
    """{"cameras": {nombre: {"xyz": [...], "rpy": [...]}}, "arrangements": {nombre: [cámaras]}}."""
    with open(path) as f:
        setup = yaml.safe_load(f) or {}
    setup.setdefault("cameras", {})
    setup.setdefault("arrangements", {})
    return setup


def base_camera_poses(path: str = CAMERA_POSES_PATH) -> dict:
    """nombre -> (xyz, rpy) de cada cámara base."""
    return {name: (tuple(p["xyz"]), tuple(p["rpy"]))
            for name, p in load_camera_setup(path)["cameras"].items()}


def arrangement_cameras(name: str, path: str = CAMERA_POSES_PATH) -> list[str]:
    arrangements = load_camera_setup(path)["arrangements"]
    if name not in arrangements:
        raise ValueError(f"Arreglo de cámaras '{name}' no está en {path} (hay {sorted(arrangements)})")
    return list(arrangements[name])


def save_base_camera_poses(poses: dict, path: str = CAMERA_POSES_PATH) -> None:
    """Escribe nombre -> (xyz, rpy) conservando los arreglos del archivo."""
    try:
        arrangements = load_camera_setup(path)["arrangements"]
    except FileNotFoundError:
        arrangements = {}
    cameras = {name: {"xyz": [round(float(v), 4) for v in xyz],
                      "rpy": [round(float(v), 4) for v in rpy]}
               for name, (xyz, rpy) in poses.items()}
    with open(path, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump({"cameras": cameras, "arrangements": arrangements}, f,
                       default_flow_style=None, sort_keys=False)
