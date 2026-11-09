"""De qué grabación sale cada episodio de un dataset LeRobot (<dataset>/meta/splatsim_sources.yaml).

Lo escribe splatsim/utils/data/lerobot_conversion.py al convertir y lo lee scripts/xarm5/build_registry.py.
Viaja dentro del dataset, así que el enlace dataset -> grabación sobrevive aunque la carpeta
convertida no exista en esta máquina (p.ej. un render hecho en un pod de la nube, del que solo
se baja el dataset). Si la carpeta convertida es un render de render_recordings.py, se guarda
también su render_info.yaml, con la grabación original en `source`.
"""
from pathlib import Path

import yaml

SOURCES_FILE = "splatsim_sources.yaml"
RENDER_INFO_FILE = "render_info.yaml"   # lo escribe scripts/xarm5/render_recordings.py

_HEADER = """\
# Recording folder of each episode of this dataset (written by splatsim/utils/data/lerobot_conversion.py,
# read by scripts/xarm5/build_registry.py). dataset_episodes[i] comes from episodes[i] of `folder`.
"""


def describe_folder(folder: Path) -> dict:
    """Entrada de una carpeta convertida, sin episodios todavía."""
    entry = {"folder": Path(folder).name, "episodes": [], "dataset_episodes": []}
    info_path = Path(folder) / RENDER_INFO_FILE
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text()) or {}
        leaves = (info.get("leaf_variation") or {}).get("leaves") or []
        entry["render"] = {
            "source": Path(info["source"]).name if info.get("source") else None,
            "fps": info.get("fps"),
            "cameras": info.get("cameras"),
            "leaves": len(leaves),
            "rendered": info.get("rendered"),
        }
    return entry


def write_sources(dataset_root: Path, sources: list[dict]) -> None:
    path = Path(dataset_root) / "meta" / SOURCES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump({"sources": sources}, f, sort_keys=False, default_flow_style=None, width=120)


def read_sources(dataset_root: Path) -> list[dict] | None:
    """Lista escrita por write_sources, o None si el dataset no la tiene (datasets antiguos)."""
    path = Path(dataset_root) / "meta" / SOURCES_FILE
    if not path.exists():
        return None
    return (yaml.safe_load(path.read_text()) or {}).get("sources") or []
