"""Episodios descartados de una carpeta de grabaciones (<carpeta>/selection.yaml).

Por defecto todos los episodios se usan; selection.yaml lista los descartados. Lo escribe
scripts/xarm5/review_recordings.py y lo respeta splatsim/utils/lerobot_transform_v2.py.
"""
from pathlib import Path

import yaml

SELECTION_FILE = "selection.yaml"

_HEADER = """\
# Episodes of this folder excluded from datasets (written by scripts/xarm5/review_recordings.py;
# splatsim/utils/lerobot_transform_v2.py skips them unless --ignore_selection).
"""


def load_discarded(folder) -> set[str]:
    path = Path(folder) / SELECTION_FILE
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text()) or {}
    return {str(e) for e in data.get("discarded") or []}


def save_discarded(folder, discarded) -> None:
    path = Path(folder) / SELECTION_FILE
    names = sorted(discarded, key=lambda n: (len(n), n))
    with open(path, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump({"discarded": names}, f, default_flow_style=None)
