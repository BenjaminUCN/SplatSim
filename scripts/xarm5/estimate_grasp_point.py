"""Estima el punto de agarre (tomate en el marco del efector) con el último frame de las demos.

    python scripts/xarm5/estimate_grasp_point.py --recordings ~/data/bc_data/gello_3plants

Usa el último .pkl legible de cada episodio (sin los descartados en selection.yaml), toma la
mediana y escribe configs/grasp_point.yaml, que usan run_policy_trials.py, compare_trials.py
y las etiquetas de parada de review_recordings.py.
"""
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raíz del repo
from splatsim.utils.episode_selection import load_discarded  # noqa: E402
from splatsim.utils.grasp_point import GRASP_POINT_PATH, tomato_in_ee  # noqa: E402
from splatsim.utils.paths import contract  # noqa: E402

APPLE_KEY = "plastic_apple_position"


@dataclass
class Args:
    recordings: tuple[str, ...] = ("~/data/bc_data/gello_3plants",)
    out: str = GRASP_POINT_PATH


def last_readable(episode: Path) -> dict | None:
    for p in sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem), reverse=True):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001  .pkl cortado al final de la grabación
            continue
    return None


def main(args: Args):
    rels, sources = [], []
    for folder in (Path(r).expanduser() for r in args.recordings):
        discarded = load_discarded(folder)
        episodes = [e for e in sorted(folder.iterdir()) if e.is_dir() and e.name not in discarded
                    and any(e.glob("*.pkl"))]
        n = 0
        for ep in episodes:
            obs = last_readable(ep)
            if obs is None or obs.get(APPLE_KEY) is None:
                continue
            rels.append(tomato_in_ee(obs["state"], obs[APPLE_KEY]))
            n += 1
        sources.append({"folder": contract(str(folder)), "episodes": n})
        print(f"{folder}: {n} episodios")
    if not rels:
        raise SystemExit("Ningún episodio con posición del tomate")
    rels = np.array(rels)
    point = np.median(rels, axis=0)
    err = np.linalg.norm(rels - point, axis=1)
    data = {
        "point": [round(float(v), 4) for v in point],
        "spread_std": [round(float(v), 4) for v in rels.std(axis=0)],
        "final_error_p50": round(float(np.median(err)), 4),
        "final_error_p90": round(float(np.percentile(err, 90)), 4),
        "sources": sources,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(args.out, "w") as f:
        f.write("# Tomato position in the end-effector frame at the end of the demos (m), and how far\n"
                "# each demo's last frame is from it. Written by scripts/xarm5/estimate_grasp_point.py.\n")
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    print(f"punto de agarre {data['point']} m; error del último frame p50 {data['final_error_p50']} m, "
          f"p90 {data['final_error_p90']} m → {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
