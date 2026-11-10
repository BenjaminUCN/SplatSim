"""Escribe el punto de agarre (dónde debe quedar el tomate en el marco del efector) en
configs/grasp_point.yaml, que usan run_policy_trials.py, compare_trials.py, las etiquetas de parada
de review_recordings.py y plan_trajectories.py.

    python scripts/xarm5/estimate_grasp_point.py [--point gripper|demos] [--recordings ~/data/bc_data/gello_3plants]

--point gripper (por defecto): el centro del espacio entre las pinzas, desde la URDF del robot
(grasp_point.gripper_center, ≈ [0, 0, 0.129]). --point demos: la mediana de dónde quedó el tomate
en el último frame de las demos (≈ [-0.019, 0.003, 0.168] en gello_3plants, a la altura de la
punta de los dedos: se grabaron sin gripper). En los dos casos anota cómo terminan las demos
respecto del punto elegido (último .pkl legible de cada episodio, sin los descartados).
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
from splatsim.utils.data.episode_selection import load_discarded  # noqa: E402
from typing import Literal  # noqa: E402

from splatsim.utils.trials.grasp_point import (  # noqa: E402
    GRASP_POINT_PATH,
    POINT_HEADER,
    SUCCESS_RADIUS,
    gripper_center,
    tomato_in_ee,
)
from splatsim.utils.paths import OBJECTS_CONFIG, contract, expand, load_yaml  # noqa: E402

from splatsim.robots.xarm.config import APPLE_KEY  # noqa: E402


@dataclass
class Args:
    point: Literal["gripper", "demos"] = "gripper"
    """gripper: centro entre las pinzas (URDF); demos: mediana del final de las demos"""
    recordings: tuple[str, ...] = ("~/data/bc_data/gello_3plants",)
    robot_name: str | None = None
    """robot de configs/object_configs/objects.yaml cuya URDF da el centro de la pinza; por defecto
    xarm5_lite_plant_v3_hand_held si existe, si no xarm5_lite_plant_v3"""
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
    demo_point = np.median(rels, axis=0)
    if args.point == "gripper":
        robots = load_yaml(OBJECTS_CONFIG)
        name = args.robot_name or ("xarm5_lite_plant_v3_hand_held" if "xarm5_lite_plant_v3_hand_held" in robots
                                   else "xarm5_lite_plant_v3")
        urdf = expand(robots[name]["urdf_path"][0])
        point = gripper_center(urdf)
    else:
        point = demo_point
    err = np.linalg.norm(rels - point, axis=1)
    r = lambda v: [round(float(x), 4) + 0.0 for x in v]  # noqa: E731  (+0.0: sin -0.0)
    data = {
        "point": r(point),
        "point_source": "gripper_center" if args.point == "gripper" else "demos",
        "demo_point": r(demo_point),
        "demo_spread_std": r(rels.std(axis=0)),
        # último frame de las demos respecto de point
        "final_error_p50": round(float(np.median(err)), 4),
        "final_error_p90": round(float(np.percentile(err, 90)), 4),
        "final_within_success": round(float(np.mean(err <= SUCCESS_RADIUS)), 3),
        "sources": sources,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(args.out, "w") as f:
        f.write(POINT_HEADER)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    print(f"punto de agarre {data['point']} m ({data['point_source']}); último frame de las demos: "
          f"p50 {data['final_error_p50']} m, p90 {data['final_error_p90']} m, "
          f"{data['final_within_success']:.0%} a ≤ {SUCCESS_RADIUS} m → {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
