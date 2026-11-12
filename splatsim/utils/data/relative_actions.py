"""Acciones relativas al estado para la Diffusion Policy (train_diffusion.py --relative_actions).

Las acciones del dataset son la pose absoluta del efector [x, y, z, roll, pitch, yaw, gripper]. Con
acciones relativas, cada chunk se entrena como la pose objetivo menos la pose del estado en el último
paso observado (el instante en que se predice el chunk); los ángulos se envuelven a [-pi, pi) y el
gripper (índice 6) queda absoluto. Al evaluar, el agente suma esa misma pose de referencia
(splatsim/agents/lerobot_diffusion_agent.py).

Las estadísticas de normalización de la acción se recalculan sobre los deltas que ve el entrenamiento
(cada frame de referencia con cada desplazamiento del chunk, recortado dentro del episodio como hace
LeRobot) y se guardan en el modelo como relative_action_stats.json: el agente las usa en vez de las
del dataset.
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

RELATIVE_DIMS = 6          # pos (3) + euler (3); el gripper queda absoluto
ANGLE_DIMS = slice(3, 6)
STATS_FILE = "relative_action_stats.json"


def wrap(a):
    """Ángulos a [-pi, pi) (numpy o torch)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_relative(action: torch.Tensor, state: torch.Tensor, ref_step: int) -> torch.Tensor:
    """action (B, H, D) absoluta, state (B, n_obs, D). Devuelve la acción relativa al estado ref_step."""
    ref = state[:, ref_step, :RELATIVE_DIMS].unsqueeze(1)
    out = action.clone()
    out[..., :RELATIVE_DIMS] = action[..., :RELATIVE_DIMS] - ref
    out[..., ANGLE_DIMS] = wrap(out[..., ANGLE_DIMS])
    return out


def relative_action_stats(dataset_root: str, action_offsets: list[int], base: dict) -> dict:
    """Estadísticas de la acción relativa: las de base (stats["action"] del dataset) con las dimensiones
    0-5 reemplazadas. action_offsets: los action_delta_indices de la política, relativos al frame de
    referencia (el último observado)."""
    files = sorted(glob.glob(str(Path(dataset_root).expanduser() / "data" / "**" / "*.parquet"), recursive=True))
    import pandas as pd
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"])
                    for f in files]).sort_values(["episode_index", "frame_index"])
    deltas = []
    for _, ep in df.groupby("episode_index"):
        actions = np.stack(ep["action"].to_numpy())[:, :RELATIVE_DIMS]
        states = np.stack(ep["observation.state"].to_numpy())[:, :RELATIVE_DIMS]
        t = np.arange(len(ep))
        for offset in action_offsets:
            d = actions[np.clip(t + offset, 0, len(ep) - 1)] - states
            d[:, ANGLE_DIMS] = wrap(d[:, ANGLE_DIMS])
            deltas.append(d)
    deltas = np.concatenate(deltas)
    stats = {k: np.array(v, dtype=np.float64) for k, v in base.items()}
    new = {"min": deltas.min(0), "max": deltas.max(0), "mean": deltas.mean(0), "std": deltas.std(0)}
    for q in ("q01", "q10", "q50", "q90", "q99"):
        new[q] = np.quantile(deltas, int(q[1:]) / 100, axis=0)
    for key, value in new.items():
        if key in stats and stats[key].shape and len(stats[key]) > RELATIVE_DIMS:
            stats[key][:RELATIVE_DIMS] = value
    return stats


def save_stats(stats: dict, folder) -> None:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}
    (Path(folder) / STATS_FILE).write_text(json.dumps(out, indent=1))


def load_stats(folder) -> dict | None:
    path = Path(folder) / STATS_FILE
    if not path.is_file():
        return None
    return {k: np.array(v, dtype=np.float32) for k, v in json.loads(path.read_text()).items()}
