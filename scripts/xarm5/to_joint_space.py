"""Copia de un dataset LeRobot del xArm5 con el estado y la acción en articulaciones (5 + pinza).

Las imágenes no cambian: los videos se enlazan (hardlink) y solo se reescriben el estado y la acción de
data/*.parquet, sus nombres en meta/info.json y sus estadísticas (meta/stats.json y las de cada episodio en
meta/episodes/). Los valores salen de las grabaciones (bc_data) que lista meta/splatsim_sources.yaml.
Cada cuadro del dataset se busca en los .pkl de su episodio por su estado (pose de link_eef), así los
cuadros que la conversión descartó por imágenes faltantes no corren la correspondencia.

    python scripts/xarm5/to_joint_space.py \\
        --src ~/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261114_percep400_4cam \\
        --dst ~/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261114_percep400_4cam_joints

Con --dst_only_data escribe solo data/ y meta/ (sin videos): para llevar a otra máquina que ya tiene --src
y armar ahí la copia con `cp -al <src>/videos <dst>/`.
Ver splatsim/utils/data/action_space.py.
"""
import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from splatsim.utils.data.action_space import JOINT_NAMES, joint_state_action
from splatsim.utils.data.dataset_sources import read_sources

STATE, ACTION = "observation.state", "action"
QUANTILES = (1, 10, 50, 90, 99)
MATCH_TOL = 1e-5        # el estado del parquet es el float32 del .pkl


def pkl_files(folder: Path) -> list[Path]:
    def key(p: Path):
        try:
            return (0, int(p.stem), "")
        except ValueError:
            return (1, 0, p.stem)
    return sorted(folder.glob("*.pkl"), key=key)


def episode_sources(src: Path, recordings: Path) -> dict[int, Path]:
    """Índice de episodio del dataset -> carpeta del episodio grabado."""
    out = {}
    for entry in read_sources(src) or []:
        folder = recordings / entry["folder"]
        for ep, ds_ep in zip(entry["episodes"], entry["dataset_episodes"]):
            out[int(ds_ep)] = folder / str(ep)
    if not out:
        raise SystemExit(f"{src}/meta/splatsim_sources.yaml no lista episodios")
    return out


def match_episode(states: np.ndarray, folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Estado y acción en articulaciones de cada fila de `states` (pose, (n, 7)), en orden."""
    joint_states, joint_actions = [], []
    files = iter(pkl_files(folder))
    for i, target in enumerate(states):
        for path in files:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if np.abs(np.asarray(data["state"], dtype=np.float32).flatten() - target).max() <= MATCH_TOL:
                s, a = joint_state_action(data)
                joint_states.append(s)
                joint_actions.append(a)
                break
        else:
            raise SystemExit(f"{folder}: el cuadro {i} del dataset no está en los .pkl")
    return np.stack(joint_states), np.stack(joint_actions)


def stats(x: np.ndarray) -> dict[str, np.ndarray]:
    out = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    for q in QUANTILES:
        out[f"q{q:02d}"] = np.quantile(x, q / 100, axis=0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--recordings", type=Path, default=Path("~/data/bc_data"))
    p.add_argument("--dst_only_data", action="store_true", help="sin videos/ (ver arriba)")
    args = p.parse_args()
    src, dst, recordings = args.src.expanduser(), args.dst.expanduser(), args.recordings.expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} ya existe")
    info = json.loads((src / "meta/info.json").read_text())
    if info["features"][STATE].get("names") == JOINT_NAMES:
        raise SystemExit(f"{src} ya está en articulaciones")
    sources = episode_sources(src, recordings)

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src / "meta", tmp / "meta")
    if (src / "images").exists():
        shutil.copytree(src / "images", tmp / "images")
    if not args.dst_only_data and (src / "videos").exists():
        shutil.copytree(src / "videos", tmp / "videos", copy_function=os.link)

    per_episode: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for data_file in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(data_file)
        df = table.to_pandas()
        new_state = np.zeros((len(df), 6), dtype=np.float32)
        new_action = np.zeros((len(df), 6), dtype=np.float32)
        for ep, rows in df.groupby("episode_index").indices.items():
            rows = rows[np.argsort(df["frame_index"].to_numpy()[rows])]
            s, a = match_episode(np.stack(df[STATE].to_numpy()[rows]).astype(np.float32), sources[int(ep)])
            new_state[rows], new_action[rows] = s, a
            per_episode[int(ep)] = (s, a)
            print(f"episodio {ep}: {len(rows)} cuadros", flush=True)
        list6 = pa.list_(pa.float32(), 6)
        cols = {STATE: pa.array(list(new_state), type=list6), ACTION: pa.array(list(new_action), type=list6)}
        for name, col in cols.items():
            i = table.schema.get_field_index(name)
            table = table.set_column(i, pa.field(name, list6), col)
        meta = dict(table.schema.metadata or {})
        if b"huggingface" in meta:          # la copia de las features que guarda la librería datasets
            hf = json.loads(meta[b"huggingface"])
            for name in (STATE, ACTION):
                hf["info"]["features"][name]["length"] = 6
            hf.pop("fingerprint", None)
            meta[b"huggingface"] = json.dumps(hf).encode()
            table = table.replace_schema_metadata(meta)
        out = tmp / data_file.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

    for name in (STATE, ACTION):
        info["features"][name]["shape"] = [6]
        info["features"][name]["names"] = list(JOINT_NAMES)
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))

    all_states = np.concatenate([per_episode[e][0] for e in sorted(per_episode)])
    all_actions = np.concatenate([per_episode[e][1] for e in sorted(per_episode)])
    global_stats = json.loads((src / "meta/stats.json").read_text())
    for name, x in ((STATE, all_states), (ACTION, all_actions)):
        global_stats[name] = {k: v.tolist() for k, v in stats(x).items()}
    (tmp / "meta/stats.json").write_text(json.dumps(global_stats, indent=4))

    for ep_file in sorted((tmp / "meta/episodes").rglob("*.parquet")):
        ep_df = pd.read_parquet(ep_file)
        for name, idx in ((STATE, 0), (ACTION, 1)):
            values = [stats(per_episode[int(e)][idx]) for e in ep_df["episode_index"]]
            for k in values[0]:
                ep_df[f"stats/{name}/{k}"] = [v[k].astype(np.int64 if k == "count" else np.float64).tolist() for v in values]
        ep_df.to_parquet(ep_file, index=False)

    tmp.rename(dst)
    print(f"{dst}: {len(per_episode)} episodios, {len(all_states)} cuadros en articulaciones")


if __name__ == "__main__":
    main()
