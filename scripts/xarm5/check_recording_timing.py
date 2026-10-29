"""Verifica el dt entre frames de las trayectorias grabadas.

Sim: el dt simulado (sim_time) debe ser fijo. Robot real: el reloj (time) nunca es exacto, así que
muestra la media, desviación y extremos del dt contra el periodo 1/control_hz, y cuenta los frames
atrasados (dt > 1.5 periodos).

Uso: python scripts/xarm5/check_recording_timing.py [traj_folder ...]
     (por defecto traj_folder de configs/folder_configs.yaml). Una carpeta sin episodios
     (000/, 001/, ...) se recorre un nivel más: p.ej. la salida de replay_recordings.py
     (<fecha>/position/000, <fecha>/interp/000, ...).
"""
import glob
import os
import pickle
import sys

import numpy as np
from splatsim.utils.paths import load_yaml

LATE = 1.5   # periodos: un dt mayor cuenta como frame atrasado


def episodes_in(folder: str) -> list[str]:
    found = sorted(glob.glob(os.path.join(folder, "[0-9][0-9][0-9]")))
    if found:
        return found
    return sorted(glob.glob(os.path.join(folder, "*", "[0-9][0-9][0-9]")))


def check(episode: str, folder: str) -> None:
    pkls = sorted(glob.glob(os.path.join(episode, "*.pkl")))
    pngs = glob.glob(os.path.join(episode, "image_1", "*.png"))
    frames = []
    for path in pkls:
        with open(path, "rb") as f:
            frames.append(pickle.load(f))

    name = os.path.relpath(episode, folder)
    if len(frames) < 2:
        print(f"{name}: {len(frames)} frames")
        return
    head = f"{name}: {len(frames)} frames, {len(pngs)} png"
    if all("sim_time" in o for o in frames):
        times = [o["sim_time"] for o in frames]
        dts = np.round(np.diff(times), 6)
        ok = len(np.unique(dts)) <= 1
        print(f"{head}, duración {times[-1] - times[0]:.3f} s, "
              f"dt {np.unique(dts).tolist()} {'OK' if ok else 'IRREGULAR'}")
    elif all("time" in o for o in frames):
        dts = np.diff([o["time"] for o in frames]) * 1000
        hz = frames[0].get("control_hz")
        target = f" (objetivo {1000 / hz:.1f})" if hz else ""
        late = f", {int((dts > LATE * 1000 / hz).sum())} atrasados" if hz else ""
        print(f"{head}, duración {dts.sum() / 1000:.3f} s, dt ms{target}: media {dts.mean():.1f}, "
              f"desv {dts.std():.1f}, mín {dts.min():.1f}, máx {dts.max():.1f}{late}")
    else:
        print(f"{head}, sin 'sim_time' ni 'time' (grabado antes del fix)")


if __name__ == "__main__":
    folders = sys.argv[1:] or [load_yaml("configs/folder_configs.yaml")["traj_folder"]]
    for folder in folders:
        folder = os.path.expanduser(folder)
        episodes = episodes_in(folder)
        if not episodes:
            print(f"{folder}: sin episodios (000/, 001/, ...)")
        for episode in episodes:
            check(episode, folder)
