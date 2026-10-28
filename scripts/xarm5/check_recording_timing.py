"""Verifica que las trayectorias grabadas tengan un dt simulado fijo entre frames.

Uso: python scripts/xarm5/check_recording_timing.py [traj_folder]
     (por defecto traj_folder de configs/folder_configs.yaml)
"""
import glob
import os
import pickle
import sys

import numpy as np
import yaml

if __name__ == "__main__":
    if len(sys.argv) > 1:
        traj_folder = sys.argv[1]
    else:
        with open("configs/folder_configs.yaml") as f:
            traj_folder = yaml.safe_load(f)["traj_folder"]

    for episode in sorted(glob.glob(os.path.join(traj_folder, "[0-9][0-9][0-9]"))):
        pkls = sorted(glob.glob(os.path.join(episode, "*.pkl")))
        pngs = glob.glob(os.path.join(episode, "image_1", "*.png"))
        times = []
        for path in pkls:
            with open(path, "rb") as f:
                times.append(pickle.load(f).get("sim_time"))

        name = os.path.basename(episode)
        if None in times:
            print(f"{name}: {len(pkls)} frames, sin 'sim_time' (grabado antes del fix)")
            continue
        dts = np.round(np.diff(times), 6)
        ok = len(np.unique(dts)) <= 1
        print(f"{name}: {len(pkls)} frames, {len(pngs)} png, "
              f"duración {times[-1] - times[0]:.3f} s, "
              f"dt {np.unique(dts).tolist()} {'OK' if ok else 'IRREGULAR'}")
