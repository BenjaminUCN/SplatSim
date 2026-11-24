"""Detector de cierre: estima desde wrist_rgb si cerrar la cortadora ahora cortaría el tallo.

En los ensayos de la política el brazo pasa por poses de corte pero la orden de cerrar llega tarde: el oráculo
(--close_when_cuttable, que lee peduncle_cuttable de la simulación) corta mucho más. Este detector aprende esa
señal desde la imagen de la muñeca, que también existe en el robot real. Las etiquetas son gratis: cada paso de
cada ensayo guarda peduncle_cuttable.

    # 1. cuadros de wrist_rgb (reducidos) y etiquetas de los ensayos guardados
    python scripts/xarm5/close_detector.py cache --runs loc_noocc_cut_a4j_s1_30k_1118 ... --out cache.npz
    # 2. clasificador; --folds 4 entrena uno por grupo de posiciones (las de pos_idx % 4) y guarda las
    #    probabilidades fuera de grupo para replay
    python scripts/xarm5/close_detector.py train --cache cache.npz --out detector_dir [--folds 4]
    # 3. qué habría pasado en los ensayos guardados con la regla (exacto hasta que la regla dispara)
    python scripts/xarm5/close_detector.py replay --cache cache.npz --probs detector_dir/oof.npz --runs ...
"""
from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from splatsim.agents.close_detector import OPEN_MAX, SIZE, make_model, resize, to_tensor

TRIALS = Path.home() / "data/policy_trials"
NEAR_M = 0.06               # todos los cuadros con el filo a menos de esto del tallo; el resto, uno de cada FAR_EVERY
FAR_EVERY = 4
CLOSE_COMMAND = 0.5


@dataclass
class Args:
    mode: Literal["cache", "train", "replay"]
    runs: list[str] = field(default_factory=list)   # corridas en ~/data/policy_trials
    cache: str = ""
    out: str = ""
    probs: str = ""
    folds: int = 4
    epochs: int = 6
    batch_size: int = 128
    workers: int = 16
    device: str = "cuda"
    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95])
    consecutive: list[int] = field(default_factory=lambda: [1, 2, 3])


def wrist_frames(image_dir: Path) -> dict[int, np.ndarray]:
    """Cuadros de wrist_rgb por número de paso, empaquetados (wrist_rgb.mp4 + videos.json) o en PNG."""
    import imageio.v3 as iio
    video = image_dir / "wrist_rgb.mp4"
    if video.exists():
        numbers = json.loads((image_dir / "videos.json").read_text())["wrist_rgb"]
        return dict(zip(numbers, iio.imread(video)))
    return {int(p.stem.rsplit("_", 1)[1]): iio.imread(p) for p in sorted(image_dir.glob("wrist_rgb_*.png"))}


def trial_rows(job) -> dict | None:
    """Los cuadros que sirven de un ensayo y su serie completa de pasos (para replay)."""
    run, folder, pos, ok = job
    folder = Path(folder)
    steps = sorted(folder.glob("*.pkl"))
    if not steps:
        return None
    frames = wrist_frames(folder / "image_1")
    cuttable, cmd, grip, err, keep = [], [], [], [], []
    for i, f in enumerate(steps):
        d = pickle.load(open(f, "rb"))
        c = d.get("peduncle_cuttable")
        cuttable.append(-1 if c is None else int(bool(c)))
        cmd.append(float(d["joint_command"][5]))
        grip.append(float(d["gripper_position"][0]))
        e = d.get("peduncle_error")
        err.append(np.nan if e is None else float(e))
        number = int(f.stem)
        if c is not None and grip[-1] < OPEN_MAX and number in frames and (err[-1] < NEAR_M or i % FAR_EVERY == 0):
            keep.append(i)
    images = np.stack([resize(frames[int(steps[i].stem)]) for i in keep]) if keep else np.zeros((0, *SIZE, 3), np.uint8)
    return {"run": run, "folder": str(folder), "pos": int(pos), "ok": bool(ok), "keep": np.array(keep, int),
            "images": images, "cuttable": np.array(cuttable, np.int8), "cmd": np.array(cmd, np.float32),
            "grip": np.array(grip, np.float32), "err": np.array(err, np.float32)}


def build_cache(args: Args):
    import pandas as pd
    jobs = []
    for run in args.runs:
        for csv in sorted((TRIALS / run).glob("*/trials.csv")):
            t = pd.read_csv(csv)
            jobs += [(run, r.folder, r.pos_idx, bool(r.peduncle_cut) and bool(r.peduncle_held)) for r in t.itertuples()]
    print(f"{len(jobs)} ensayos de {len(args.runs)} corridas")
    trials = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(trial_rows, jobs, chunksize=4)):
            if r is not None:
                trials.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    images = np.concatenate([t["images"] for t in trials])
    trial_of = np.concatenate([np.full(len(t["keep"]), k) for k, t in enumerate(trials)])
    step_of = np.concatenate([t["keep"] for t in trials])
    labels = np.concatenate([t["cuttable"][t["keep"]] for t in trials])
    meta = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items() if k not in ("images", "keep")}
            for t in trials]
    np.savez(args.out, images=images, trial=trial_of, step=step_of, label=labels,
             pos=np.array([t["pos"] for t in trials])[trial_of], meta=json.dumps(meta))
    print(f"{args.out}: {len(images)} cuadros, {labels.mean():.1%} cortables, {len(trials)} ensayos")


def augment(x):
    """Brillo/contraste y un corrimiento de pocos píxeles (la cámara de la muñeca real no queda exacta)."""
    import torch
    n = x.shape[0]
    x = x * (1 + 0.2 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)) + 0.2 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)
    dy, dx = np.random.randint(-4, 5, 2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))


def fit(images, labels, args: Args, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = args.epochs * (len(images) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
    pos_weight = torch.tensor((1 - labels.mean()) / max(labels.mean(), 1e-3), device=args.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for epoch in range(args.epochs):
        order = np.random.permutation(len(images))
        total = 0.0
        for b in range(len(images) // args.batch_size):
            idx = np.sort(order[b * args.batch_size:(b + 1) * args.batch_size])
            x = augment(to_tensor(images[idx], args.device))
            y = torch.as_tensor(labels[idx], device=args.device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = loss_fn(model(x).squeeze(1).float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"    época {epoch + 1}: pérdida {total / max(1, len(images) // args.batch_size):.4f}", flush=True)
    return model.eval()


def predict(model, images, args: Args) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(images), 512):
            out.append(torch.sigmoid(model(to_tensor(images[b:b + 512], args.device)).squeeze(1).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def train(args: Args):
    data = np.load(args.cache)
    images, labels, pos = data["images"], data["label"].astype(np.float32), data["pos"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(images), np.nan, np.float32)
    for k in range(args.folds):
        test = pos % args.folds == k
        print(f"grupo {k}: posiciones {sorted(set(pos[test]))}, {(~test).sum()} cuadros para entrenar", flush=True)
        model = fit(images[~test], labels[~test], args, seed=k)
        oof[test] = predict(model, images[test], args)
        save_detector(model, out / f"fold{k}", args, held_out=sorted(int(v) for v in set(pos[test])))
        print(f"  AUC fuera de grupo: {auc(labels[test], oof[test]):.3f}", flush=True)
    np.savez(out / "oof.npz", prob=oof)
    print(f"AUC fuera de grupo, total: {auc(labels, oof):.3f}")
    print("modelo final con todos los cuadros", flush=True)
    model = fit(images, labels, args, seed=args.folds)
    save_detector(model, out, args)


def save_detector(model, folder: Path, args: Args, held_out: list[int] | None = None) -> None:
    """detector.pt y detector.json; held_out: posiciones de la grilla que no vio (para evaluarlo en ellas)."""
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "detector.pt")
    (folder / "detector.json").write_text(json.dumps({"size": SIZE, "open_max": OPEN_MAX, "cache": args.cache,
                                                      "held_out_positions": held_out}))


def first_fire(prob: np.ndarray, threshold: float, consecutive: int) -> int | None:
    run = 0
    for i, p in enumerate(prob):
        run = run + 1 if p >= threshold else 0
        if run >= consecutive:
            return i
    return None


def replay(args: Args):
    """Regla "cierra el detector o la política, lo que ocurra primero": hasta ese paso la trayectoria grabada es la
    misma, así que el resultado es el del ensayo si cierra primero la política, o peduncle_cuttable en el paso en
    que dispara el detector (aproximación: con hold_while_closing el brazo queda quieto mientras cierra)."""
    import pandas as pd
    data = np.load(args.cache)
    meta = json.loads(str(data["meta"]))
    prob = np.load(args.probs)["prob"]
    rows = []
    for t_idx, t in enumerate(meta):
        if args.runs and t["run"] not in args.runs:
            continue
        sel = data["trial"] == t_idx
        steps, p = data["step"][sel], prob[sel]
        n = len(t["cuttable"])
        series = np.zeros(n, np.float32)   # cuadros sin imagen guardada (lejos del tallo, uno de cada FAR_EVERY): 0
        series[steps] = np.nan_to_num(p)
        cmd = np.array(t["cmd"])
        closes = np.flatnonzero(cmd >= CLOSE_COMMAND)
        policy_close = int(closes[0]) if closes.size else None
        for th in args.thresholds:
            for k in args.consecutive:
                fire = first_fire(series, th, k)
                if fire is not None and (policy_close is None or fire < policy_close):
                    ok, who = t["cuttable"][fire] == 1, "detector"
                else:
                    ok, who = t["ok"], "política"
                rows.append({"run": t["run"], "pos": t["pos"], "threshold": th, "consecutive": k, "ok": bool(ok),
                             "who": who, "base_ok": t["ok"]})
    df = pd.DataFrame(rows)
    n = df.groupby(["threshold", "consecutive"]).size().iloc[0]
    summary = df.groupby(["threshold", "consecutive"]).agg(
        exito=("ok", "sum"), base=("base_ok", "sum"),
        cierra_detector=("who", lambda w: int((w == "detector").sum())))
    print(f"{n} ensayos; éxito = corta (cortado y sujeto aproximado) con la regla; base = la política sola")
    print(summary.to_string())


def main(args: Args):
    {"cache": build_cache, "train": train, "replay": replay}[args.mode](args)


if __name__ == "__main__":
    main(tyro.cli(Args))
