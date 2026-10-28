"""Relaciona grabaciones (bc_data), datasets LeRobot y modelos entrenados en un registro YAML.

    python scripts/xarm5/build_registry.py
    python scripts/xarm5/build_registry.py --recordings ~/data/bc_data --datasets ~/lerobot/splatsim_test_data \\
        --models ~/lerobot/examples/tutorial/diffusion/outputs --out configs/registry.yaml

Cómo se relacionan:
  - modelo -> dataset: exacto. El preprocesador del modelo guarda las estadísticas de
    normalización del dataset (min/max de observation.state y n° de frames), que se
    comparan con meta/stats.json de cada dataset. Si ninguno coincide (dataset borrado o
    reconvertido), se propone el último dataset creado antes del entrenamiento con las
    mismas cámaras (dataset_match: date).
  - dataset -> grabaciones: por la secuencia de largos de episodio. Cada episodio del
    dataset guarda su largo y cada carpeta de bc_data tiene un .pkl por frame; se busca
    el desplazamiento que alinea ambas secuencias (un dataset puede juntar varias carpetas).
    Se admite que cada episodio del dataset tenga 0-2 frames menos (frames_dropped).

El script se puede volver a correr: conserva los campos manuales (name, description,
notes) de cada entrada del registro existente y recalcula el resto.
"""
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import yaml
from safetensors.numpy import load_file

MANUAL_FIELDS = ("name", "description", "notes")
MIN_MATCHED_EPISODES = 10    # episodios coincidentes mínimos para relacionar una grabación
MIN_MATCH_RATIO = 0.8        # fracción de episodios solapados que deben coincidir
MAX_DROPPED_FRAMES = 2       # algunas conversiones descartan 0-2 frames por episodio

_HEADER = """\
# Registro de grabaciones (bc_data), datasets LeRobot y modelos. Generado por
# scripts/xarm5/build_registry.py; vuelve a correrlo tras grabar, convertir o entrenar.
# Edita a mano solo name / description / notes: se conservan al regenerar.
#   name: nombre corto para elegir el modelo (vacío -> se usa suggested_name)
#   description: qué se intentó mejorar (p.ej. "1 cámara, más pasos")
"""


@dataclass
class Args:
    recordings: str = "~/data/bc_data"
    datasets: str = "~/lerobot/splatsim_test_data"
    models: str = "~/lerobot/examples/tutorial/diffusion/outputs"
    out: str = "configs/registry.yaml"


# ── utilidades ───────────────────────────────────────────────────────────────
def _mtime(path) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path))


def _fmt(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else None


def _date_from_name(name: str, year: int) -> datetime | None:
    """Fecha del sufijo del nombre: ..._YYMMDD_HHMM o ..._MMDD_HHMM (año de `year`)."""
    if m := re.search(r"(\d{6})_(\d{4})$", name):
        return datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M")
    if m := re.search(r"(\d{4})_(\d{4})$", name):
        try:
            return datetime.strptime(f"{year}{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
        except ValueError:
            return None
    return None


def _ranges(indices: list[int]) -> str:
    """[0,1,2,5,6] -> '0-2,5-6'."""
    out, start = [], None
    for i, v in enumerate(indices):
        if start is None:
            start = v
        if i + 1 == len(indices) or indices[i + 1] != v + 1:
            out.append(f"{start}-{v}" if v != start else f"{v}")
            start = None
    return ",".join(out)


# ── grabaciones ──────────────────────────────────────────────────────────────
_IMG = re.compile(r"^(.+_rgb)_(\d+)\.png$")


def scan_recordings(root: Path) -> dict:
    recordings = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        episodes = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()),
                          key=lambda p: int(p.name))
        if not episodes:
            continue
        lengths = [sum(1 for f in os.scandir(ep) if f.name.endswith(".pkl")) for ep in episodes]
        recordings[folder.name] = {
            "path": str(folder),
            "episodes": len(episodes),
            "frames": int(sum(lengths)),
            **_recording_images(episodes[0]),
            "first_episode": _fmt(_mtime(episodes[0])),
            "last_episode": _fmt(_mtime(episodes[-1])),
            "_lengths": lengths,
            "_episode_names": [int(ep.name) for ep in episodes],
            "_hashes": [_first_pkl_hash(ep) for ep in episodes],
        }
    _find_shared_episodes(recordings)
    return recordings


def _first_pkl_hash(episode: Path) -> str | None:
    pkls = sorted(episode.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem))
    return hashlib.md5(pkls[0].read_bytes()).hexdigest() if pkls else None


def _find_shared_episodes(recordings: dict) -> None:
    """Episodios copiados entre carpetas (mismo primer .pkl): shared_episodes."""
    owners = {}
    for name, rec in recordings.items():
        for ep, h in zip(rec["_episode_names"], rec["_hashes"]):
            if h:
                owners.setdefault(h, []).append((name, ep))
    for name, rec in recordings.items():
        shared = {}
        for ep, h in zip(rec["_episode_names"], rec["_hashes"]):
            for other, _ in owners.get(h, []):
                if other != name:
                    shared.setdefault(other, []).append(ep)
        rec["shared_episodes"] = {o: _ranges(eps) for o, eps in shared.items()}


def _recording_images(episode: Path) -> dict:
    """Cámaras y numeración de imágenes del episodio (misma lógica que lerobot_conversion)."""
    for name in ("image_1", "images"):
        image_dir = episode / name
        if image_dir.is_dir():
            break
    else:
        return {"cameras": [], "image_numbering": None}
    cams = sorted({m.group(1) for f in os.scandir(image_dir) if (m := _IMG.match(f.name))},
                  key=lambda k: (k.startswith("wrist"), k != "base_rgb", k))
    pkls = sorted((p.stem for p in episode.glob("*.pkl")), key=lambda s: (len(s), s))[:5]
    per_frame = bool(pkls and cams) and all(
        (image_dir / f"{c}_{s}.png").exists() for s in pkls for c in cams)
    return {"cameras": cams, "image_numbering": "per_frame" if per_frame else "interleaved"}


# ── datasets ─────────────────────────────────────────────────────────────────
def scan_datasets(root: Path) -> dict:
    datasets = {}
    for info_path in sorted(root.glob("**/meta/info.json")):
        ds_dir = info_path.parent.parent
        key = str(ds_dir.relative_to(root))
        info = json.loads(info_path.read_text())
        feats = info["features"]
        images = [k for k in feats if k.startswith("observation.images.")]
        stats_path = ds_dir / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        created = _date_from_name(ds_dir.name, _mtime(info_path).year) or _mtime(info_path)
        datasets[key] = {
            "path": str(ds_dir),
            "created": _fmt(created),
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "fps": info["fps"],
            "robot_type": info.get("robot_type"),
            "cameras": [k.removeprefix("observation.images.") for k in images],
            "image_shape": list(feats[images[0]]["shape"]) if images else None,
            "state_dim": feats["observation.state"]["shape"][0],
            "_created": created,
            "_lengths": _episode_lengths(ds_dir),
            "_state_stats": stats.get("observation.state"),
        }
    return datasets


def _episode_lengths(ds_dir: Path) -> list[int]:
    files = sorted(glob.glob(str(ds_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"]) for f in files])
    return df.sort_values("episode_index")["length"].astype(int).tolist()


def match_sources(ds_lengths: list[int], recordings: dict) -> list[dict]:
    """Segmentos de grabaciones cuya secuencia de largos coincide con la del dataset."""
    sources = []
    L = np.asarray(ds_lengths)
    for name, rec in recordings.items():
        R = np.asarray(rec["_lengths"])
        best = None
        for shift in range(-(len(R) - 1), len(L)):        # dataset_ep = recording_ep + shift
            r0, r1 = max(0, -shift), min(len(R), len(L) - shift)
            if r1 - r0 < MIN_MATCHED_EPISODES:
                continue
            diff = R[r0:r1] - L[r0 + shift:r1 + shift]    # frames que perdió cada episodio
            hits = np.nonzero((diff >= 0) & (diff <= MAX_DROPPED_FRAMES))[0] + r0
            if len(hits) >= MIN_MATCHED_EPISODES and (best is None or len(hits) > len(best[1])):
                best = (shift, hits)
        if best is None:
            continue
        shift, hits = best
        hits = _longest_run(hits)
        dropped = int(sum(R[i] - L[i + shift] for i in hits))
        span = hits[-1] - hits[0] + 1
        if len(hits) < MIN_MATCHED_EPISODES or len(hits) < MIN_MATCH_RATIO * span:
            continue
        rec_eps = list(range(hits[0], hits[-1] + 1))
        sources.append({
            "recording": name,
            "recording_episodes": _ranges([rec["_episode_names"][i] for i in rec_eps]),
            "dataset_episodes": _ranges([i + shift for i in rec_eps]),
            "matched_episodes": f"{len(hits)}/{span}",
            "frames_dropped": dropped,
        })
    return sorted(sources, key=lambda s: int(s["dataset_episodes"].split("-")[0].split(",")[0]))


def _longest_run(hits: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Tramo más largo de coincidencias con huecos de a lo más max_gap episodios; descarta
    coincidencias sueltas (largos iguales por azar) lejos del tramo real."""
    runs = np.split(hits, np.nonzero(np.diff(hits) > max_gap)[0] + 1)
    return max(runs, key=len)


# ── modelos ──────────────────────────────────────────────────────────────────
def scan_models(root: Path) -> dict:
    models = {}
    for d in sorted(p for p in root.glob("*/*") if p.is_dir()):
        policy_dir, pre_dir = _model_files(d)
        if policy_dir is None:
            continue
        cfg = json.loads((policy_dir / "config.json").read_text())
        ckpts = sorted((d / "checkpoints").glob("step_*")) if (d / "checkpoints").is_dir() else []
        steps = [int(re.match(r"step_(\d+)", c.name).group(1)) for c in ckpts]
        weights = policy_dir / "model.safetensors"
        finished = (d / "model.safetensors").exists()
        started = _date_from_name(d.name, _mtime(weights).year)
        train_cfg = d / "train_config.yaml"
        models[str(d.relative_to(root))] = {
            "path": str(d),
            "started": _fmt(started),
            "saved": _fmt(_mtime(weights)),
            "finished": finished,
            "policy": cfg.get("type"),
            "cameras": [k.removeprefix("observation.images.")
                        for k in cfg.get("input_features", {}) if k.startswith("observation.images.")],
            "n_obs_steps": cfg.get("n_obs_steps"),
            "horizon": cfg.get("horizon"),
            "n_action_steps": cfg.get("n_action_steps"),
            "resize_shape": cfg.get("resize_shape"),
            "crop_shape": cfg.get("crop_shape"),
            "vision_backbone": cfg.get("vision_backbone"),
            "checkpoint_steps": steps,
            "train_config": yaml.safe_load(train_cfg.read_text()) if train_cfg.exists() else None,
            "_started": started or _mtime(weights),
            "_state_stats": _normalizer_stats(pre_dir),
        }
    return models


def _model_files(d: Path):
    """(carpeta con config.json + pesos, carpeta del preprocesador): la final o el último checkpoint."""
    if (d / "config.json").exists() and (d / "model.safetensors").exists():
        return d, d
    ckpts = sorted((d / "checkpoints").glob("step_*")) if (d / "checkpoints").is_dir() else []
    for c in reversed(ckpts):
        if (c / "policy" / "config.json").exists():
            return c / "policy", c / "preprocessor"
    return None, None


def _normalizer_stats(pre_dir: Path) -> dict | None:
    files = glob.glob(str(pre_dir / "*normalizer_processor.safetensors"))
    if not files:
        return None
    t = load_file(files[0])
    if "observation.state.min" not in t:
        return None
    return {k: t[f"observation.state.{k}"] for k in ("min", "max", "count") if f"observation.state.{k}" in t}


def match_dataset(model: dict, datasets: dict) -> tuple[str | None, str | None, list[str]]:
    """(dataset, cómo se relacionó, otros datasets con las mismas estadísticas)."""
    ms = model["_state_stats"]
    exact = []
    if ms is not None:
        for key, ds in datasets.items():
            st = ds["_state_stats"]
            if not st or len(st["min"]) != len(ms["min"]):
                continue
            same = (np.allclose(st["min"], ms["min"], atol=1e-5)
                    and np.allclose(st["max"], ms["max"], atol=1e-5)
                    and ("count" not in ms or "count" not in st
                         or int(np.ravel(st["count"])[0]) == int(ms["count"][0])))
            if same:
                exact.append(key)
    if exact:
        # copias idénticas: la creada más cerca antes del entrenamiento
        exact.sort(key=lambda k: _distance_before(datasets[k]["_created"], model["_started"]))
        return exact[0], "stats", exact[1:]

    candidates = [k for k, ds in datasets.items()
                  if ds["cameras"] and set(model["cameras"]) <= set(ds["cameras"])
                  and ds["_created"] <= model["_started"] and ds["episodes"] > 0]
    if candidates:
        return max(candidates, key=lambda k: datasets[k]["_created"]), "date", []
    return None, None, []


def _distance_before(created: datetime, started: datetime) -> float:
    delta = (started - created).total_seconds()
    return delta if delta >= 0 else float("inf")


# ── registro ─────────────────────────────────────────────────────────────────
def _public(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _with_manual(section: dict, old: dict) -> dict:
    """Pone primero los campos manuales y luego el resto. Un campo manual conserva lo
    escrito antes en el registro; si está vacío, toma el valor del train_config del modelo."""
    out = {}
    for key, entry in section.items():
        prev = old.get(key) or {}
        train_cfg = entry.get("train_config") or {}
        manual = {f: prev.get(f) or train_cfg.get(f) or "" for f in MANUAL_FIELDS}
        out[key] = {**manual, **_public(entry)}
    return out


def suggest_name(model: dict, datasets: dict) -> str:
    """Nombre corto a partir de la metadata: <n>cam-<grabación>-<episodios>ep-h<horizon>-<MMDD>."""
    parts = [f"{len(model['cameras'])}cam"]
    ds = datasets.get(model["dataset"]) if model["dataset"] else None
    if ds:
        sources = [s["recording"].removeprefix("gello_").strip("_") or s["recording"]
                   for s in ds["sources"]]
        parts += ["+".join(dict.fromkeys(sources)) or "unknown", f"{ds['episodes']}ep"]
    parts.append(f"h{model['horizon']}")
    if model["_started"]:
        parts.append(model["_started"].strftime("%m%d"))
    return "-".join(parts)


def main(args: Args) -> None:
    rec_root, ds_root, model_root = (Path(os.path.expanduser(p))
                                     for p in (args.recordings, args.datasets, args.models))
    print(f"Grabaciones: {rec_root}")
    recordings = scan_recordings(rec_root)
    print(f"  {len(recordings)} carpetas")
    print(f"Datasets: {ds_root}")
    datasets = scan_datasets(ds_root)
    print(f"  {len(datasets)} datasets")
    print(f"Modelos: {model_root}")
    models = scan_models(model_root)
    print(f"  {len(models)} modelos")

    for ds in datasets.values():
        ds["sources"] = match_sources(ds["_lengths"], recordings)
    for m in models.values():
        m["dataset"], m["dataset_match"], m["same_stats_datasets"] = match_dataset(m, datasets)
        m["suggested_name"] = suggest_name(m, datasets)

    # relaciones inversas, para leer el registro desde cualquier punto
    for key, ds in datasets.items():
        ds["models"] = [k for k, m in models.items() if m["dataset"] == key]
    for key, rec in recordings.items():
        rec["datasets"] = [k for k, ds in datasets.items()
                           if any(s["recording"] == key for s in ds["sources"])]

    try:
        with open(args.out) as f:
            old = yaml.safe_load(f) or {}
    except FileNotFoundError:
        old = {}

    registry = {
        "generated": _fmt(datetime.now()),
        "roots": {"recordings": str(rec_root), "datasets": str(ds_root), "models": str(model_root)},
        "recordings": _with_manual(recordings, old.get("recordings", {})),
        "datasets": _with_manual(datasets, old.get("datasets", {})),
        "models": _with_manual(models, old.get("models", {})),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump(registry, f, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)

    _print_summary(recordings, datasets, models)
    print(f"\nRegistro escrito en {args.out}")


def _print_summary(recordings, datasets, models) -> None:
    print("\nModelo -> dataset -> grabaciones")
    for key, m in models.items():
        ds = datasets.get(m["dataset"]) if m["dataset"] else None
        srcs = ", ".join(f"{s['recording']}[{s['recording_episodes']}]" for s in ds["sources"]) if ds else "-"
        print(f"  {key:45s} {m['dataset'] or '?':45s} ({m['dataset_match'] or '-':5s}) {srcs or '?'}")
    orphans = [k for k, ds in datasets.items() if not ds["sources"] and ds["episodes"] > 0]
    if orphans:
        print(f"\nDatasets sin grabación encontrada: {', '.join(orphans)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
