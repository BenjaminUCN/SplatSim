"""Compara datasets de entrenamiento: cuánto duran y cómo se mueven las demos, si se usa el gripper,
dónde está el tomate y cuántas demos caen cerca de cada punto de la grilla de evaluación.

    python scripts/xarm5/compare_datasets.py \\
        "3plants mirrored=splatsim_xarm5_lite_10fps_261105_1459@gello_3plants_mirrored" \\
        "benjamin=splatsim_xarm5_lite_10fps_261109_1803@gello_test_viser" \\
        --out docs/experiments/2026-11-10/datasets

Cada entrada es "etiqueta=dataset@grabaciones":
- dataset: carpeta LeRobot (nombre dentro de --datasets_root o ruta). De sus parquet salen la
  duración, el movimiento del efector (observation.state[:3], el último eslabón) y el gripper
  (action[6]).
- grabaciones: carpetas de ~/data/bc_data separadas por coma, de donde sale la posición del tomate de
  cada episodio (meta.json o el primer .pkl). Una carpeta espejada (mirror_recordings.py) ya trae el
  tomate espejado, así que se usa tal cual. Sin "@…", se usan las grabaciones de origen que anota
  configs/registry.yaml (build_registry.py); revísalas, porque el registro puede asociar un espejo a un
  dataset que no lo tiene.

Las posiciones del tomate se agrupan por episodios consecutivos a menos de --same_position (2 cm):
así se ve si la teleoperación repite varias demos por posición. Se excluyen los episodios
descartados en selection.yaml y los marcados con "valid": false.

Salida (--out, por defecto ~/data/policy_trials/comparisons/datasets_<fecha>/): summary.md,
metrics.json, datasets_duracion.png y datasets_tomate.png.
"""
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # raíz del repo
from splatsim.utils.data.episode_selection import load_discarded  # noqa: E402

# paleta de referencia (modo claro): categórica en orden fijo, una serie = un color
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class Args:
    datasets: tyro.conf.Positional[tuple[str, ...]]   # "etiqueta=dataset@grabacion1,grabacion2"
    datasets_root: str = "~/lerobot/splatsim_test_data"
    recordings_root: str = "~/data/bc_data"
    grid_path: str = "configs/eval_grid_4x4.yaml"     # grilla de evaluación (run_policy_trials.py)
    near_dist: float = 0.10       # m: una demo está "cerca" de un punto de la grilla
    idle_speed: float = 0.01      # m/s: por debajo, el efector se considera quieto
    gripper_range: float = 0.3    # el episodio "cierra el gripper" si action[6] varía más que esto
    same_position: float = 0.02   # m: episodios consecutivos con el tomate más cerca son la misma posición
    trial_seconds: float = 15.0   # duración de los ensayos de evaluación (línea de referencia)
    out: str | None = None


@dataclass
class Entry:
    label: str
    dataset: Path
    recordings: list[str]


def parse_entry(spec: str, args: Args) -> Entry:
    label, _, rest = spec.partition("=")
    if not rest:
        label, rest = Path(spec.split("@")[0]).name, spec
    name, _, recs = rest.partition("@")
    path = Path(name).expanduser()
    if not path.is_absolute() or not path.exists():
        path = Path(args.datasets_root).expanduser() / name
    if not (path / "meta" / "info.json").exists():
        raise SystemExit(f"{spec}: {path} no es un dataset LeRobot")
    recordings = [r for r in recs.split(",") if r] if recs else registry_recordings(path.name)
    if not recordings:
        raise SystemExit(f"{spec}: sin grabaciones (agrega @carpeta; el registro no las tiene)")
    return Entry(label, path, recordings)


def registry_recordings(dataset_name: str) -> list[str]:
    path = ROOT / "configs/registry.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    sources = ((data.get("datasets") or {}).get(dataset_name) or {}).get("sources") or []
    return list(dict.fromkeys(s["recording"] for s in sources if s.get("recording")))


# ── lectura ──────────────────────────────────────────────────────────────────
def _load_pkl(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001  hay pkl dañados en grabaciones antiguas
        return None


def recording_episodes(folder: Path) -> list[dict]:
    """Por episodio: posición inicial del tomate y distancia final último eslabón–tomate."""
    discarded = load_discarded(folder)
    out = []
    for ep in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()):
        if ep.name in discarded:
            continue
        apple = None
        meta = ep / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("valid") is False:
                continue
            apple = m.get("apple_position")
        pkls = sorted(ep.glob("*.pkl"), key=lambda p: (len(p.stem), p.stem))
        if len(pkls) < 2:
            continue
        first, last = _load_pkl(pkls[0]), _load_pkl(pkls[-1])
        if first is None or last is None:
            continue
        apple = apple if apple is not None else first.get("plastic_apple_position")
        if apple is None:
            continue
        final = None
        if last.get("state") is not None and last.get("plastic_apple_position") is not None:
            final = float(np.linalg.norm(np.asarray(last["state"])[:3] - np.asarray(last["plastic_apple_position"])))
        out.append({"episode": ep.name, "apple": np.asarray(apple, float), "final_dist": final})
    return out


def dataset_frames(path: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(path.glob("data/**/*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                   for f in files).sort_values(["episode_index", "frame_index"])
    return df, json.loads((path / "meta" / "info.json").read_text())


# ── métricas ─────────────────────────────────────────────────────────────────
def motion_metrics(df: pd.DataFrame, fps: float, args: Args) -> pd.DataFrame:
    rows = []
    for _, ep in df.groupby("episode_index"):
        pos = np.stack(ep["observation.state"].to_numpy())[:, :3]
        grip = np.stack(ep["action"].to_numpy())[:, 6]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        speed = step * fps
        moving = np.where(speed >= args.idle_speed)[0]
        rows.append({
            "duration": len(pos) / fps, "path": step.sum(), "net": float(np.linalg.norm(pos[-1] - pos[0])),
            "speed_moving": float(np.median(speed[moving])) if len(moving) else 0.0,
            "idle_frac": float((speed < args.idle_speed).mean()) if len(speed) else 1.0,
            "lead_idle": (moving[0] if len(moving) else len(speed)) / fps,
            "trail_idle": ((len(speed) - 1 - moving[-1]) if len(moving) else len(speed)) / fps,
            "grip_range": float(np.ptp(grip)),
        })
    return pd.DataFrame(rows)


def position_groups(apples: np.ndarray, tol: float) -> list[list[int]]:
    """Episodios consecutivos con el tomate a menos de tol del primero del grupo."""
    groups: list[list[int]] = []
    for i, a in enumerate(apples):
        if groups and np.linalg.norm(a - apples[groups[-1][0]]) < tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def pct(a, p) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def summarize(entry: Entry, grid: np.ndarray, args: Args) -> tuple[dict, np.ndarray, np.ndarray]:
    df, info = dataset_frames(entry.dataset)
    fps = info["fps"]
    mm = motion_metrics(df, fps, args)
    eps = [e for rec in entry.recordings
           for e in recording_episodes(Path(args.recordings_root).expanduser() / rec)]
    apples = np.array([e["apple"] for e in eps]) if eps else np.zeros((0, 3))
    groups = position_groups(apples, args.same_position)
    per_demo = [len(g) for g in groups for _ in g]           # tamaño del grupo visto desde cada demo
    dist = np.linalg.norm(apples[:, None, :] - grid[None], axis=2) if len(apples) else np.zeros((0, len(grid)))
    near = (dist < args.near_dist).sum(0)
    finals = [e["final_dist"] for e in eps if e["final_dist"] is not None]
    cams = [k.removeprefix("observation.images.") for k in info["features"] if k.startswith("observation.images.")]
    return {
        "label": entry.label, "dataset": entry.dataset.name, "recordings": entry.recordings,
        "fps": fps, "episodes": int(info["total_episodes"]), "frames": int(info["total_frames"]),
        "cameras": cams, "tomato_episodes": len(apples),
        "duration_s": [pct(mm.duration, 50), pct(mm.duration, 10), pct(mm.duration, 90)],
        "over_trial": float((mm.duration > args.trial_seconds).mean()),
        "path_m": pct(mm.path, 50), "net_m": pct(mm.net, 50), "speed_moving": pct(mm.speed_moving, 50),
        "idle_frac": pct(mm.idle_frac, 50), "lead_idle_s": pct(mm.lead_idle, 50), "trail_idle_s": pct(mm.trail_idle, 50),
        "grip_closes": float((mm.grip_range > args.gripper_range).mean()), "grip_range_median": pct(mm.grip_range, 50),
        "positions": len(groups), "demos_same_position": pct(per_demo, 50) if per_demo else float("nan"),
        "single_position_demos": sum(1 for g in groups if len(g) == 1),
        "tomato": {ax: [pct(apples[:, i], 50), pct(apples[:, i], 10), pct(apples[:, i], 90)] for i, ax in enumerate("xyz")},
        "grid_near_median": float(np.median(near)) if len(near) else 0.0, "grid_near_min": int(near.min()) if len(near) else 0,
        "grid_uncovered": int((near == 0).sum()), "grid_nearest_median": float(np.median(dist.min(0))) if len(apples) else float("nan"),
        "final_dist_median": float(np.median(finals)) if finals else None,
    }, apples, mm.duration.to_numpy()


# ── gráficos ─────────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def plot_durations(labels, durations, trial_seconds, out: Path):
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(labels) + 1.5), 3.4), facecolor=SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)
    for i, (label, d) in enumerate(zip(labels, durations)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=8, color=CATEGORICAL[i % 8], alpha=0.5,
                   linewidths=0, zorder=2)
        ax.hlines(np.median(d), i - 0.28, i + 0.28, color=TEXT, linewidth=2, zorder=3)
    ax.axhline(trial_seconds, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(labels) - 0.5, trial_seconds, f"ensayo de evaluación: {trial_seconds:g} s", ha="right",
            va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=12 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
    ax.set_ylabel("Duración de la demo (s)", fontsize=8.5, color=TEXT_2)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
    ax.set_ylim(0, None)
    ax.set_title("Duración de las demos (barra = mediana)", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_tomatoes(labels, apples, grid, out: Path):
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.4), facecolor=SURFACE,
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax, label, A) in enumerate(zip(axes[0], labels, apples)):
        _style(ax)
        if len(A):
            ax.scatter(A[:, 1], A[:, 2], s=9, color=CATEGORICAL[i % 8], alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(grid[:, 1], grid[:, 2], s=34, marker="s", facecolors="none", edgecolors=TEXT,
                   linewidths=1.2, zorder=3)
        ax.set_title(f"{label} (n = {len(A)})", fontsize=8.5, color=TEXT)
        ax.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
        ax.set_xlabel("y (m)", fontsize=8, color=TEXT_2)
    axes[0][0].set_ylabel("z (m)", fontsize=8, color=TEXT_2)
    x = np.median(grid[:, 0])
    fig.suptitle(f"Posición del tomate en las demos (puntos) y grilla de evaluación (cuadrados, x ≈ {x:.2f})",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ── informe ──────────────────────────────────────────────────────────────────
def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def markdown(results: list[dict], args: Args) -> str:
    r3 = lambda v: f"{v:.2f}"  # noqa: E731
    general = _table(
        ["Dataset", "fps", "Episodios", "Frames", "Cámaras", "Duración, mediana (p10–p90)", f"> {args.trial_seconds:g} s",
         "Velocidad al moverse", "Tiempo quieto"],
        [[f"{r['label']} (`{r['dataset']}`)", f"{r['fps']:g}", str(r["episodes"]), f"{r['frames']:,}", ", ".join(r["cameras"]),
          f"{r['duration_s'][0]:.1f} s ({r['duration_s'][1]:.1f}–{r['duration_s'][2]:.1f})", f"{r['over_trial']:.0%}",
          f"{r['speed_moving'] * 100:.1f} cm/s", f"{r['idle_frac']:.0%}"] for r in results])
    motion = _table(
        ["Dataset", "Recorrido", "Desplazamiento neto", "Quieto al inicio", "Quieto al final", "Cierran el gripper",
         "Rango del gripper, mediana"],
        [[r["label"], f"{r['path_m']:.2f} m", f"{r['net_m']:.2f} m", f"{r['lead_idle_s']:.1f} s", f"{r['trail_idle_s']:.1f} s",
          f"{r['grip_closes']:.0%}", f"{r['grip_range_median']:.3f}"] for r in results])
    tomato = _table(
        ["Dataset", "Grabaciones", "Episodios con tomate", "Posiciones distintas", "Con una sola demo",
         "Demos en la misma posición (mediana por demo)", "x (p10–p90)", "y (p10–p90)", "z (p10–p90)"],
        [[r["label"], ", ".join(r["recordings"]), str(r["tomato_episodes"]), str(r["positions"]), str(r["single_position_demos"]),
          f"{r['demos_same_position']:.0f}"] +
         [f"{r3(r['tomato'][a][0])} ({r3(r['tomato'][a][1])} a {r3(r['tomato'][a][2])})" for a in "xyz"] for r in results])
    grid = _table(
        ["Dataset", f"Demos a < {args.near_dist * 100:.0f} cm de cada punto, mediana", "Puntos sin ninguna demo cerca",
         "Demo más cercana, mediana", "Distancia final último eslabón–tomate"],
        [[r["label"], f"{r['grid_near_median']:.1f}", str(r["grid_uncovered"]), f"{r['grid_nearest_median']:.2f} m",
          f"{r['final_dist_median']:.3f} m" if r["final_dist_median"] is not None else "—"] for r in results])
    return "\n".join([
        f"# Comparación de datasets ({datetime.now():%Y-%m-%d %H:%M})", "",
        f"Grilla: `{args.grid_path}`. Quieto = efector a menos de {args.idle_speed * 100:.0f} cm/s. "
        f"Cierra el gripper = action[6] varía más de {args.gripper_range}. Misma posición = episodios "
        f"consecutivos con el tomate a menos de {args.same_position * 100:.0f} cm.", "",
        "## General", "", general, "", "![Duración de las demos](datasets_duracion.png)", "",
        "## Movimiento y gripper", "", motion, "",
        "## Posición del tomate", "", tomato, "",
        "## Cobertura de la grilla de evaluación", "", grid, "",
        "![Posición del tomate](datasets_tomate.png)", ""])


def main(args: Args):
    entries = [parse_entry(spec, args) for spec in args.datasets]
    grid = np.array(yaml.safe_load(open(Path(args.grid_path).expanduser()))["positions"], float)
    out = Path(args.out or f"~/data/policy_trials/comparisons/datasets_{datetime.now():%y%m%d_%H%M%S}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    results, apples, durations = [], [], []
    for entry in entries:
        r, a, d = summarize(entry, grid, args)
        results.append(r)
        apples.append(a)
        durations.append(d)
        print(f"[datasets] {entry.label}: {r['episodes']} episodios, duración mediana {r['duration_s'][0]:.1f} s, "
              f"{r['tomato_episodes']} con tomate ({', '.join(entry.recordings)})")
    labels = [e.label for e in entries]
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    plot_durations(labels, durations, args.trial_seconds, out / "datasets_duracion.png")
    plot_tomatoes(labels, apples, grid, out / "datasets_tomate.png")
    (out / "summary.md").write_text(markdown(results, args))
    print(f"[datasets] Resultados en {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
