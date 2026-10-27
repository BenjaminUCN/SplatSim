"""Compara modelos a partir de sus ensayos (trials.csv de run_policy_trials.py): tasa de éxito
con intervalo de confianza, test estadístico, desglose por posición del tomate y gráficos.

    python scripts/compare_trials.py ~/data/policy_trials/2cam-h64 ~/data/policy_trials/1cam-h64
    python scripts/compare_trials.py base=~/data/policy_trials/A/261027_150541 new=~/data/policy_trials/B

Cada entrada es una carpeta de modelo (se juntan todas sus corridas <fecha>/trials.csv) o una
corrida; "etiqueta=ruta" cambia el nombre que aparece en tablas y gráficos.

Éxito: error de agarre mínimo <= --success_dist (4 cm) durante el ensayo, es decir, el tomate
llegó a estar en el punto de agarre del efector (configs/grasp_point.yaml). --metric elige
min_grasp_err (por defecto), final_grasp_err, o las distancias crudas efector–tomate min_dist /
final_dist (que no bajan de ~13 cm porque se miden desde el último link). Para ensayos antiguos
sin columnas de agarre, se calculan desde los .pkl grabados de cada ensayo. Tests: con 2 modelos, exacto de Fisher; con más, chi-cuadrado de
independencia y Fisher por pares con corrección de Holm. Los intervalos son de Wilson (95%).

Salida (--out, por defecto ~/data/policy_trials/comparisons/<fecha>/): summary.md,
summary.csv, per_position.csv, success_rate.png, success_by_position.png, distances.png.
"""
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splatsim.utils.grasp_point import SUCCESS_RADIUS, grasp_error, load_grasp_point  # noqa: E402

# paleta de referencia (modo claro): una serie = un color; magnitud = rampa azul de un tono
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
METRICS = {"min_grasp_err": "min grasp error", "final_grasp_err": "final grasp error",
           "min_dist": "min EE–tomato distance", "final_dist": "final EE–tomato distance"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass
class Args:
    runs: tyro.conf.Positional[tuple[str, ...]]   # carpetas de modelo o de corrida (etiqueta=ruta)
    success_dist: float = SUCCESS_RADIUS          # m: umbral de éxito
    metric: str = "min_grasp_err"                 # min_grasp_err, final_grasp_err, min_dist o final_dist
    out: str | None = None


# ── datos ────────────────────────────────────────────────────────────────────
def load_runs(entries) -> pd.DataFrame:
    frames = []
    for entry in entries:
        label, _, path = entry.rpartition("=") if "=" in entry else ("", "", entry)
        path = Path(os.path.expanduser(path))
        csvs = [path / "trials.csv"] if (path / "trials.csv").exists() else sorted(path.glob("*/trials.csv"))
        csvs = [c for c in csvs if c.stat().st_size > 0]
        if not csvs:
            raise SystemExit(f"No hay trials.csv en {path}")
        label = label or (path.name if not (path / "trials.csv").exists() or not path.name[:6].isdigit()
                          else f"{path.parent.name}/{path.name}")
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        if df.empty:
            raise SystemExit(f"{path}: trials.csv sin ensayos")
        if "min_grasp_err" not in df or df.min_grasp_err.isna().any():
            df = _backfill_grasp(df)
        df["model"] = label
        frames.append(df)
        print(f"{label}: {len(df)} ensayos de {len(csvs)} corrida(s)")
    return pd.concat(frames, ignore_index=True)


def _backfill_grasp(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el error de agarre de ensayos antiguos desde los .pkl de cada carpeta de ensayo."""
    point = load_grasp_point()
    finals, mins = [], []
    for folder in df.folder:
        errs = []
        for p in sorted(Path(folder).glob("*.pkl"), key=lambda p: (len(p.stem), p.stem)):
            try:
                with open(p, "rb") as f:
                    obs = pickle.load(f)
                errs.append(float(grasp_error(obs["state"], obs["plastic_apple_position"], point)))
            except Exception:  # noqa: BLE001  .pkl ilegible: se omite ese frame
                continue
        finals.append(errs[-1] if errs else np.nan)
        mins.append(min(errs) if errs else np.nan)
    print(f"  error de agarre calculado desde los .pkl de {len(df)} ensayos")
    return df.assign(final_grasp_err=finals, min_grasp_err=mins)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ── análisis ─────────────────────────────────────────────────────────────────
def summarize(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        d = df[df.model == m]
        k, n = int(d.success.sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append({"model": m, "trials": n, "successes": k, "success_rate": k / n,
                     "ci_low": lo, "ci_high": hi, "positions": d.pos_idx.nunique(),
                     "median_min_grasp_err": d.min_grasp_err.median(),
                     "median_final_grasp_err": d.final_grasp_err.median()})
    return pd.DataFrame(rows)


def tests(summary: pd.DataFrame) -> list[str]:
    table = np.array([[r.successes, r.trials - r.successes] for r in summary.itertuples()])
    lines = []
    if len(summary) < 2:
        return ["Un solo modelo: no hay comparación."]
    if table[:, 0].sum() in (0, table.sum()):
        return ["Todos los ensayos tienen el mismo resultado (todos éxito o todos fallo): "
                "no hay diferencia que testear. Revisa --success_dist."]
    if len(summary) == 2:
        _, p = stats.fisher_exact(table)
        lines.append(f"Fisher exacto ({summary.model[0]} vs {summary.model[1]}): p = {p:.4f}")
        return lines
    chi2, p, dof, _ = stats.chi2_contingency(table)
    lines.append(f"Chi-cuadrado de independencia (modelo × éxito): χ²({dof}) = {chi2:.2f}, p = {p:.4f}")
    pairs = list(combinations(range(len(summary)), 2))
    raw = [stats.fisher_exact(table[[i, j]])[1] for i, j in pairs]
    for (i, j), p_raw, p_adj in zip(pairs, raw, holm(raw)):
        lines.append(f"  {summary.model[i]} vs {summary.model[j]}: Fisher p = {p_raw:.4f}, Holm p = {p_adj:.4f}")
    return lines


# ── gráficos ─────────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.label.set_color(TEXT_2)
    ax.xaxis.label.set_color(TEXT_2)


def plot_success(summary: pd.DataFrame, out: Path, threshold_text: str):
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(summary) + 1.5), 3.6), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(summary))
    rate = summary.success_rate.to_numpy()
    err = np.vstack([rate - summary.ci_low, summary.ci_high - rate])
    ax.bar(x, rate, width=0.55, color=SERIES, zorder=2)
    ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1.2, capsize=4, zorder=3)
    for xi, r in zip(x, summary.itertuples()):
        ax.text(xi, r.ci_high + 0.03, f"{r.success_rate:.0%}\n{r.successes}/{r.trials}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(x, summary.model, rotation=20 if len(summary) > 3 else 0, ha="right" if len(summary) > 3 else "center")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("Success rate (95% Wilson CI)")
    ax.set_title(f"Success rate per model ({threshold_text})", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_by_position(df: pd.DataFrame, models: list[str], out: Path):
    positions = sorted(df.pos_idx.unique())
    rate = np.full((len(models), len(positions)), np.nan)
    text = [["" for _ in positions] for _ in models]
    for i, m in enumerate(models):
        for j, p in enumerate(positions):
            d = df[(df.model == m) & (df.pos_idx == p)]
            if len(d):
                rate[i, j] = d.success.mean()
                text[i][j] = f"{int(d.success.sum())}/{len(d)}"
    fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(positions) + 2.5), 0.5 * len(models) + 1.6),
                           facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(rate), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(positions)):
            if text[i][j]:
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                        color="white" if rate[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(len(positions)), [str(p) for p in positions])
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("Tomato grid position")
    ax.tick_params(which="both", colors=TEXT_2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(positions)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)   # separación de 2px entre celdas
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    cbar.ax.tick_params(colors=TEXT_2, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title("Success by tomato position (successes / trials)", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_distances(df: pd.DataFrame, models: list[str], metric: str, threshold: float, out: Path):
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(models) + 1.5), 3.6), facecolor=SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        v = df[df.model == m][metric].to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=SERIES, alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color=TEXT, linewidth=2, zorder=4)
    ax.axhline(threshold, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(models)), models, rotation=20 if len(models) > 3 else 0,
                  ha="right" if len(models) > 3 else "center")
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    label = "Minimum" if metric.startswith("min") else "Final"
    what = "grasp error" if "grasp" in metric else "EE–tomato distance"
    ax.set_ylabel(f"{label} {what} (m)")
    ax.set_title(f"{label} {what} per trial\nbar = median · dashed = success ≤ {threshold:.2f} m",
                 fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    cells = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


# ── main ─────────────────────────────────────────────────────────────────────
def main(args: Args):
    if args.metric not in METRICS:
        raise SystemExit(f"--metric debe ser uno de {list(METRICS)}")
    df = load_runs(args.runs)
    models = list(dict.fromkeys(df.model))
    df["success"] = df[args.metric] <= args.success_dist
    out = Path(os.path.expanduser(args.out or
               f"~/data/policy_trials/comparisons/{datetime.now().strftime('%y%m%d_%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize(df, models)
    test_lines = tests(summary)
    per_pos = (df.groupby(["model", "pos_idx"]).agg(trials=("success", "size"), successes=("success", "sum"),
                                                    median_min_grasp_err=("min_grasp_err", "median"))
               .reset_index())
    summary.to_csv(out / "summary.csv", index=False)
    per_pos.to_csv(out / "per_position.csv", index=False)

    threshold_text = f"{METRICS[args.metric]} ≤ {args.success_dist:.2f} m"
    plot_success(summary, out / "success_rate.png", threshold_text)
    plot_by_position(df, models, out / "success_by_position.png")
    plot_distances(df, models, args.metric, args.success_dist, out / "distances.png")

    table = summary.assign(success_rate=lambda s: s.success_rate.map("{:.0%}".format),
                           ci=lambda s: [f"{a:.0%}–{b:.0%}" for a, b in zip(s.ci_low, s.ci_high)])
    md = [f"# Trial comparison ({datetime.now():%Y-%m-%d %H:%M})", "",
          f"Success: {threshold_text}.", "",
          _markdown_table(table[["model", "trials", "successes", "success_rate", "ci", "positions",
                                 "median_min_grasp_err", "median_final_grasp_err"]]),
          "", "## Tests", "", *[f"- {line.strip()}" for line in test_lines],
          "", "## Sources", "", *[f"- `{e}`" for e in args.runs],
          "", "![success](success_rate.png)", "![by position](success_by_position.png)",
          "![distances](distances.png)"]
    (out / "summary.md").write_text("\n".join(md) + "\n")

    print()
    print(table[["model", "trials", "successes", "success_rate", "ci"]].to_string(index=False))
    print(*test_lines, sep="\n")
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
