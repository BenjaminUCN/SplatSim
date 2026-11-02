"""Compara los modos de control a partir de reproducciones (summary.csv de replay_recordings.py).

    python scripts/xarm5/compare_replays.py ~/data/bc_data/real_demos_replay [--exclude 001]
    python scripts/xarm5/compare_replays.py <run1>/summary.csv <run2>/summary.csv --metric rms_err

Cada entrada es un summary.csv o una carpeta (se juntan todos sus <fecha>/summary.csv). Como la
misma trayectoria se reproduce en cada modo, la comparación es pareada: por métrica, solo entran
los episodios con todos los modos, Friedman entre todos los modos y Wilcoxon pareado (exacto) por
cada par, con corrección de Holm. Los errores se comparan en escala log (cambio relativo: un 10 %
menos vale lo mismo en una trayectoria lenta que en una rápida); el retraso, en segundos.

Por par también da el tamaño del efecto (d de Cohen pareado) y cuántas trayectorias harían falta
para detectarlo con potencia 80 % (t pareada con α corregido por Holm, más un 5 % para Wilcoxon):
con pocas trayectorias es una estimación gruesa, sirve para planificar la siguiente tanda.

Avisos: episodios con retraso en el tope de la búsqueda (MAX_LAG de replay_recordings.py: el brazo
no siguió, p.ej. un choque), y si hay efecto de orden (columna order: métrica según el modo fue 1°,
2° o 3°). Salida: tabla por pantalla y, con --out, summary.md y per_episode.csv.
"""
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_recordings import MAX_LAG  # noqa: E402

METRICS = {"rms_err": "RMS error (rad)", "max_err": "max error (rad)", "final_err": "final error (rad)",
           "lag_s": "lag (s)"}
LOG_METRICS = ("rms_err", "max_err")     # cambio relativo; final_err y lag pueden ser 0
ALPHA, POWER = 0.05, 0.8


@dataclass
class Args:
    inputs: tyro.conf.Positional[tuple[str, ...]]
    metric: tuple[str, ...] = ("rms_err", "max_err", "lag_s")
    exclude: tuple[str, ...] = ()
    """episodios de origen a dejar fuera (su número, p.ej. 001)"""
    out: str | None = None


def load(inputs) -> pd.DataFrame:
    frames = []
    for entry in inputs:
        path = Path(os.path.expanduser(entry))
        csvs = [path] if path.is_file() else sorted(path.glob("*/summary.csv")) + sorted(path.glob("summary.csv"))
        for c in csvs:
            if c.stat().st_size and len(df := pd.read_csv(c)):
                df["run_dir"] = str(c.parent)
                frames.append(df)
    if not frames:
        raise SystemExit("No hay summary.csv con filas en las entradas")
    df = pd.concat(frames, ignore_index=True)
    df["episode"] = df.source_episode.map(lambda s: os.path.basename(str(s).rstrip("/")))
    return df


def n_needed(d: float, comparisons: int) -> int | None:
    """Trayectorias para detectar d con t pareada (α/comparisons, potencia 80 %), +5 % Wilcoxon."""
    if not np.isfinite(d) or d == 0:
        return None
    alpha = ALPHA / comparisons
    for n in range(3, 2000):
        tc = stats.t.ppf(1 - alpha / 2, n - 1)
        if 1 - stats.nct.cdf(tc, n - 1, abs(d) * np.sqrt(n)) >= POWER:
            return int(np.ceil(n / 0.955))
    return None


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def analyze(df: pd.DataFrame, metric: str, modes: list[str]) -> tuple[list[str], pd.DataFrame]:
    table = df.pivot_table(index="episode", columns="mode", values=metric, aggfunc="mean")[modes].dropna()
    lines = [f"### {METRICS.get(metric, metric)} — {len(table)} trajectories with all modes"]
    if len(table) < 2:
        return lines + ["Not enough paired trajectories."], table
    use_log = metric in LOG_METRICS and (table > 0).all().all()
    values = np.log(table) if use_log else table
    med = table.median()
    lines.append("Median per mode: " + ", ".join(f"{m} {med[m]:.4f}" for m in modes))
    if len(modes) >= 3:
        fr = stats.friedmanchisquare(*[values[m] for m in modes])
        lines.append(f"Friedman (all modes): χ² = {fr.statistic:.2f}, p = {fr.pvalue:.4f}")
    pairs = list(itertools.combinations(modes, 2))
    rows = []
    for a, b in pairs:
        diff = values[a] - values[b]
        p = stats.wilcoxon(diff, method="exact").pvalue if (diff != 0).any() else 1.0
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.inf * np.sign(diff.mean())
        change = (f"{(np.exp(diff.mean()) - 1) * 100:+.1f} %" if use_log
                  else f"{diff.mean():+.4f}")
        wins = int((diff < 0).sum())
        rows.append([a, b, change, wins, len(diff), p, d])
    adjusted = holm([r[5] for r in rows])
    # con n trayectorias el menor p exacto de Wilcoxon (dos colas) es 2/2^n
    min_n = next(n for n in range(2, 50) if 2 / 2 ** n < ALPHA / len(pairs))
    lines += ["", "| pair | mean change (a vs b) | a lower in | Wilcoxon p | Holm p | d | trajectories for 80 % power |",
              "|---|---|---|---|---|---|---|"]
    for (a, b, change, wins, n, p, d), padj in zip(rows, adjusted):
        need = n_needed(d, len(pairs))
        need = max(need, min_n) if need else None
        lines.append(f"| {a} vs {b} | {change} | {wins}/{n} | {p:.4f} | {padj:.4f} | {d:+.2f} | "
                     f"{need if need else '—'} |")
    if len(table) < min_n:
        lines.append(f"\nWith {len(table)} trajectories no pair can reach p < {ALPHA / len(pairs):.4f} "
                     f"(Holm, {len(pairs)} pairs): at least {min_n} are needed even if one mode wins in all.")
    return lines, table


def main(args: Args):
    df = load(args.inputs)
    if args.exclude:
        drop = df.episode.isin(args.exclude)
        print(f"Excluidos: {sorted(set(df.episode[drop]))} ({int(drop.sum())} filas)")
        df = df[~drop]
    modes = [m for m in ("position", "interp", "velocity") if m in set(df["mode"])] + \
        sorted(set(df["mode"]) - {"position", "interp", "velocity"})
    out = [f"# Replay comparison — {df.episode.nunique()} trajectories, modes {', '.join(modes)}", ""]
    capped = df[df.lag_s >= MAX_LAG - 1e-9]
    if len(capped):
        out.append(f"**Warning**: lag at the search limit ({MAX_LAG} s) in "
                   + ", ".join(f"{e} ({m})" for e, m in zip(capped.episode, capped["mode"]))
                   + ": the arm did not follow (a collision?); consider --exclude.")
    if "order" in df and df.order.notna().any():
        by_order = df.groupby("order").rms_err.median()
        out.append("RMS error by position in the episode's sequence (order effect check): "
                   + ", ".join(f"{int(k)}° {v:.4f}" for k, v in by_order.items()))
    if "commanded" in df and df.commanded.nunique() > 1:
        out.append("**Note**: mixes trajectories with and without joint_command as reference.")
    tables = {}
    for metric in args.metric:
        lines, table = analyze(df, metric, modes)
        out += ["", *lines]
        tables[metric] = table
    text = "\n".join(out)
    print(text)
    if args.out:
        root = Path(os.path.expanduser(args.out))
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.md").write_text(text + "\n")
        pd.concat(tables, axis=1).to_csv(root / "per_episode.csv")
        print(f"\nResultados en {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
