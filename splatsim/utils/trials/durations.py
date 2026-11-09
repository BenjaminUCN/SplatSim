"""Duración (trial_seconds) de los ensayos de run_policy_trials.py.

Las corridas de un modelo quedan en <output_dir>/<modelo>/<fecha>/. Si se cambia trial_seconds entre
corridas, juntarlas mezcla ensayos con más o menos tiempo para llegar al tomate: run_policy_trials.py
avisa al lanzar y compare_trials.py las separa por duración. trials.csv trae la columna trial_seconds;
en las corridas anteriores a esa columna se lee del meta.json de cada ensayo.
"""
import csv
import json
from pathlib import Path


def trial_seconds(trial_folder) -> float | None:
    """trial_seconds del meta.json de un ensayo, o None si no está."""
    try:
        value = json.loads((Path(trial_folder) / "meta.json").read_text()).get("trial_seconds")
    except (OSError, ValueError):
        return None
    return float(value) if value is not None else None


def run_durations(run_dir) -> set[float]:
    """Duraciones de los ensayos de una corrida (<modelo>/<fecha>/)."""
    run_dir = Path(run_dir)
    durations = set()
    csv_path = run_dir / "trials.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                value = row.get("trial_seconds") or None
                value = float(value) if value else trial_seconds(row.get("folder") or "")
                if value is not None:
                    durations.add(value)
    if not durations:   # corrida sin filas en el CSV (interrumpida): de los meta.json
        durations = {s for d in sorted(run_dir.iterdir()) if d.is_dir()
                     if (s := trial_seconds(d)) is not None}
    return durations


def other_durations(model_dir, seconds: float, skip=None) -> dict[float, list[str]]:
    """Corridas del modelo con otra duración que `seconds`: {duración: [carpetas]}."""
    found: dict[float, list[str]] = {}
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return found
    for run in sorted(p for p in model_dir.iterdir() if p.is_dir() and p != Path(skip or "")):
        for s in run_durations(run):
            if abs(s - seconds) > 1e-6:
                found.setdefault(s, []).append(run.name)
    return found
