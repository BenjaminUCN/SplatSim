"""trials.csv de una corrida de run_policy_trials.py, para revisar sus ensayos en review_recordings.py:
una fila por episodio (carpeta 000, 001, ...), éxito y filtros por resultado, oclusión y posición."""
import csv
import math
from pathlib import Path

from splatsim.utils.trials.grasp_point import SUCCESS_RADIUS

TRIALS_FILE = "trials.csv"
ALL = "all"
RESULTS = (ALL, "success", "fail")


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def is_success(row: dict) -> bool | None:
    """Lo que marcó el operador (robot real) o error mínimo de agarre ≤ SUCCESS_RADIUS, como
    compare_trials.py. None si la fila no tiene ninguno de los dos."""
    manual = (row.get("manual_success") or "").strip().lower()
    if manual in ("1", "true", "success"):
        return True
    if manual in ("0", "false", "fail"):
        return False
    err = _number(row.get("min_grasp_err"))
    return None if err is None else err <= SUCCESS_RADIUS


def load_trials(path: Path) -> dict[str, dict]:
    """Episodio (nombre de su carpeta) -> fila de trials.csv, con "success" agregado. {} si la
    carpeta no es una corrida de ensayos."""
    file = Path(path) / TRIALS_FILE
    if not file.exists():
        return {}
    with open(file, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        if row.get("folder"):
            out[Path(row["folder"]).name] = {**row, "success": is_success(row)}
    return out


def level_text(value) -> str:
    """'25' para 25.0; '' sin valor."""
    x = _number(value)
    return "" if x is None else f"{x:g}"


def options(rows: dict[str, dict], column: str) -> list[str]:
    """Valores distintos de una columna (ordenados como números si lo son), con "all" adelante."""
    values = {level_text(r.get(column)) if _number(r.get(column)) is not None else (r.get(column) or "")
              for r in rows.values()}
    values.discard("")
    return [ALL, *sorted(values, key=lambda v: (_number(v) is None, _number(v) or 0.0, v))]


def matches(row: dict | None, result: str = ALL, occlusion: str = ALL, z: str = ALL, position: str = ALL) -> bool:
    """Si el episodio pasa los filtros ("all" no filtra). Sin fila, solo pasa si no se filtra nada."""
    if row is None:
        return (result, occlusion, z, position) == (ALL, ALL, ALL, ALL)
    if result != ALL and row["success"] is not (result == "success"):
        return False
    if occlusion != ALL and level_text(row.get("occlusion_target")) != occlusion:
        return False
    if z != ALL and level_text(row.get("z")) != z:
        return False
    return position == ALL or (row.get("pos_idx") or "") == position


def short_label(row: dict) -> str:
    """Para la lista de episodios: ✓/✗, error mínimo y oclusión."""
    mark = {True: "✓", False: "✗"}.get(row["success"], "?")
    err = _number(row.get("min_grasp_err"))
    parts = [mark + (f" {err * 100:.1f} cm" if err is not None else "")]
    if level_text(row.get("occlusion_target")):
        parts.append(f"occ {level_text(row['occlusion_target'])} %")
    parts.append(f"pos {row.get('pos_idx', '?')}")
    return " · ".join(parts)


def details(row: dict) -> list[str]:
    """Líneas de markdown con el resultado del ensayo."""
    def cm(column):
        x = _number(row.get(column))
        return "–" if x is None else f"{x * 100:.1f} cm"

    result = {True: "**success**", False: "**fail**"}.get(row["success"], "no result")
    lines = [f"Trial: {result} — grasp error min {cm('min_grasp_err')}, final {cm('final_grasp_err')}"]
    where = f"position {row.get('pos_idx', '?')} (x {row.get('x')}, y {row.get('y')}, z {row.get('z')}), trial {row.get('trial')}"
    t = _number(row.get("time_to_success"))
    lines.append(where + (f", success at {t:.1f} s" if t is not None else ""))
    if level_text(row.get("occlusion_target")):
        leaf = f"leaf `{row.get('leaf') or '-'}` in {row.get('leaf_camera') or '-'}: {row.get('leaf_status')}"
        fit = level_text(row.get("occlusion_fit"))
        lines.append(f"occlusion target {level_text(row['occlusion_target'])} %"
                     + (f", fitted {fit} %" if fit else "") + f" — {leaf}")
    seen = [f"{c.removeprefix('occlusion_')} {level_text(v)} %" for c, v in row.items()
            if c.startswith("occlusion_") and c.endswith("_rgb") and level_text(v)]
    if seen:
        lines.append("tomato hidden at start: " + ", ".join(seen))
    for c in row:   # --track_occlusion: track_<cámara>_mean, _max, _pre_success, _in_view
        if c.startswith("track_") and c.endswith("_mean") and level_text(row[c]):
            cam = c.removeprefix("track_").removesuffix("_mean")
            v = {k: level_text(row.get(f"track_{cam}_{k}")) or "–" for k in ("mean", "max", "pre_success", "in_view")}
            lines.append(f"tomato hidden during the trial in {cam}: mean {v['mean']} %, max {v['max']} %, "
                         f"before success {v['pre_success']} %, in view {v['in_view']} of the time")
    if row.get("blank_cameras"):
        lines.append(f"cameras blacked out for the policy: {row['blank_cameras']}")
    return lines
