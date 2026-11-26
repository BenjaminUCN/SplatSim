"""Índice de carpetas para review_recordings.py: fecha, modelo y parámetros de cada corrida de
run_policy_trials.py, para filtrar el desplegable de carpetas y mostrar de qué se trata cada una.

Una corrida es <modelo>/<AAMMDD_HHMMSS>/ con trials.csv; sus parámetros salen de trials.csv y del
meta.json del primer episodio. Las demás carpetas (grabaciones de GELLO, renderizadas, ...) solo tienen
nombre y fecha (la de modificación de la carpeta).
"""
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from splatsim.utils.trials import trial_table

ALL = "all"
STAMP = re.compile(r"^(\d{6}_\d{6})$")
TASKS = (ALL, "grasp", "cut", "recordings")
LEAF = (ALL, "no leaf", "leaf")
SORTS = ("newest first", "oldest first", "name")


@dataclass
class RunInfo:
    path: str                       # relativa a --root, como en el desplegable
    when: datetime
    trials: int = 0                 # 0: no es una corrida de ensayos
    successes: int = 0
    task: str = "recordings"        # grasp | cut | recordings
    leaf: str = ""                  # "no leaf" | "leaf" | "" (no es corrida)
    sampler: str = ""               # "DDIM-16", "DDPM-100" o "" si la corrida no lo guardó
    params: dict = field(default_factory=dict)   # para mostrar: nombre -> valor (solo los que no son por defecto)

    @property
    def date(self) -> str:
        return self.when.strftime("%Y-%m-%d")

    @property
    def is_run(self) -> bool:
        return self.trials > 0


def _stamp(path: Path) -> datetime | None:
    m = STAMP.match(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
    except ValueError:
        return None


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _first_meta(path: Path, rows: dict) -> dict:
    for name in sorted(rows):
        try:
            return json.loads((path / name / "meta.json").read_text())
        except (OSError, ValueError):
            continue
    return {}


def _params(rows: list[dict], meta: dict) -> dict:
    """Parámetros de evaluación que distinguen corridas del mismo modelo (los por defecto no se listan)."""
    first = rows[0]
    out = {}
    ckpt = meta.get("checkpoint")
    if ckpt:
        out["model"] = Path(ckpt).name
    steps = first.get("n_action_steps") or meta.get("n_action_steps")
    if steps:
        out["action steps"] = steps
    if meta.get("ik_orientation") not in (None, "off"):
        out["IK"] = meta["ik_orientation"]
    blank = first.get("blank_cameras") or ", ".join(meta.get("blank_cameras") or [])
    if blank:
        out["blanked"] = blank
    for key in ("gripper_threshold", "waypoint_tolerance"):
        value = first.get(key) or meta.get(key)
        if value not in (None, ""):
            out[key.replace("_", " ")] = value
    for key in ("hold_while_closing", "latch_closed", "close_when_cuttable"):
        if meta.get(key):
            out[key.replace("_", " ")] = "yes"
    if meta.get("close_detector"):
        out["close detector"] = (f"{Path(str(meta['close_detector'])).name} "
                                 f"p>={meta.get('detector_threshold')} x{meta.get('detector_consecutive')}")
    if meta.get("relative_actions"):
        out["relative actions"] = "yes"
    seconds = first.get("trial_seconds") or meta.get("trial_seconds")
    if seconds:
        out["trial s"] = f"{_number(seconds):g}" if _number(seconds) is not None else seconds
    positions = {r.get("pos_idx") for r in rows}
    out["positions"] = len(positions)
    return out


def describe(root: Path, rel: str) -> RunInfo:
    """RunInfo de una carpeta del desplegable (relativa a root)."""
    path = Path(root) / rel
    when = _stamp(path)
    if when is None:
        try:
            when = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            when = datetime.fromtimestamp(0)
    rows = trial_table.load_trials(path)
    if not rows:
        return RunInfo(rel, when)
    values = list(rows.values())
    first = values[0]
    cut = any((r.get("peduncle_held") or "").strip() for r in values)
    leaf = any((_number(r.get("occlusion_target")) or 0) > 0 for r in values)
    sched, steps = first.get("noise_scheduler") or "", first.get("num_inference_steps") or ""
    return RunInfo(rel, when, trials=len(values), successes=sum(1 for r in values if r["success"]),
                   task="cut" if cut else "grasp", leaf="leaf" if leaf else "no leaf",
                   sampler=f"{sched}-{steps}" if sched and steps else sched,
                   params=_params(values, _first_meta(path, rows)))


def label(info: RunInfo, short: bool = True) -> str:
    """Texto del desplegable: fecha y hora, carpeta y, en corridas, resultado. short: sin la subcarpeta
    AAMMDD_HHMMSS de la corrida, que repite la fecha."""
    path = info.path
    if short and "/" in path and STAMP.match(path.rsplit("/", 1)[1]):
        path = path.rsplit("/", 1)[0]
    text = f"{info.when:%m-%d %H:%M} · {path}"
    if info.is_run:
        what = "cuts" if info.task == "cut" else "✓"
        text += f" · {info.successes}/{info.trials} {what}"
    return text


def date_options(infos: list[RunInfo]) -> list[str]:
    """Fechas con carpetas, la más reciente primero, con "all" adelante."""
    return [ALL, *sorted({i.date for i in infos}, reverse=True)]


def sampler_options(infos: list[RunInfo]) -> list[str]:
    return [ALL, *sorted({i.sampler for i in infos if i.sampler})]


def matches(info: RunInfo, date: str = ALL, search: str = "", task: str = ALL, leaf: str = ALL,
            sampler: str = ALL) -> bool:
    """Si la carpeta pasa los filtros. search: palabras que deben aparecer todas (sin distinguir
    mayúsculas) en la ruta o en los parámetros; "-palabra" excluye."""
    if date != ALL and info.date != date:
        return False
    if task != ALL and info.task != task:
        return False
    if leaf != ALL and info.leaf != leaf:
        return False
    if sampler != ALL and info.sampler != sampler:
        return False
    text = " ".join([info.path, *map(str, info.params.values())]).lower()
    for word in search.lower().split():
        if word.startswith("-") and len(word) > 1:
            if word[1:] in text:
                return False
        elif word not in text:
            return False
    return True


def ordered(infos: list[RunInfo], sort: str = SORTS[0]) -> list[RunInfo]:
    if sort == "name":
        return sorted(infos, key=lambda i: i.path)
    return sorted(infos, key=lambda i: (i.when, i.path), reverse=sort == "newest first")


def details(info: RunInfo) -> str:
    """Markdown con la corrida abierta."""
    if not info.is_run:
        return f"`{info.path}` · {info.when:%Y-%m-%d %H:%M}"
    rate = 100 * info.successes / info.trials
    what = "cuts (peduncle held)" if info.task == "cut" else "successes"
    lines = [f"**{info.when:%Y-%m-%d %H:%M}** · {info.task}, {info.leaf} · "
             f"{info.successes}/{info.trials} {what} ({rate:.0f} %)"]
    params = dict(info.params)
    if info.sampler:
        params = {"sampler": info.sampler, **params}
    lines.append(" · ".join(f"{k} `{v}`" for k, v in params.items()))
    return "  \n".join(lines)
