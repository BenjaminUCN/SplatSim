"""Revisa grabaciones de GELLO: trayectoria 3D del efector con gradiente inicio→fin, imágenes
por frame y selección de episodios (los descartados no entran a los datasets).

    python scripts/review_recordings.py                      # carpetas de ~/data/bc_data
    python scripts/review_recordings.py --root ~/data/bc_data --port 8081

Abre http://localhost:8081. No necesita el simulador.

Los .pkl traen las imágenes (≈7 MB cada uno), así que leer un episodio toma ~1-2 s. Las
trayectorias se guardan en caché (~/.cache/splatsim/trails/) y un hilo carga el resto de la
carpeta en segundo plano para poder mostrar todas a la vez.

La selección se guarda en <carpeta>/selection.yaml (splatsim/utils/episode_selection.py);
lerobot_transform_v2.py omite los episodios descartados.

Etiquetas de parada para el clasificador visual (splatsim/utils/stop_labels.py): en sim se
proponen solas con el error de agarre (configs/grasp_point.yaml); en la carpeta Stop labels se
aceptan, se corrigen ("válido desde este frame") o se excluyen, y quedan en
<carpeta>/stop_labels.yaml. Las grabaciones reales (sin posición del tomate) se etiquetan a mano.

Carpetas renderizadas con scripts/render_recordings.py (traen render_info.yaml): se superpone la
trayectoria original de teleoperación (~30 Hz, naranja) a la remuestreada (puntos azules, un
punto por frame) y la carpeta Resampling compara ambas en el tiempo.
"""
import io
import hashlib
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser
import yaml
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splatsim.utils.episode_selection import load_discarded, save_discarded  # noqa: E402
from splatsim.utils.viser_page import set_page_title  # noqa: E402
from splatsim.utils.grasp_point import GRASP_POINT_PATH, grasp_error, load_grasp_point  # noqa: E402
from splatsim.utils.stop_labels import (  # noqa: E402
    VALID_RADIUS,
    auto_valid_from,
    episode_labels,
    load_stop_labels,
    save_stop_labels,
)

CACHE_DIR = Path("~/.cache/splatsim/trails").expanduser()
APPLE_KEY = "plastic_apple_position"
COLORMAP = "viridis"         # inicio (violeta) -> fin (amarillo)
PLAYBACK_FPS = 10
_IMG = re.compile(r"^(.+_rgb)_(\d+)\.png$")
# colores de la trayectoria por etiqueta de parada (paleta de referencia)
LABEL_COLORS = {1: (0, 131, 0), 0: (42, 120, 214), -1: (175, 175, 170)}
RENDER_INFO_FILE = "render_info.yaml"   # lo escribe scripts/render_recordings.py
# comparación original / remuestreada (paleta de referencia: serie 1 azul, serie 2 naranja)
RESAMPLED_HEX, SOURCE_HEX = "#2a78d6", "#eb6834"
RESAMPLED_RGB, SOURCE_RGB = (42, 120, 214), (235, 104, 52)


@dataclass
class Args:
    root: str = "~/data/bc_data"   # carpeta con carpetas de grabaciones (gello, gello_3plants, ...)
    port: int = 8081


def _pkl_key(p: Path):
    return (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem)


# ── datos de una carpeta ─────────────────────────────────────────────────────
class RecordingFolder:
    """Episodios de una carpeta y sus trayectorias (con caché en disco)."""

    def __init__(self, path: Path):
        self.path = path
        self.episodes = sorted((d for d in path.iterdir() if d.is_dir() and any(d.glob("*.pkl"))),
                               key=lambda d: (not d.name.isdigit(), int(d.name) if d.name.isdigit() else 0, d.name))
        self.names = [d.name for d in self.episodes]
        self.discarded = load_discarded(path)
        self.stop_labels = load_stop_labels(path)
        try:
            self.grasp_point = load_grasp_point()
        except FileNotFoundError:
            self.grasp_point = None
            print(f"[Review] Falta {GRASP_POINT_PATH} (scripts/estimate_grasp_point.py): "
                  "sin etiquetas automáticas")
        key = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
        self._cache_path = CACHE_DIR / f"{path.name}-{key}.pkl"
        self._cache = self._load_cache()
        self._lock = threading.Lock()
        self._images = {}
        # carpeta renderizada desde grabaciones de solo estado: de dónde salió
        info_path = path / RENDER_INFO_FILE
        self.render_info = yaml.safe_load(info_path.read_text()) if info_path.exists() else None
        self._source = None

    @property
    def source(self) -> "RecordingFolder | None":
        """Carpeta original (teleoperación, solo estado) si esta es una renderizada."""
        if self._source is None and self.render_info:
            src = Path(self.render_info["source"])
            if src.is_dir():
                self._source = RecordingFolder(src)
        return self._source

    def _load_cache(self) -> dict:
        try:
            with open(self._cache_path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return {}

    def save_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = dict(self._cache)
        tmp = self._cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, self._cache_path)

    def cached(self, name: str) -> dict | None:
        """Trayectoria en caché si sigue vigente (misma fecha de modificación)."""
        with self._lock:
            entry = self._cache.get(name)
        ep = self.path / name
        if entry and entry["mtime"] == ep.stat().st_mtime and "rpy" in entry:
            return entry
        return None

    def trail(self, name: str) -> dict:
        """Posición del efector, gripper y tomate por frame (lee los .pkl si no está en caché)."""
        entry = self.cached(name)
        if entry is not None:
            return entry
        ep = self.path / name
        ee, rpy, grip, t, apple, bad, src = [], [], [], [], None, [], []
        for p in sorted(ep.glob("*.pkl"), key=_pkl_key):
            try:
                with open(p, "rb") as f:
                    obs = pickle.load(f)
            except Exception:  # noqa: BLE001  p.ej. un .pkl cortado al detener la grabación
                bad.append(p.name)
                continue
            ee.append(np.asarray(obs["state"][:3], dtype=float))
            rpy.append(np.asarray(obs["state"][3:6], dtype=float))
            grip.append(float(np.ravel(obs["state"])[6]) if len(obs["state"]) > 6 else 0.0)
            t.append(obs.get("sim_time", np.nan))
            if apple is None and obs.get(APPLE_KEY) is not None:
                apple = np.asarray(obs[APPLE_KEY], dtype=float)
            if "source_frames" in obs:
                src.append(obs["source_frames"])
        entry = {"mtime": ep.stat().st_mtime, "ee": np.array(ee).reshape(-1, 3),
                 "rpy": np.array(rpy).reshape(-1, 3),
                 "gripper": np.array(grip), "sim_time": np.array(t, dtype=float), "apple": apple,
                 "unreadable": bad,
                 # render_recordings.py: (frame i, frame i+1, peso) de la grabación original
                 "source_frames": np.array(src, dtype=float) if src else None}
        with self._lock:
            self._cache[name] = entry
        return entry

    def images(self, name: str) -> dict[str, list[Path]]:
        """Cámara -> imágenes del episodio en orden (una por frame, sirve para ambas numeraciones)."""
        if name not in self._images:
            image_dir = next((self.path / name / d for d in ("image_1", "images")
                              if (self.path / name / d).is_dir()), None)
            per_cam = {}
            if image_dir is not None:
                for f in os.scandir(image_dir):
                    if m := _IMG.match(f.name):
                        per_cam.setdefault(m.group(1), []).append((int(m.group(2)), Path(f.path)))
            self._images[name] = {cam: [p for _, p in sorted(files)] for cam, files in sorted(per_cam.items())}
        return self._images[name]

    def grasp_err(self, entry: dict) -> np.ndarray | None:
        """Error de agarre por frame (None si no hay tomate o punto de agarre)."""
        if entry["apple"] is None or self.grasp_point is None or not len(entry["ee"]):
            return None
        state = np.hstack([entry["ee"], entry["rpy"]])
        return grasp_error(state, entry["apple"], self.grasp_point)

    def labels(self, name: str, entry: dict) -> tuple[np.ndarray, str]:
        return episode_labels(len(entry["ee"]), self.stop_labels.get(name), self.grasp_err(entry))

    def set_stop_label(self, name: str, label: dict | None) -> None:
        if label is None:
            self.stop_labels.pop(name, None)
        else:
            self.stop_labels[name] = label
        save_stop_labels(self.path, self.stop_labels)

    def set_discarded(self, name: str, discarded: bool) -> None:
        (self.discarded.add if discarded else self.discarded.discard)(name)
        save_discarded(self.path, self.discarded)


# ── app ──────────────────────────────────────────────────────────────────────
class Reviewer:
    def __init__(self, root: Path, port: int):
        self.root = root
        self.folder: RecordingFolder | None = None
        self.episode: str | None = None
        self.trail_data: dict | None = None
        self.playing = False
        self._loader_gen = 0          # al cambiar de carpeta, el hilo de carga anterior se detiene
        self._loaded = 0
        self._need_background = False

        self.v = viser.ViserServer(port=port, label="Recordings review", verbose=False)
        set_page_title(self.v, "Recordings review")
        self.v.scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)
        self.v.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1, section_size=0.5)
        self._build_gui()
        self.v.gui.main_panel.dock_right()
        self.v.gui.main_panel.set_width(380)
        folders = self._folders()
        self.folder_select.options = folders or ["-"]
        if folders:
            self.open_folder(folders[0])
        print(f"[Review] http://localhost:{port}")

    def _folders(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and any(e.is_dir() and any(e.glob("*.pkl")) for e in d.iterdir()))

    # ── GUI ──────────────────────────────────────────────────────────────────
    def _build_gui(self):
        gui = self.v.gui
        with gui.add_folder("Recordings"):
            self.folder_select = gui.add_dropdown("Folder", ["-"])
            self.load_progress = gui.add_progress_bar(0.0)
            self.load_status = gui.add_markdown("")
        with gui.add_folder("Episode"):
            self.filter_select = gui.add_dropdown("Show", ["all", "kept", "discarded", "stop: unlabeled"],
                                                  initial_value="all")
            self.episode_select = gui.add_dropdown("Episode", ["-"])
            nav = gui.add_button_group("Navigate", ["◀ Prev", "Next ▶"])
            self.discard_box = gui.add_checkbox("Discard episode", False,
                                                hint="Excluded from datasets (selection.yaml)")
            self.episode_info = gui.add_markdown("")
        with gui.add_folder("Stop labels"):
            self.stop_info = gui.add_markdown("")
            self.stop_plot = gui.add_uplot(
                (np.arange(2.0), np.zeros(2), np.zeros(2)),
                ({"label": "frame"},
                 {"label": "grasp error (cm)", "stroke": "#2a78d6", "width": 2},
                 {"label": f"in position ≤ {VALID_RADIUS * 100:.0f} cm", "stroke": "#008300",
                  "width": 1, "dash": [4, 3]}),
                height=170, aspect=2.0)
            set1 = gui.add_button_group("Label", ["Accept auto", "Valid from this frame"])
            set2 = gui.add_button_group("Other", ["No valid frames", "Exclude", "Reset"])
        with gui.add_folder("Playback"):
            self.frame_slider = gui.add_slider("Frame", 0, 1, 1, 0)
            self.play_button = gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        with gui.add_folder("View"):
            self.show_all = gui.add_checkbox("Show other trails", True)
            self.color_mode = gui.add_dropdown("Trail color", ["gradient", "stop label"],
                                               initial_value="gradient")
            self.show_source = gui.add_checkbox(
                "Show source recording", True,
                hint="Rendered folders: original teleoperation trail (orange) and resampled frames (blue)")
            gui.add_markdown(f"Gradient: start → end (`{COLORMAP}`). Stop label: green = in position, "
                             "blue = not yet, gray = not used. Other trails: gray = kept, red = discarded")

        self.resample_folder = gui.add_folder("Resampling", expand_by_default=True)
        with self.resample_folder:
            self.resample_info = gui.add_markdown("")
            self.resample_plot = gui.add_image(np.full((8, 8, 3), 252, np.uint8), format="png")
        self.resample_folder.visible = False

        cams = gui.add_panel()
        self._cams_tab = cams.add_tab("Cameras", viser.Icon.CAMERA)
        with self._cams_tab:
            self.image_area = gui.add_markdown("")
        self.images = {}
        cams.dock_left()
        cams.set_width(340)

        self.folder_select.on_update(lambda e: e.client and self.open_folder(self.folder_select.value))
        self.filter_select.on_update(lambda e: e.client and self._refresh_episode_list())
        self.episode_select.on_update(
            lambda e: e.client and self.open_episode(self.episode_select.value.split(" ")[0]))
        nav.on_click(lambda e: self._step_episode(-1 if e.target.value.startswith("◀") else 1))
        self.discard_box.on_update(lambda e: e.client and self._toggle_discard())
        self.frame_slider.on_update(lambda _: self._show_frame(int(self.frame_slider.value)))
        self.play_button.on_click(lambda _: self._toggle_play())
        self.show_all.on_update(lambda _: self._draw_background())
        self.color_mode.on_update(lambda _: self.trail_data is not None and self._draw_trail())
        self.show_source.on_update(lambda _: self.trail_data is not None and self._draw_source())
        set1.on_click(lambda e: self._stop_action(e.target.value))
        set2.on_click(lambda e: self._stop_action(e.target.value))

    # ── carpeta ──────────────────────────────────────────────────────────────
    def open_folder(self, name: str):
        self.folder = RecordingFolder(self.root / name)
        self.folder_select.value = name
        self._refresh_episode_list()
        visible = self._visible_names()
        if visible:
            self.open_episode(visible[0])
        self._loader_gen += 1
        threading.Thread(target=self._load_all, args=(self.folder, self._loader_gen), daemon=True).start()

    def _load_all(self, folder: RecordingFolder, gen: int):
        """Hilo: carga (y guarda en caché) todas las trayectorias de la carpeta."""
        t0 = time.monotonic()
        for i, name in enumerate(folder.names):
            if gen != self._loader_gen:
                return
            fresh = folder.cached(name) is None
            try:
                folder.trail(name)
            except Exception as e:  # noqa: BLE001  un episodio roto no detiene la carga del resto
                print(f"[Review] {folder.path.name}/{name}: {type(e).__name__}: {e}")
            self._loaded = i + 1
            if fresh and (i + 1) % 10 == 0:
                folder.save_cache()
            self._need_background = True
        folder.save_cache()
        print(f"[Review] {folder.path.name}: {len(folder.names)} trayectorias en {time.monotonic() - t0:.0f} s")

    def _visible_names(self) -> list[str]:
        f, mode = self.folder, self.filter_select.value
        if mode == "kept":
            return [n for n in f.names if n not in f.discarded]
        if mode == "discarded":
            return [n for n in f.names if n in f.discarded]
        if mode == "stop: unlabeled":
            return [n for n in f.names if n not in f.stop_labels and n not in f.discarded]
        return list(f.names)

    def _label(self, name: str) -> str:
        entry = self.folder.cached(name)
        frames = f" · {len(entry['ee'])} fr" if entry else ""
        broken = " · ⚠" if entry and entry.get("unreadable") else ""
        manual = self.folder.stop_labels.get(name)
        stop = (" · ⊘" if manual.get("exclude") else " · ✓") if manual else ""
        return f"{name}{frames}{broken}{stop}" + (" · ✗" if name in self.folder.discarded else "")

    def _refresh_episode_list(self):
        names = self._visible_names()
        options = [self._label(n) for n in names] or ["-"]
        self.episode_select.options = options
        if self.episode in names:
            self.episode_select.value = options[names.index(self.episode)]

    # ── episodio ─────────────────────────────────────────────────────────────
    def open_episode(self, name: str):
        if self.folder is None or name not in self.folder.names:
            return
        self.playing = False
        self.play_button.label = "Play"
        self.episode = name
        self.trail_data = self.folder.trail(name)
        self.discard_box.value = name in self.folder.discarded
        self._refresh_episode_list()
        self._draw_trail()
        self._build_image_widgets()
        n = len(self.trail_data["ee"])
        self.frame_slider.max = max(n - 1, 1)
        self.frame_slider.value = 0
        self._show_frame(0)
        self._show_info()
        self._show_stop()
        self._draw_background()
        self._draw_source()
        self._show_resampling()

    def _step_episode(self, delta: int):
        names = self._visible_names()
        if not names:
            return
        i = names.index(self.episode) if self.episode in names else -1
        self.open_episode(names[(i + delta) % len(names)])

    def _toggle_discard(self):
        self.folder.set_discarded(self.episode, self.discard_box.value)
        self._refresh_episode_list()
        self._show_info()
        self._draw_background()

    def _stop_action(self, action: str):
        f, d, name = self.folder, self.trail_data, self.episode
        if d is None:
            return
        if action == "Accept auto":
            err = f.grasp_err(d)
            if err is None:
                self.stop_info.content = "**No automatic label** (no tomato position): set it by hand"
                return
            f.set_stop_label(name, {"valid_from": auto_valid_from(err)})
        elif action == "Valid from this frame":
            f.set_stop_label(name, {"valid_from": int(self.frame_slider.value)})
        elif action == "No valid frames":
            f.set_stop_label(name, {"valid_from": None})
        elif action == "Exclude":
            f.set_stop_label(name, {"exclude": True})
        elif action == "Reset":
            f.set_stop_label(name, None)
        self._refresh_episode_list()
        self._show_stop()
        self._draw_trail()

    def _show_stop(self):
        f, d = self.folder, self.trail_data
        n = len(d["ee"])
        labels, source = f.labels(self.episode, d)
        err = f.grasp_err(d)
        manual = f.stop_labels.get(self.episode)
        if source == "excluded":
            head = "**excluded** from the classifier dataset"
        elif source == "none":
            head = "no label yet (no tomato position): set **Valid from this frame** or **No valid frames**"
        else:
            vf = manual["valid_from"] if source == "manual" else auto_valid_from(err)
            where = f"valid from frame **{vf}**" if vf is not None else "**no valid frames**"
            head = f"{'**manual**' if source == 'manual' else 'auto (not reviewed)'}: {where}"
        counts = f"{(labels == 1).sum()} in position · {(labels == 0).sum()} not · {(labels == -1).sum()} unused"
        lines = [head, counts]
        if err is not None:
            lines.append(f"grasp error: last frame {err[-1] * 100:.1f} cm, min {err.min() * 100:.1f} cm")
            if source == "manual" and manual.get("valid_from") is not None and \
                    err[manual["valid_from"]:].max() > 2 * VALID_RADIUS:
                lines.append("⚠ the arm is > 6 cm from the grasp point after valid_from")
        done = sum(1 for n_ in f.names if n_ in f.stop_labels)
        lines.append(f"folder: {done}/{len(f.names)} episodes labeled by hand")
        self.stop_info.content = "  \n".join(lines)
        x = np.arange(n, dtype=float)
        y = err * 100 if err is not None else np.zeros(n)
        self.stop_plot.data = (x, y, np.full(n, VALID_RADIUS * 100))

    def _show_info(self):
        d, f = self.trail_data, self.folder
        n = len(d["ee"])
        lines = [f"**{self.episode}** — {n} frames"]
        t = d["sim_time"]
        if np.isfinite(t).all() and n > 1:
            lines[0] += f", {t[-1] - t[0]:.1f} s (dt {np.median(np.diff(t)):.3f})"
        if d["apple"] is not None:
            dist = np.linalg.norm(d["ee"] - d["apple"], axis=1)
            lines.append(f"EE–tomato: final {dist[-1]:.3f} m, min {dist.min():.3f} m (frame {dist.argmin()})")
        g = d["gripper"]
        if g.max() - g.min() > 0.1:
            lines.append(f"gripper {g[0]:.2f} → {g[-1]:.2f}")
        if d.get("unreadable"):
            lines.append(f"**⚠ unreadable files:** {', '.join(d['unreadable'])} (skipped)")
        kept = len(f.names) - len(f.discarded)
        lines.append(f"folder: {kept} kept, {len(f.discarded)} discarded of {len(f.names)}")
        self.episode_info.content = "  \n".join(lines)

    # ── dibujo ───────────────────────────────────────────────────────────────
    def _draw_trail(self):
        d = self.trail_data
        ee = d["ee"]
        if len(ee) >= 2:
            if self.color_mode.value == "stop label":
                labels, _ = self.folder.labels(self.episode, d)
                c = np.array([LABEL_COLORS[int(v)] for v in labels], np.uint8)
            else:
                c = (colormaps[COLORMAP](np.linspace(0, 1, len(ee)))[:, :3] * 255).astype(np.uint8)
            points = np.stack([ee[:-1], ee[1:]], axis=1)
            colors = np.stack([c[:-1], c[1:]], axis=1)
            self.v.scene.add_line_segments("/trail", points, colors, thickness=3, thickness_units="screen")
            self.v.scene.add_icosphere("/trail_start", radius=0.01, color=tuple(int(x) for x in c[0]), position=ee[0])
            self.v.scene.add_icosphere("/trail_end", radius=0.01, color=tuple(int(x) for x in c[-1]), position=ee[-1])
        if d["apple"] is not None:
            self.v.scene.add_icosphere("/tomato", radius=0.03, color=(220, 40, 30), opacity=0.7,
                                       position=d["apple"])
        self._ee_marker = self.v.scene.add_icosphere("/ee", radius=0.012, color=(255, 255, 255),
                                                     position=ee[0])

    def _source_trail(self) -> dict | None:
        """Trayectoria original del episodio actual (carpetas renderizadas)."""
        src = self.folder.source if self.folder else None
        if src is None or self.episode not in src.names:
            return None
        return src.trail(self.episode)

    def _draw_source(self):
        """Original (línea y muestras naranjas) y frames remuestreados (puntos azules)."""
        source = self._source_trail() if self.show_source.value else None
        if source is None or len(source["ee"]) < 2:
            for name in ("/source", "/source_samples", "/resampled_samples"):
                self.v.scene.add_point_cloud(name, np.zeros((0, 3)), colors=(0, 0, 0), point_size=0.001)
            return
        ee_s = source["ee"]
        self.v.scene.add_line_segments("/source", np.stack([ee_s[:-1], ee_s[1:]], axis=1), SOURCE_RGB,
                                       thickness=1.5, thickness_units="screen")
        self.v.scene.add_point_cloud("/source_samples", ee_s, colors=SOURCE_RGB, point_size=0.004,
                                     point_shape="circle")
        self.v.scene.add_point_cloud("/resampled_samples", self.trail_data["ee"], colors=RESAMPLED_RGB,
                                     point_size=0.008, point_shape="circle")

    def _show_resampling(self):
        """Carpeta Resampling: cifras y figura original vs remuestreada (solo carpetas renderizadas)."""
        source = self._source_trail()
        d = self.trail_data
        self.resample_folder.visible = source is not None
        if source is None:
            return
        ts, tr = source["sim_time"], d["sim_time"]
        info = self.folder.render_info
        lines = [f"rendered at **{info['fps']} fps** from `{Path(info['source']).name}` "
                 f"({', '.join(info.get('cameras', []))})"]
        if len(ts) > 1 and np.isfinite(ts).all():
            gaps = np.diff(ts)
            lines.append(f"original: {len(ts)} frames, {1 / np.median(gaps):.1f} Hz, "
                         f"largest gap {gaps.max() * 1000:.0f} ms")
        if len(tr) > 1 and np.isfinite(tr).all():
            dt = np.diff(tr)
            lines.append(f"resampled: {len(tr)} frames, dt {dt.mean() * 1000:.1f} ± {dt.std() * 1000:.2f} ms")
            # pose recalculada desde las articulaciones interpoladas vs la trayectoria original
            interp = np.stack([np.interp(tr, ts, source["ee"][:, i]) for i in range(3)], axis=1)
            dev = np.linalg.norm(d["ee"] - interp, axis=1)
            lines.append(f"resampled EE vs original path: max {dev.max() * 1000:.1f} mm, "
                         f"mean {dev.mean() * 1000:.1f} mm")
        self.resample_info.content = "  \n".join(lines)
        self.resample_plot.image = _resampling_figure(source, d)

    def _draw_background(self):
        """Las demás trayectorias cargadas, finas: gris si se usan, rojo si están descartadas."""
        self._need_background = False
        if self.folder is None or not self.show_all.value:
            self.v.scene.add_line_segments("/others", np.zeros((0, 2, 3)), (0, 0, 0))
            return
        segs, cols = [], []
        for name in self._visible_names():
            entry = self.folder.cached(name)
            if name == self.episode or entry is None or len(entry["ee"]) < 2:
                continue
            ee = entry["ee"]
            segs.append(np.stack([ee[:-1], ee[1:]], axis=1))
            color = (200, 60, 60) if name in self.folder.discarded else (150, 150, 150)
            cols.append(np.broadcast_to(np.array(color, np.uint8), (len(ee) - 1, 2, 3)))
        if segs:
            self.v.scene.add_line_segments("/others", np.concatenate(segs), np.concatenate(cols),
                                           thickness=1, thickness_units="screen")

    def _build_image_widgets(self):
        for handle in self.images.values():
            handle.remove()
        self.images = {}
        cams = self.folder.images(self.episode)
        with self._cams_tab:                      # pestaña Cameras del panel izquierdo
            for cam in cams:
                self.images[cam] = self.v.gui.add_image(np.zeros((240, 320, 3), np.uint8),
                                                        label=cam, format="jpeg")
        self.image_area.content = "" if cams else "No images in this episode"

    def _show_frame(self, k: int):
        d = self.trail_data
        if d is None or not len(d["ee"]):
            return
        k = min(k, len(d["ee"]) - 1)
        self._ee_marker.position = d["ee"][k]
        for cam, files in self.folder.images(self.episode).items():
            if cam in self.images and k < len(files):
                img = Image.open(files[k]).convert("RGB")
                self.images[cam].image = np.asarray(img.resize((img.width // 2, img.height // 2)))

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_button.label = "Pause" if self.playing else "Play"

    def run(self):
        last_bg = 0.0
        while True:
            if self.playing and self.trail_data is not None:
                k = int(self.frame_slider.value) + 1
                if k > self.frame_slider.max:
                    self._toggle_play()
                else:
                    self.frame_slider.value = k
            if self.folder is not None:
                total = len(self.folder.names)
                self.load_progress.value = 100.0 * self._loaded / max(total, 1)
                self.load_status.content = (f"{self._loaded}/{total} trails loaded" if self._loaded < total
                                            else f"{total} episodes")
                # mientras se cargan, se redibujan las demás trayectorias cada 2 s
                if self._need_background and time.monotonic() - last_bg > 2.0:
                    last_bg = time.monotonic()
                    self._draw_background()
                    self._refresh_episode_list()
            time.sleep(1 / PLAYBACK_FPS)


def _resampling_figure(source: dict, resampled: dict) -> np.ndarray:
    """Posición del efector (x, y, z) y gripper en el tiempo: original (línea con sus muestras)
    y remuestreada (un punto por frame). Small multiples con el eje de tiempo compartido."""
    from matplotlib.figure import Figure
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    t0 = source["sim_time"][0]
    ts, tr = source["sim_time"] - t0, resampled["sim_time"] - t0
    rows = [("x (m)", source["ee"][:, 0], resampled["ee"][:, 0]),
            ("y (m)", source["ee"][:, 1], resampled["ee"][:, 1]),
            ("z (m)", source["ee"][:, 2], resampled["ee"][:, 2]),
            ("gripper", source["gripper"], resampled["gripper"])]
    fig = Figure(figsize=(3.6, 4.6), dpi=200, facecolor=surface)
    axes = fig.subplots(len(rows), 1, sharex=True)
    for ax, (label, ys, yr) in zip(axes, rows):
        ax.set_facecolor(surface)
        ax.plot(ts, ys, color=SOURCE_HEX, linewidth=1, marker="o", markersize=1.6,
                label=f"original ({len(ts)} frames)")
        ax.plot(tr, yr, linestyle="none", marker="o", markersize=3.2, color=RESAMPLED_HEX,
                markeredgecolor=surface, markeredgewidth=0.6, label=f"resampled ({len(tr)} frames)")
        ax.set_ylabel(label, fontsize=6, color=muted)
        if label == "gripper":   # escala completa: si no, el ruido de un gripper quieto parece movimiento
            ax.set_ylim(min(-0.05, ys.min(), yr.min()), max(1.05, ys.max(), yr.max()))
        ax.tick_params(labelsize=5, colors=muted, length=2)
        ax.grid(True, color="#e6e5e0", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    axes[-1].set_xlabel("time (s)", fontsize=6, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=5.5, frameon=False,
               labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=surface)
    return np.asarray(Image.open(buf).convert("RGB"))


def main(args: Args):
    Reviewer(Path(args.root).expanduser(), args.port).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
