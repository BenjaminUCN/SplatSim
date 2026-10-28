"""Grabación de trayectorias (.pkl + .png) con control por teclado, para el sim y el xArm5 real.

Lo que necesita del servidor (backend):
  path                 carpeta de grabaciones por defecto
  clock()              tiempo de la grabación en s (simulado en el sim, del reloj en el real)
  record_state_only    True: guardar solo el estado en cada comando (sim con sim_clock=realtime)
  state_observation()  la observación sin imágenes que se guarda en ese caso
"""
import json
import os
import pickle
import shutil
import sys
import threading

import cv2
import numpy as np


class TrajectoryRecorder:
    """
    Graba un frame por cada comando recibido (cada RECORD_EVERY comandos).

    - Normal (sim con sim_clock=fixed, robot real): usa la observación que el cliente pide
      justo después del comando, con imágenes. En el sim el dt simulado entre frames es fijo
      (every / control_hz); en el real es el del lazo del cliente.
    - record_state_only (sim con sim_clock=realtime, teleoperación): guarda en el momento del
      comando solo el estado (server.state_observation(), con sim_time y sim_joint_positions),
      sin renderizar; las imágenes se generan después a un fps fijo con
      scripts/render_recordings.py.

    El hilo del teclado solo deja una petición; start/stop se ejecutan en tick(),
    desde el serve_loop y con el lock de la simulación tomado.

    Estructura en disco:
      <server.path>/000/00001.pkl, 00002.pkl, ...
      <server.path>/000/image_1/<cam>_00001.png, ...
    """

    def __init__(self, server, max_seconds: float, every: int = 1,
                 state_only_max_seconds: float | None = None):
        self.s = server
        self.max_seconds = max_seconds   # duración máxima por defecto, en el tiempo de server.clock()
        self.state_only_max_seconds = state_only_max_seconds or max_seconds  # ídem solo estado
        self._state_only = False         # la grabación en curso guarda solo el estado
        self._max_seconds = max_seconds  # duración máxima de la grabación en curso
        self.every = every

        self.active = False
        self._request = None         # "start" / "stop", lo escribe el hilo del teclado
        self._start_time = 0.0
        self._commands = 0           # comandos recibidos durante la grabación
        self._pending = False        # hubo un comando y aún no se vio su observación
        self._length = 0             # frames guardados (numera .pkl y .png)
        self._count = 0              # índice de carpeta de la trayectoria
        self._root = None            # carpeta base de la grabación en curso

    # ── teclado ──────────────────────────────────────────────────────────────
    def start_keyboard_listener(self) -> None:
        def _listen():
            print("[Recording] Listo — 'r' + Enter para grabar, 's' + Enter para detener.")
            for line in sys.stdin:
                ch = line.strip().lower()
                if ch == "r":
                    self._request = "start"
                elif ch == "s":
                    self._request = "stop"
        threading.Thread(target=_listen, daemon=True).start()

    # ── control ──────────────────────────────────────────────────────────────
    @property
    def _folder(self) -> str:
        return os.path.join(self._root, str(self._count).zfill(3))

    def start(self, max_seconds: float | None = None, metadata: dict | None = None,
              root: str | None = None) -> str | None:
        """Empieza una grabación; devuelve su carpeta (None si ya había una activa).
        `root` reemplaza la carpeta base (por defecto server.path)."""
        if self.active:
            print("[Recording] Ya se está grabando. Pulsa 's' primero.")
            return None
        self._root = os.path.expanduser(root or self.s.path)
        os.makedirs(self._root, exist_ok=True)
        # siguiente índice tras las carpetas numeradas existentes (ignora otros archivos)
        existing = [int(d) for d in os.listdir(self._root) if d.isdigit()]
        self._count = max(existing) + 1 if existing else 0
        os.makedirs(self._folder, exist_ok=True)
        if metadata is not None:
            with open(os.path.join(self._folder, "meta.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        self._state_only = self.s.record_state_only
        self._max_seconds = max_seconds or (self.state_only_max_seconds if self._state_only
                                            else self.max_seconds)
        self._length = 0
        self._commands = 0
        self._pending = False
        self._start_time = self.s.clock()
        self.active = True
        kind = "solo estado; render después con scripts/render_recordings.py" if self._state_only else "con imágenes"
        print(f"[Recording] ▶ Grabando en {self._folder} (un frame cada {self.every} comandos, {kind})")
        return self._folder

    def stop(self) -> None:
        if not self.active:
            print("[Recording] No hay grabación activa.")
            return
        self.active = False
        if self._length == 0:
            shutil.rmtree(self._folder, ignore_errors=True)
            print("[Recording] ■ Detenido — sin frames, carpeta eliminada.")
        else:
            print(f"[Recording] ■ Guardado — {self._length} frames en {self._folder}")

    def tick(self) -> None:
        """Llamar una vez por vuelta de serve_loop: atiende las peticiones del teclado."""
        request, self._request = self._request, None
        if request == "start":
            self.start()
        elif request == "stop":
            self.stop()

    # ── hooks del servidor ───────────────────────────────────────────────────
    def on_command(self) -> None:
        """Llamar al final de command_joint_state."""
        if self.active:
            self._commands += 1
            self._pending = self._commands % self.every == 0
            if self._pending and self._state_only:
                self._pending = False
                self._save_frame(self.s.state_observation())
                self._check_length()

    def on_observation(self, obs) -> None:
        """Llamar desde get_observations: guarda la observación si sigue a un comando."""
        if not (self.active and self._pending):
            return
        self._pending = False
        self._save_frame(obs)
        self._check_length()

    def _check_length(self) -> None:
        if self.s.clock() - self._start_time >= self._max_seconds - 1e-9:
            self.stop()

    # ── guardado ─────────────────────────────────────────────────────────────
    def _save_frame(self, obs) -> None:
        self._length += 1
        save_frame(self._folder, self._length, obs)


def to_uint8_image(img) -> np.ndarray:
    """Imagen RGB HxWx3 uint8 desde un tensor CxHxW en [0, 1] (render del splat) o un array
    HxWx3 (uint8, o float en [0, 1]) de una cámara real."""
    if hasattr(img, "detach"):
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def save_frame(folder: str, index: int, obs: dict) -> None:
    """Escribe <folder>/<index>.pkl y, si la observación trae cámaras, image_1/<cam>_<index>.png."""
    images = {name: img for name, img in obs.items() if name.endswith("_rgb") and img is not None}
    if images:
        image_dir = os.path.join(folder, "image_1")
        os.makedirs(image_dir, exist_ok=True)
    for name, img in images.items():
        frame = to_uint8_image(img)
        # el índice coincide con el del .pkl
        path = os.path.join(image_dir, f"{name}_{index:05d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    with open(os.path.join(folder, f"{index:05d}.pkl"), "wb") as f:
        pickle.dump(obs, f)
