"""Grabación de trayectorias (.pkl + .png) con control por teclado."""
import os
import pickle
import shutil
import sys
import threading

import cv2
import numpy as np


class TrajectoryRecorder:
    """
    Graba un frame por cada comando recibido (cada RECORD_EVERY comandos), usando
    la observación que el cliente pide justo después del comando. Así el dt simulado
    entre frames es fijo: every / control_hz.

    El hilo del teclado solo deja una petición; start/stop se ejecutan en tick(),
    desde el serve_loop y con el lock de la simulación tomado.

    Estructura en disco:
      <server.path>/000/00001.pkl, 00002.pkl, ...
      <server.path>/000/image_1/<cam>_00001.png, ...
    """

    def __init__(self, server, max_steps: int, every: int = 1):
        self.s = server
        self.max_steps = max_steps   # duración máxima en pasos de simulación
        self.every = every

        self.active = False
        self._request = None         # "start" / "stop", lo escribe el hilo del teclado
        self._start_step = 0
        self._commands = 0           # comandos recibidos durante la grabación
        self._pending = False        # hubo un comando y aún no se vio su observación
        self._length = 0             # frames guardados (numera .pkl y .png)
        self._count = 0              # índice de carpeta de la trayectoria

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
        return os.path.join(self.s.path, str(self._count).zfill(3))

    def start(self) -> None:
        if self.active:
            print("[Recording] Ya se está grabando. Pulsa 's' primero.")
            return
        self._count = len(os.listdir(self.s.path))
        os.makedirs(self._folder, exist_ok=True)
        self._length = 0
        self._commands = 0
        self._pending = False
        self._start_step = self.s.sim_steps
        self.active = True
        print(f"[Recording] ▶ Grabando en {self._folder} (un frame cada {self.every} comandos)")

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

    def on_observation(self, obs) -> None:
        """Llamar desde get_observations: guarda la observación si sigue a un comando."""
        if not (self.active and self._pending):
            return
        self._pending = False
        self._save_frame(obs)
        if self.s.sim_steps - self._start_step >= self.max_steps:
            self.stop()

    # ── guardado ─────────────────────────────────────────────────────────────
    def _save_frame(self, obs) -> None:
        self._length += 1

        image_dir = os.path.join(self._folder, "image_1")
        os.makedirs(image_dir, exist_ok=True)
        for name, img in obs.items():
            if not name.endswith("_rgb") or img is None:
                continue
            frame = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            # el índice coincide con el del .pkl
            path = os.path.join(image_dir, f"{name}_{self._length:05d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with open(os.path.join(self._folder, f"{self._length:05d}.pkl"), "wb") as f:
            pickle.dump(obs, f)
