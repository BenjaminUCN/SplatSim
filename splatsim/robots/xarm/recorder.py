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
    Estructura en disco:
      <server.path>/000/00001.pkl, 00002.pkl, ...
      <server.path>/000/image_1/<cam>_00001.png, ...
    """

    def __init__(self, server, hz: int, max_seconds: float, every: int = 1):
        self.s = server
        self.hz = hz
        self.max_seconds = max_seconds
        self.every = every

        self.active = False
        self._tick = 0
        self._length = 0     # frames guardados (numera .pkl y .png)
        self._count = 0      # índice de carpeta de la trayectoria

    # ── teclado ──────────────────────────────────────────────────────────────
    def start_keyboard_listener(self) -> None:
        def _listen():
            print("[Recording] Listo — 'r' + Enter para grabar, 's' + Enter para detener.")
            for line in sys.stdin:
                ch = line.strip().lower()
                if ch == "r":
                    self.start()
                elif ch == "s":
                    self.stop()
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
        self._tick = 0
        self.active = True
        print(f"[Recording] ▶ Grabando en {self._folder} "
              f"(cada {self.every} ticks ≈ {self.hz // self.every} fps)")

    def stop(self) -> None:
        if not self.active:
            print("[Recording] No hay grabación activa.")
            return
        self.active = False
        if self._length == 0:
            shutil.rmtree(self._folder, ignore_errors=True)
            print("[Recording] ■ Detenido — sin frames, carpeta eliminada.")
        else:
            self._save_frame()  # último frame antes de cerrar
            print(f"[Recording] ■ Guardado — {self._length} frames en {self._folder}")

    def tick(self) -> None:
        """Llamar una vez por vuelta de serve_loop."""
        if not self.active:
            return
        if self._tick >= self.hz * self.max_seconds:
            self.stop()
            return
        if self._tick % self.every == 0:
            self._save_frame()
        self._tick += 1

    # ── guardado ─────────────────────────────────────────────────────────────
    def _save_frame(self) -> None:
        self._length += 1
        obs = self.s.get_observations()

        image_dir = os.path.join(self._folder, "image_1")
        os.makedirs(image_dir, exist_ok=True)
        for name, img in obs.items():
            if not name.endswith("_rgb") or img is None:
                continue
            frame = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            # el índice ahora coincide con el del .pkl
            path = os.path.join(image_dir, f"{name}_{self._length:05d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with open(os.path.join(self._folder, f"{self._length:05d}.pkl"), "wb") as f:
            pickle.dump(obs, f)
