"""Vista previa continua de algunas cámaras para teleoperar (sim_clock=realtime).

El cliente de teleoperación manda comandos a la frecuencia de control sin pedir observaciones
(renderizar cuesta ~0.2 s); para guiarse con lo que verá la política pide la vista previa con
server.get_preview(). Un hilo la renderiza sin parar mientras se siga pidiendo, con
server.render_cameras(), que solo toma el lock de la física para copiar el estado.
"""
import threading
import time

from splatsim.robots.station.recorder import to_uint8_image

from .config import SIM_DT

IDLE_AFTER = 2.0   # s sin peticiones tras los que el hilo deja de renderizar


class PreviewStream:
    def __init__(self, server):
        self.s = server
        self._names: list[str] = []
        self._requested = 0.0
        self._frame = None           # {"id", "sim_time", "images"}
        self._id = 0
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, names: list[str], since: int):
        self._names = names
        self._requested = time.monotonic()
        self._wake.set()
        frame = self._frame
        if frame is None or frame["id"] <= since or set(frame["images"]) != set(names):
            return None
        return frame

    def _run(self):
        while True:
            if time.monotonic() - self._requested > IDLE_AFTER:
                self._wake.clear()
                self._wake.wait()
                continue
            names = self._names
            sim_time = self.s.sim_steps * SIM_DT
            images = self.s.render_cameras(names)
            self.s._set_recent_images(images)            # el panel las reusa
            self._id += 1
            self._frame = {
                "id": self._id,
                "sim_time": sim_time,
                "images": {name: to_uint8_image(img) for name, img in images.items()},
            }
