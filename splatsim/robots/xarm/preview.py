"""Vista previa continua de algunas cámaras para teleoperar (sim_clock=realtime).

El cliente de teleoperación manda comandos a la frecuencia de control sin pedir observaciones
(renderizar cuesta ~0.2 s); para guiarse con lo que verá la política pide la vista previa con
server.get_preview(). Un hilo la renderiza sin parar mientras se siga pidiendo, con
server.render_cameras(), que solo toma el lock de la física para copiar el estado.

Varios pueden pedirla a la vez (el cliente de GELLO y las vistas del panel): se renderiza la unión
de las cámaras pedidas en los últimos IDLE_AFTER s y cada uno recibe las suyas.
"""
import threading
import time

from splatsim.robots.station.recorder import to_uint8_image

from .config import SIM_DT

IDLE_AFTER = 2.0   # s sin peticiones de una cámara tras los que deja de renderizarse


class PreviewStream:
    def __init__(self, server):
        self.s = server
        self._requested: dict[str, float] = {}   # cámara -> última vez que se pidió
        self._lock = threading.Lock()
        self._frame = None           # {"id", "sim_time", "images"}
        self._id = 0
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, names: list[str], since: int):
        """El último render con las cámaras `names` si es más nuevo que `since`, o None."""
        now = time.monotonic()
        with self._lock:
            for name in names:
                self._requested[name] = now
        self._wake.set()
        frame = self._frame
        if frame is None or frame["id"] <= since or not set(names) <= set(frame["images"]):
            return None
        return {**frame, "images": {n: frame["images"][n] for n in names}}

    def _wanted(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return [n for n, t in self._requested.items() if now - t <= IDLE_AFTER]

    def _run(self):
        while True:
            self._wake.clear()           # antes de mirar: una petición que llegue después lo despierta
            names = self._wanted()
            if not names:
                self._wake.wait()
                continue
            sim_time = self.s.sim_steps * SIM_DT
            images = self.s.render_cameras(names)
            self.s._set_recent_images(images)            # el panel las reusa
            self._id += 1
            self._frame = {
                "id": self._id,
                "sim_time": sim_time,
                "images": {name: to_uint8_image(img) for name, img in images.items()},
            }
