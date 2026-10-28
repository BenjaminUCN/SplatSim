"""Cliente de los camera nodes (scripts/launch_camera_nodes.py) con timeout.

El ZMQClientCamera de gello espera para siempre si el nodo no está corriendo, y con él el servidor
del robot entero; este falla con un mensaje claro y rehace el socket.
"""
import pickle

import zmq


class ZMQCamera:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 2.0):
        self.port, self.host, self.timeout = port, host, timeout
        self._context = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def read(self, img_size=None):
        """(imagen RGB uint8 HxWx3, profundidad) del camera node."""
        self._socket.send(pickle.dumps(img_size))
        try:
            return pickle.loads(self._socket.recv())
        except zmq.Again:
            self._socket.close()          # un REQ sin respuesta queda inutilizable
            self._connect()
            raise TimeoutError(f"El camera node del puerto {self.port} no responde "
                               "(¿corre scripts/launch_camera_nodes.py?)") from None
