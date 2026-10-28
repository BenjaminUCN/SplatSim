"""Servidor ZMQ de un robot: los métodos de ZMQClientRobot más los REMOTE_METHODS que el robot
expone (grabación, reset, vista previa...). Lo usan el servidor del sim y el del xArm5 real.

Si un método falla, se responde {"error": ...} y se sigue sirviendo (antes el hilo moría y el
cliente quedaba esperando para siempre)."""
import pickle
import threading
import traceback
from typing import Any

import zmq

from splatsim.robots.robot import Robot


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                try:
                    result = self._dispatch(method, args)
                except Exception as e:  # noqa: BLE001  se responde el error y se sigue sirviendo
                    traceback.print_exc()
                    result = {"error": f"{method}: {type(e).__name__}: {e}"}

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass
                # print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def _dispatch(self, method: str, args: dict):
        if method == "num_dofs":
            return self._robot.num_dofs()
        if method == "get_joint_state":
            return self._robot.get_joint_state()
        if method == "command_joint_state":
            return self._robot.command_joint_state(**args)
        if method == "set_object_pose":
            return self._robot.set_object_pose(**args)
        if method == "get_observations":
            return self._robot.get_observations()
        if method in getattr(self._robot, "REMOTE_METHODS", ()):
            # métodos extra que el robot expone explícitamente por ZMQ
            return getattr(self._robot, method)(**args)
        raise NotImplementedError(f"Invalid method: {method}")

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()
