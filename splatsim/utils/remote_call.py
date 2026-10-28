"""Llamadas a métodos del servidor que no están en ZMQClientRobot (REMOTE_METHODS del servidor)."""
import pickle


def call(robot, method: str, **args):
    """robot: un ZMQClientRobot conectado al servidor."""
    robot._socket.send(pickle.dumps({"method": method, "args": args}))
    result = pickle.loads(robot._socket.recv())
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{method}: {result['error']}")
    return result
