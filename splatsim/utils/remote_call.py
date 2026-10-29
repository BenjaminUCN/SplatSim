"""Llamadas a métodos del servidor que no están en ZMQClientRobot (REMOTE_METHODS del servidor)."""
import pickle
import time


def call(robot, method: str, **args):
    """robot: un ZMQClientRobot conectado al servidor."""
    robot._socket.send(pickle.dumps({"method": method, "args": args}))
    result = pickle.loads(robot._socket.recv())
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{method}: {result['error']}")
    return result


def ask_operator(robot, text: str, options: list[str]) -> str:
    """Pregunta en el panel del servidor (caja Operator) y espera la respuesta."""
    print(f"[Operator] {text} {options}")
    prompt_id = call(robot, "operator_prompt", text=text, options=options)
    while (answer := call(robot, "operator_answer", prompt_id=prompt_id)) is None:
        time.sleep(0.2)
    return answer
