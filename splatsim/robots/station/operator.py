"""Preguntas al operador: un cliente (run_policy_trials.py) las hace por ZMQ y el panel las muestra
con un botón por opción. En el robot real así se pide poner el tomate antes de cada ensayo y
marcar el resultado al final (no hay posición del tomate para medirlo)."""
import threading


class OperatorPrompt:
    def __init__(self):
        self._lock = threading.Lock()
        self.id = 0                  # cambia con cada pregunta: el panel la redibuja
        self.text: str | None = None
        self.options: list[str] = []
        self._answer: str | None = None

    def ask(self, text: str | None, options: list[str]) -> int:
        """Muestra una pregunta (text None la quita); devuelve su id para leer la respuesta."""
        with self._lock:
            self.id += 1
            self.text, self.options, self._answer = text, list(options), None
            return self.id

    def reply(self, prompt_id: int, option: str) -> None:
        with self._lock:
            if prompt_id == self.id and self.text is not None:
                self._answer, self.text, self.options = option, None, []

    def poll(self, prompt_id: int) -> str | None:
        with self._lock:
            return self._answer if prompt_id == self.id else None


class OperatorMixin:
    """REMOTE_METHODS operator_prompt / operator_answer para un servidor con self.operator."""

    def operator_prompt(self, text: str | None = None, options=()) -> int:
        return self.operator.ask(text, list(options))

    def operator_answer(self, prompt_id: int) -> str | None:
        return self.operator.poll(prompt_id)
