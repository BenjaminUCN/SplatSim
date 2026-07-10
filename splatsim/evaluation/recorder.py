"""
splatsim/evaluation/recorder.py

Módulo independiente de registro de episodios de evaluación.

No modifica ni importa lógica interna de LeRobotDiffusionAgent ni de RobotEnv.
El formato de cada frame es compatible con el PKL que genera save_frame() de gello,
extendido con campos opcionales de información privilegiada del entorno (poses de objetos).

Uso típico en run_env_sim.py:

    if args.record_eval:
        recorder = EvaluationRecorder(output_dir=args.eval_output_dir)

    while True:
        action = agent.act(obs)
        obs_prev = obs
        obs = env.step(action)

        if args.record_eval:
            recorder.record(obs_prev, action, obs, step=step_count)

    # en KeyboardInterrupt / finally:
    if args.record_eval:
        recorder.save_episode()
"""

from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class EvaluationRecorder:
    """
    Registra episodios generados por cualquier agente de SplatSim.

    Cada episodio se almacena en un subdirectorio propio:
        <output_dir>/
            episode_000/
                frame_000000.pkl
                frame_000001.pkl
                ...
                metadata.pkl
            episode_001/
                ...

    El formato de cada frame PKL es:
        {
            "step":        int,
            "timestamp":   str  (ISO 8601),
            "obs":         dict  (observación antes de la acción),
            "action":      np.ndarray,
            "next_obs":    dict  (observación después de la acción),
            # campos opcionales extraídos de obs/next_obs si existen:
            "plastic_apple_position":    np.ndarray | None,
            "plastic_apple_orientation": np.ndarray | None,
        }

    Parámetros
    ----------
    output_dir : str | Path
        Directorio raíz donde se crearán las carpetas de episodios.
        Se crea automáticamente si no existe.
    episode_prefix : str
        Prefijo para nombrar los subdirectorios (default: "episode").
    verbose : bool
        Si True, imprime mensajes de estado al registrar y guardar.
    """

    def __init__(
        self,
        output_dir: str | Path = "evaluation",
        episode_prefix: str = "episode",
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.episode_prefix = episode_prefix
        self.verbose = verbose

        self._frames: list[dict[str, Any]] = []
        self._episode_index: int = self._next_episode_index()
        self._step: int = 0
        self._episode_start_time: Optional[datetime.datetime] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"[EvaluationRecorder] Directorio de salida: {self.output_dir}")
            print(f"[EvaluationRecorder] Próximo episodio: {self._episode_index:03d}")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def record(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Registra un frame en el buffer interno.

        No escribe a disco hasta que se llame a save_episode().

        Parámetros
        ----------
        obs : dict
            Observación *antes* de ejecutar la acción (salida de env.get_obs()
            o del paso anterior de env.step()).
        action : np.ndarray
            Acción enviada al entorno en este paso.
        next_obs : dict
            Observación *después* de ejecutar la acción (salida de env.step()).
        step : int | None
            Número de paso. Si es None se usa un contador interno.
        """
        if self._episode_start_time is None:
            self._episode_start_time = datetime.datetime.now()

        current_step = step if step is not None else self._step

        frame: dict[str, Any] = {
            "step": current_step,
            "timestamp": datetime.datetime.now().isoformat(),
            "obs": self._copy_obs(obs),
            "action": np.array(action, dtype=np.float32),
            "next_obs": self._copy_obs(next_obs),
            # Información privilegiada del entorno — se extrae si está disponible
            "plastic_apple_position": self._extract_field(
                next_obs, "plastic_apple_position"
            ),
            "plastic_apple_orientation": self._extract_field(
                next_obs, "plastic_apple_orientation"
            ),
        }

        self._frames.append(frame)
        self._step += 1

        if self.verbose and current_step % 50 == 0:
            print(
                f"[EvaluationRecorder] Episodio {self._episode_index:03d} "
                f"— frame {current_step} registrado."
            )

    def save_episode(self) -> Path:
        """
        Escribe todos los frames del buffer a disco y resetea el estado interno.

        Returns
        -------
        Path
            Ruta al directorio del episodio recién guardado.
        """
        if not self._frames:
            if self.verbose:
                print("[EvaluationRecorder] No hay frames que guardar.")
            return self.output_dir

        episode_dir = self._episode_dir(self._episode_index)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Guardar cada frame como PKL individual (compatible con save_frame)
        for frame in self._frames:
            frame_path = episode_dir / f"frame_{frame['step']:06d}.pkl"
            with open(frame_path, "wb") as f:
                pickle.dump(frame, f)

        # Guardar metadatos del episodio
        metadata = {
            "episode_index": self._episode_index,
            "total_steps": len(self._frames),
            "start_time": (
                self._episode_start_time.isoformat()
                if self._episode_start_time
                else None
            ),
            "end_time": datetime.datetime.now().isoformat(),
            "action_dim": (
                int(self._frames[0]["action"].shape[0]) if self._frames else None
            ),
        }
        with open(episode_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        if self.verbose:
            print(
                f"[EvaluationRecorder] Episodio {self._episode_index:03d} guardado "
                f"({len(self._frames)} frames) en: {episode_dir}"
            )

        self._episode_index += 1
        self.reset()

        return episode_dir

    def reset(self) -> None:
        """Limpia el buffer de frames sin escribir a disco. Útil para descartar episodios fallidos."""
        self._frames = []
        self._step = 0
        self._episode_start_time = None

    @property
    def current_episode(self) -> int:
        """Índice del episodio que se está grabando actualmente."""
        return self._episode_index

    @property
    def frame_count(self) -> int:
        """Número de frames en el buffer actual (no guardados aún)."""
        return len(self._frames)

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _next_episode_index(self) -> int:
        """Detecta episodios ya existentes en output_dir para continuar la numeración."""
        if not self.output_dir.exists():
            return 0
        existing = sorted(
            d for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith(self.episode_prefix)
        )
        if not existing:
            return 0
        last = existing[-1].name  # e.g. "episode_007"
        try:
            return int(last.split("_")[-1]) + 1
        except ValueError:
            return len(existing)

    def _episode_dir(self, index: int) -> Path:
        return self.output_dir / f"{self.episode_prefix}_{index:03d}"

    @staticmethod
    def _copy_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copia superficial del dict de observación.

        Los arrays numpy se copian (para no almacenar referencias al buffer del env).
        Las imágenes tipo tensor se convierten a numpy uint8 para compatibilidad PKL.
        """
        copied: dict[str, Any] = {}
        for key, value in obs.items():
            if value is None:
                copied[key] = None
            elif hasattr(value, "detach"):
                # Tensor PyTorch (imágenes CxHxW del simulador)
                arr = value.detach().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8 and arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                copied[key] = arr
            elif isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, (tuple, list)):
                # Tuplas PyBullet (posiciones, cuaterniones, etc.)
                copied[key] = np.array(value, dtype=np.float64)
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _extract_field(obs: Dict[str, Any], field: str) -> Optional[np.ndarray]:
        """Extrae un campo opcional del dict de observación. Devuelve None si no existe."""
        value = obs.get(field, None)
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy()
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        try:
            return np.array(value, dtype=np.float64)  # cubre tuplas y listas PyBullet
        except (TypeError, ValueError):
            return None