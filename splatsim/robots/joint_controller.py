"""Convierte objetivos articulares (de la política o de GELLO) en comandos de bajo nivel.

La política y GELLO siempre entregan posiciones; el modo solo cambia cómo se llega a ellas.
Lo usan el servidor de simulación y el driver real del xArm5, así ambos siguen la misma ley:

  position  paso hacia el objetivo con norma acotada a max_delta por tick (comportamiento
            original del driver real); el robot llega rápido y espera el siguiente objetivo.
  interp    interpola con un perfil de mínimo jerk desde el comando anterior hasta el nuevo
            objetivo a lo largo de un periodo de comandos; movimiento continuo, con un periodo
            de retraso.
  velocity  v = feedforward + kp * error, con límites de velocidad y aceleración. El
            feedforward es la velocidad que implican dos objetivos seguidos, así el brazo sigue
            moviéndose entre objetivos en vez de detenerse en cada uno.

step() devuelve ("position", q) o ("velocity", v); quien lo llama lo traduce a su API.
Los tiempos (now) los pone quien llama: tiempo simulado en el sim, time.monotonic() en el real.
"""
from dataclasses import dataclass

import numpy as np

MODES = ("position", "interp", "velocity")


@dataclass
class ControllerGains:
    max_delta: float = 0.05     # rad, norma máxima del paso por tick (position, y tope en interp)
    kp: float = 3.0             # 1/s, ganancia proporcional (velocity)
    v_max: float = 1.0          # rad/s por articulación (velocity)
    a_max: float = 5.0          # rad/s² por articulación (velocity)
    deadband: float = 0.002     # rad, error bajo el cual no se corrige si no hay feedforward
    stale_periods: float = 2.0  # sin objetivos nuevos por más de esto, se deja de extrapolar


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (max_norm / n) if n > max_norm else v


def _min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


class JointController:
    def __init__(self, mode: str, command_period: float, gains: ControllerGains | None = None):
        """command_period: periodo nominal entre objetivos (1 / hz del cliente)."""
        if mode not in MODES:
            raise ValueError(f"control_mode debe ser uno de {MODES}, no {mode!r}")
        self.mode = mode
        self.nominal_period = command_period
        self.g = gains or ControllerGains()
        self._target = None

    def reset(self, q: np.ndarray, now: float) -> None:
        """Sin movimiento pendiente: objetivo = posición actual."""
        q = np.asarray(q, dtype=float)
        self._target = q.copy()
        self._start = q.copy()          # interp: desde dónde parte el tramo actual
        self._q_cmd = q.copy()          # último comando de posición enviado
        self._v_cmd = np.zeros_like(q)  # último comando de velocidad enviado
        self._v_ff = np.zeros_like(q)
        self._t_target = now
        self._period = self.nominal_period

    def set_target(self, q_target: np.ndarray, now: float) -> None:
        q_target = np.asarray(q_target, dtype=float)
        if self._target is None:
            self.reset(q_target, now)
            return
        # periodo medido entre objetivos (acotado: un cliente que se pausa no infla el feedforward)
        dt = now - self._t_target
        self._period = float(np.clip(dt, 0.5 * self.nominal_period, 2.0 * self.nominal_period))
        self._v_ff = np.clip((q_target - self._target) / self._period, -self.g.v_max, self.g.v_max)
        self._start = self._q_cmd.copy()
        self._target = q_target
        self._t_target = now

    def step(self, q: np.ndarray, now: float, dt: float) -> tuple[str, np.ndarray]:
        """q: posición medida; dt: tiempo desde el tick anterior."""
        q = np.asarray(q, dtype=float)
        if self._target is None:
            self.reset(q, now)
        error = self._target - q

        if self.mode == "position":
            self._q_cmd = q + _clip_norm(error, self.g.max_delta)
            return "position", self._q_cmd

        if self.mode == "interp":
            # dónde debe estar al final de este tick (al inicio daría el punto anterior y, con un
            # tick por comando, el brazo nunca avanzaría)
            s = _min_jerk((now + dt - self._t_target) / self._period)
            desired = self._start + s * (self._target - self._start)
            # mismo tope de seguridad que position, respecto a la posición medida
            self._q_cmd = q + _clip_norm(desired - q, self.g.max_delta)
            return "position", self._q_cmd

        stale = now - self._t_target > self.g.stale_periods * self._period
        v_ff = np.zeros_like(q) if stale else self._v_ff
        v = v_ff + self.g.kp * error
        v[(np.abs(error) < self.g.deadband) & (v_ff == 0)] = 0.0
        v = np.clip(v, -self.g.v_max, self.g.v_max)
        dv_max = self.g.a_max * max(dt, 1e-3)
        v = self._v_cmd + np.clip(v - self._v_cmd, -dv_max, dv_max)
        self._v_cmd = v
        return "velocity", v
