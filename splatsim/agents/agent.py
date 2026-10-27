from typing_extensions import Any, Dict, Optional, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError


class DummyAgent(Agent):
    """Always commands the same joint state: `target` if given, otherwise zeros."""

    def __init__(self, num_dofs: int, target: Optional[np.ndarray] = None):
        self.num_dofs = num_dofs
        self.target = np.zeros(num_dofs) if target is None else np.asarray(target, dtype=float)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.target.copy()


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )
