"""Modo waypoint de LeRobotDiffusionAgent: repite cada acción hasta que el brazo llega o se le acaban
los pasos, y mientras tanto sigue metiendo observaciones a la cola de la política."""
import unittest
from unittest import mock

import numpy as np
import torch

import splatsim.agents.lerobot_diffusion_agent as mod
from splatsim.agents.lerobot_diffusion_agent import LeRobotDiffusionAgent


class FakePyBullet:
    def resetJointState(self, *a, **k):
        pass

    def getQuaternionFromEuler(self, e):
        return (0.0, 0.0, 0.0, 1.0)


class FakePolicy:
    """La acción k es la pose (k, 0, 0, ...): k cuenta las acciones sacadas."""

    def __init__(self):
        self.popped = 0
        self.observed = 0

    def select_action(self, frame):
        self.observed += 1
        self.popped += 1
        return torch.tensor([[float(self.popped), 0, 0, 0, 0, 0, 0]])

    def reset(self):
        pass


def make_agent(tolerance, max_steps=10):
    agent = object.__new__(LeRobotDiffusionAgent)
    agent.policy = FakePolicy()
    agent.relative_actions = False
    agent.ik_orientation = "off"
    agent.waypoint_tolerance = tolerance
    agent.waypoint_max_steps = max_steps
    agent.waypoint_counts = {"reached": 0, "timeout": 0}
    agent._held = 0
    agent._last_joints = None
    agent._chunk_reference = None
    agent.make_new_prediction = True
    agent.joint_indices = list(range(5))
    agent.dummy_robot = 0
    agent._build_obs = lambda obs: {}
    agent.postprocess = lambda a: a
    agent.draw_pose = lambda *a, **k: None
    # IK falsa: las 5 articulaciones objetivo valen la x pedida
    agent._solve_ik = lambda pos, quat, rest: [float(pos[0])] * 5

    def observe_only(frame):
        agent.policy.observed += 1
    agent._observe_only = observe_only
    return agent


def obs_at(q):
    return {"joint_positions": np.array([q] * 5 + [0.0]), "state": np.zeros(7)}


@mock.patch.object(mod, "p", FakePyBullet())
class TestWaypoints(unittest.TestCase):
    def test_default_takes_one_action_per_step(self):
        agent = make_agent(None)
        for k in range(4):
            agent.act(obs_at(k + 0.5))   # lejos del objetivo k + 1, pero bajo el filtro de 2 rad
        self.assertEqual(agent.policy.popped, 4)

    def test_repeats_until_the_arm_arrives(self):
        agent = make_agent(0.01)
        # acción 1 (objetivo 1.0): el brazo sigue lejos dos pasos y después llega
        self.assertEqual(agent.act(obs_at(0.0))[0], 1.0)
        self.assertEqual(agent.act(obs_at(0.5))[0], 1.0)
        self.assertEqual(agent.act(obs_at(0.9))[0], 1.0)
        self.assertEqual(agent.act(obs_at(1.0))[0], 2.0)   # llegó: pasa a la acción 2
        self.assertEqual(agent.policy.popped, 2)
        self.assertEqual(agent.waypoint_counts, {"reached": 1, "timeout": 0})
        self.assertEqual(agent.policy.observed, 4)          # una observación por paso

    def test_gives_up_after_max_steps(self):
        agent = make_agent(0.01, max_steps=3)
        outputs = [agent.act(obs_at(0.0))[0] for _ in range(4)]
        self.assertEqual(outputs, [1.0, 1.0, 1.0, 2.0])
        self.assertEqual(agent.waypoint_counts, {"reached": 0, "timeout": 1})

    def test_reset_starts_a_new_action(self):
        agent = make_agent(0.01)
        agent.act(obs_at(0.0))
        agent.reset()
        self.assertEqual(agent.act(obs_at(0.0))[0], 2.0)


class TestObserveOnly(unittest.TestCase):
    def test_fills_observations_and_leaves_actions_alone(self):
        from collections import deque
        from types import SimpleNamespace
        agent = object.__new__(LeRobotDiffusionAgent)
        cam = "observation.images.base_rgb"
        agent.policy = SimpleNamespace(
            config=SimpleNamespace(image_features={cam: None}),
            _queues={"observation.state": deque(maxlen=2), "observation.images": deque(maxlen=2),
                     "action": deque(maxlen=4)})
        # como el frame del preprocesador: con "action" vacía
        frame = {"observation.state": torch.zeros(1, 7), cam: torch.zeros(1, 3, 4, 4), "action": None}
        agent._observe_only(frame)
        q = agent.policy._queues
        self.assertEqual(len(q["action"]), 0)
        self.assertEqual(len(q["observation.state"]), 2)
        self.assertEqual(tuple(q["observation.images"][0].shape), (1, 1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
