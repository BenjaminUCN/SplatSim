"""Estado y acción en articulaciones: action_space.py, scripts/xarm5/to_joint_space.py y el agente.

    python -m unittest discover -s tests -p "test_action_space.py"
Arma un dataset LeRobot diminuto (solo estado y acción, sin imágenes) desde .pkl falsos, así no necesita
datos ni GPU.
"""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import splatsim.agents.lerobot_diffusion_agent as agent_mod
from splatsim.agents.lerobot_diffusion_agent import LeRobotDiffusionAgent
from splatsim.utils.data.action_space import JOINT_NAMES, action_space_of, feature_names, joint_state_action
from splatsim.utils.data.dataset_sources import write_sources

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import to_joint_space  # noqa: E402


def fake_pkl(ep: int, k: int) -> dict:
    """Cuadro k del episodio ep: pose y articulaciones distintas en cada cuadro."""
    q = np.array([ep, k, 0.1 * k, -0.1 * k, 0.0, 0.5], dtype=np.float32)
    pose = np.array([0.01 * k, ep, 0.2, 3.1, -1.0, 0.1 * k, 0.5], dtype=np.float32)
    return {"joint_positions": q, "state": pose, "action": np.append(pose[:6], 1.0).astype(np.float32)}


class JointStateActionTest(unittest.TestCase):
    def test_state_is_joints_and_action_keeps_the_gripper_command(self):
        state, action = joint_state_action(fake_pkl(2, 3))
        np.testing.assert_allclose(state, [2, 3, 0.3, -0.3, 0, 0.5], rtol=1e-6)
        np.testing.assert_allclose(action, [2, 3, 0.3, -0.3, 0, 1.0], rtol=1e-6)
        self.assertEqual(action.dtype, np.float32)

    def test_names_mark_the_space(self):
        joints = {"observation.state": {"names": feature_names("joints", 6, "state")}}
        pose = {"observation.state": {"names": feature_names("pose", 7, "state")}}
        self.assertEqual(action_space_of(joints), "joints")
        self.assertEqual(action_space_of(pose), "pose")


class ToJointSpaceTest(unittest.TestCase):
    def test_rewrites_state_action_and_stats_skipping_dropped_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            recordings = tmp / "bc_data"
            lengths = {0: 5, 1: 4}
            dropped = {(1, 2)}            # la conversión descartó este cuadro (imagen faltante)
            for ep, n in lengths.items():
                folder = recordings / "rec" / f"{ep:03d}"
                folder.mkdir(parents=True)
                for k in range(n):
                    with open(folder / f"{k + 1:05d}.pkl", "wb") as f:
                        pickle.dump(fake_pkl(ep, k), f)
            features = {"observation.state": {"dtype": "float32", "shape": (7,), "names": feature_names("pose", 7, "state")},
                        "action": {"dtype": "float32", "shape": (7,), "names": feature_names("pose", 7, "action")}}
            src = tmp / "ds"
            ds = LeRobotDataset.create("local/ds", fps=10, root=src, features=features, robot_type="xarm5_lite",
                                       use_videos=False)
            for ep, n in lengths.items():
                for k in range(n):
                    if (ep, k) not in dropped:
                        d = fake_pkl(ep, k)
                        ds.add_frame({"observation.state": d["state"], "action": d["action"], "task": "t"})
                ds.save_episode()
            ds.finalize()
            write_sources(src, [{"folder": "rec", "episodes": ["000", "001"], "dataset_episodes": [0, 1]}])

            dst = tmp / "ds_joints"
            with mock.patch.object(sys, "argv", ["x", "--src", str(src), "--dst", str(dst),
                                                 "--recordings", str(recordings)]):
                to_joint_space.main()

            out = LeRobotDataset("local/ds_joints", root=dst)
            self.assertEqual(out.meta.features["observation.state"]["names"], JOINT_NAMES)
            self.assertEqual(len(out), 8)
            expected = [(ep, k) for ep, n in lengths.items() for k in range(n) if (ep, k) not in dropped]
            for i, (ep, k) in enumerate(expected):
                state, action = joint_state_action(fake_pkl(ep, k))
                np.testing.assert_allclose(out[i]["observation.state"].numpy(), state, rtol=1e-6)
                np.testing.assert_allclose(out[i]["action"].numpy(), action, rtol=1e-6)
            stats = json.loads((dst / "meta/stats.json").read_text())
            all_states = np.stack([joint_state_action(fake_pkl(ep, k))[0] for ep, k in expected])
            np.testing.assert_allclose(stats["observation.state"]["max"], all_states.max(0), rtol=1e-6)
            np.testing.assert_allclose(stats["observation.state"]["mean"], all_states.mean(0), rtol=1e-5)
            episodes = pd.concat(pd.read_parquet(f) for f in sorted((dst / "meta/episodes").rglob("*.parquet")))
            ep1 = episodes.set_index("episode_index").loc[1]
            self.assertEqual(list(ep1["stats/observation.state/max"])[1], 3.0)   # cuadros 0, 1 y 3
            self.assertEqual(list(ep1["stats/observation.state/count"]), [3])


class FakePyBullet:
    def resetJointState(self, *a, **k):
        pass


class FakePolicy:
    def select_action(self, frame):
        return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 1.0]])


@mock.patch.object(agent_mod, "p", FakePyBullet())
class JointAgentTest(unittest.TestCase):
    def test_sends_the_predicted_joints_without_ik(self):
        agent = object.__new__(LeRobotDiffusionAgent)
        agent.policy = FakePolicy()
        agent.action_space = "joints"
        agent.relative_actions = False
        agent.waypoint_tolerance = None
        agent.make_new_prediction = True
        agent.dummy_robot = 0
        agent._held = 0
        agent.gripper_threshold = None
        agent.hold_while_closing = False
        agent.latch_closed = False
        seen = []
        agent._build_obs = lambda obs: seen.append(obs) or {}
        agent.postprocess = lambda a: a
        agent._solve_ik = mock.Mock(side_effect=AssertionError("no debe usar IK"))
        obs = {"joint_positions": np.array([0.1, 0.2, 0.3, 0.35, 0.0, 0.0]), "state": np.zeros(7)}
        command = agent.act(obs)
        np.testing.assert_allclose(command, [0.1, 0.2, 0.3, 0.4, 0.0, 1.0], rtol=1e-6)
        self.assertTrue(agent.make_new_prediction)        # una acción por paso, como en pose
        agent._solve_ik.assert_not_called()

    def test_gripper_threshold_binarizes_the_command(self):
        agent = object.__new__(LeRobotDiffusionAgent)
        agent.ee_gripper = 0.62
        agent.gripper_threshold = None
        self.assertAlmostEqual(agent.gripper_command(), 0.62)
        agent.gripper_threshold = 0.5
        self.assertEqual(agent.gripper_command(), 1.0)
        agent.ee_gripper = 0.31
        self.assertEqual(agent.gripper_command(), 0.0)

    def test_hold_while_closing_keeps_the_arm_still_until_closed(self):
        agent = object.__new__(LeRobotDiffusionAgent)
        agent._hold_joints, agent.hold_count = None, 0
        hold = lambda arm, grip, closed: agent._hold_for_closing(  # noqa: E731
            {"gripper_position": [closed]}, np.array(arm, dtype=float), grip)
        joints, grip = hold([0.1] * 5, 0.3, 0.0)                # abierta, sin orden de cerrar: la política
        np.testing.assert_allclose(joints, [0.1] * 5)
        self.assertEqual(grip, 0.3)
        joints, grip = hold([0.2] * 5, 0.6, 0.0)                # ordena cerrar: quieto aquí
        np.testing.assert_allclose(joints, [0.2] * 5)
        self.assertEqual((grip, agent.hold_count), (1.0, 1))
        joints, grip = hold([0.5] * 5, 0.2, 0.5)                # la política se mueve y abre: no se usa
        np.testing.assert_allclose(joints, [0.2] * 5)
        self.assertEqual(grip, 1.0)
        joints, grip = hold([0.5] * 5, 0.9, 0.85)               # ya cerró: sigue la política
        np.testing.assert_allclose(joints, [0.5] * 5)
        self.assertEqual((grip, agent.hold_count), (0.9, 1))
        joints, grip = hold([0.6] * 5, 0.9, 0.95)               # cerrada: no vuelve a quedarse quieto
        self.assertEqual(agent.hold_count, 1)
        _, grip = hold([0.6] * 5, 0.7, 0.0)                     # se abrió y ordena cerrar otra vez
        self.assertEqual(agent.hold_count, 2)
        _, grip = hold([0.6] * 5, 0.7, 0.0)
        np.testing.assert_allclose(_, [0.6] * 5)
        joints, _ = agent._hold_for_closing({"joint_positions": np.array([0, 0, 0, 0, 0, 0.9])},
                                            np.array([0.7] * 5), 0.7)   # sin gripper_position: joint_positions[5]
        np.testing.assert_allclose(joints, [0.7] * 5)

    def test_latch_closed_keeps_the_gripper_closed_once_it_closes(self):
        agent = object.__new__(LeRobotDiffusionAgent)
        agent._latched = False
        latch = lambda grip, closed: agent._latch({"gripper_position": [closed]}, 5, grip)  # noqa: E731
        self.assertEqual(latch(0.6, 0.3), 0.6)    # cerrando: la orden de la política
        self.assertEqual(latch(0.9, 0.85), 1.0)   # ya cerró: queda cerrada
        self.assertEqual(latch(0.1, 0.85), 1.0)   # la política quiere abrir: no
        self.assertEqual(latch(0.0, 0.4), 1.0)
        agent._latched = False                    # reset() del ensayo siguiente
        self.assertEqual(latch(0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
