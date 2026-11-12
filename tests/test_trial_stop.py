import pickle
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "xarm5"))
import run_policy_trials as rpt  # noqa: E402

HZ = 10


class FakeSocket:
    """Responde las llamadas de remote_call.call (stop/start_recording, reset_robot)."""

    def __init__(self, robot):
        self.robot = robot
        self.reply = None

    def send(self, data):
        msg = pickle.loads(data)
        if msg["method"] == "stop_recording":
            self.robot.recording = False
        if msg["method"] == "start_recording":
            self.robot.recording, self.robot.max_seconds = True, msg["args"]["max_seconds"]
            self.robot.started = self.robot.t
        self.reply = pickle.dumps("folder" if msg["method"] == "start_recording" else None)

    def recv(self):
        return self.reply


class FakeRobot:
    """Sim con reloj fijo: cada comando avanza 1/HZ. El error de agarre lo da `errors(t)`."""

    def __init__(self, errors):
        self._socket = FakeSocket(self)
        self.errors, self.t, self.recording, self.started, self.max_seconds = errors, 0.0, False, 0.0, 0.0

    def set_object_pose(self, *args, **kwargs):
        pass

    def command_joint_state(self, action):
        self.t += 1 / HZ

    def get_observations(self):
        if self.recording and self.t - self.started >= self.max_seconds - 1e-9:
            self.recording = False
        return {"sim_time": self.t, "recording": self.recording, "state": np.zeros(7),
                rpt.APPLE_KEY: np.array([self.errors(self.t), 0.0, 0.0])}


class FakeAgent:
    ik_orientation, relative_actions = "off", False
    ik_counts = {"fallback": 0}

    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(6)


def trial(errors, **kwargs):
    robot = FakeRobot(errors)
    with mock.patch.object(rpt, "grasp_error", lambda state, apple, point: apple[0]), \
            mock.patch.object(rpt, "policy_cameras", lambda agent: []):
        return rpt.run_trial(robot, FakeAgent(), (0.0, 0.0, 0.0), 15.0, {}, "/tmp", None, **kwargs)


class TestStopAfterSuccess(unittest.TestCase):
    # llega al tomate a los 3 s y después sigue acercándose hasta 1 cm
    errors = staticmethod(lambda t: 0.30 if t < 3 - 1e-9 else max(0.01, 0.04 - 0.01 * (t - 3)))

    def test_runs_the_extra_seconds_after_success_then_stops(self):
        r = trial(self.errors, stop_on_success=True, success_extra_seconds=5.0)
        self.assertAlmostEqual(r["time_to_success"], 3.0, places=3)
        self.assertEqual(r["frames"], 80)              # 3 s + 5 s a 10 Hz
        self.assertEqual(r["stopped_on_success"], 1)
        self.assertAlmostEqual(r["min_grasp_err"], 0.01)

    def test_zero_extra_seconds_stops_at_success(self):
        r = trial(self.errors, stop_on_success=True, success_extra_seconds=0.0)
        self.assertEqual(r["frames"], 30)
        self.assertAlmostEqual(r["min_grasp_err"], 0.04)

    def test_success_near_the_end_runs_to_trial_seconds(self):
        r = trial(lambda t: 0.30 if t < 12 - 1e-9 else 0.02, stop_on_success=True, success_extra_seconds=5.0)
        self.assertEqual(r["frames"], 150)
        self.assertEqual(r["stopped_on_success"], 0)

    def test_without_stop_on_success_runs_full_trial(self):
        r = trial(self.errors, stop_on_success=False, success_extra_seconds=5.0)
        self.assertEqual(r["frames"], 150)
        self.assertEqual(r["stopped_on_success"], 0)


if __name__ == "__main__":
    unittest.main()
