"""Métricas del corte del pedúnculo en run_policy_trials.run_trial (escena con cortadora y tallo)."""
import unittest

import test_trial_stop as base


class CutRobot(base.FakeRobot):
    """El filo se acerca al punto de corte 1 cm por segundo desde 6 cm; a los 5 s corta y queda sujeto
    (el error después del corte es el del tallo que se mueve con la cortadora, y no debe contar)."""

    def get_observations(self):
        obs = super().get_observations()
        cut = self.t >= 5 - 1e-9
        obs.update(peduncle_error=0.5 if cut else max(0.06 - 0.01 * self.t, 0.012),
                   peduncle_cut=cut, peduncle_held=cut,
                   # la cortadora empieza a cerrar a los 3 s y llega a 0.8 a los 4 s
                   gripper_position=[min(max(0.8 * (self.t - 3.0), 0.0), 1.0)])
        return obs


class ClosingAgent(base.FakeAgent):
    """Ordena cerrar (último valor de la acción) desde los 3 s del reloj del robot."""

    def __init__(self, robot):
        self.robot = robot

    def act(self, obs):
        return base.np.array([0.0] * 6 + [1.0 if self.robot.t >= 3 - 1e-9 else 0.2])


class TestCutMetrics(unittest.TestCase):
    def test_records_the_cut_before_the_stem_moves(self):
        robot = CutRobot(lambda t: 0.3)
        with base.mock.patch.object(base.rpt, "grasp_error", lambda state, apple, point: apple[0]), \
                base.mock.patch.object(base.rpt, "policy_cameras", lambda agent: []):
            r = base.rpt.run_trial(robot, base.FakeAgent(), (0.0, 0.0, 0.0), 15.0, {}, "/tmp", None)
        self.assertAlmostEqual(r["min_cut_err"], 0.012)
        self.assertAlmostEqual(r["time_to_cut"], 5.0, places=3)
        self.assertEqual((r["peduncle_cut"], r["peduncle_held"]), (1, 1))
        self.assertIsNone(r["cut_err_at_command"])   # FakeAgent nunca ordena cerrar

    def test_blade_error_when_the_policy_closes(self):
        robot = CutRobot(lambda t: 0.3)
        with base.mock.patch.object(base.rpt, "grasp_error", lambda state, apple, point: apple[0]), \
                base.mock.patch.object(base.rpt, "policy_cameras", lambda agent: []):
            r = base.rpt.run_trial(robot, ClosingAgent(robot), (0.0, 0.0, 0.0), 15.0, {}, "/tmp", None)
        # la primera orden de cerrar sale con el robot en t = 3.0 s y se mide en la observación de 3.1 s
        self.assertAlmostEqual(r["cut_err_at_command"], 0.029, places=4)
        self.assertAlmostEqual(r["cut_err_at_closed"], 0.02, places=4)

    def test_grasp_scene_has_no_cut_columns(self):
        r = base.trial(lambda t: 0.3)
        self.assertNotIn("min_cut_err", r)


if __name__ == "__main__":
    unittest.main()
