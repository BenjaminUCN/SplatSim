import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import torch
from e3nn import o3

from splatsim.robots.station.recorder import TrajectoryRecorder, save_frame
from splatsim.robots.xarm.splat_transform import RobotSplatTransform
from splatsim.utils.robot_splat_render_utils import get_curr_link_states, get_segmented_indices, transform_means
from splatsim.utils.utils_fk import compute_transformation


class FakeGaussians:
    """Lo que usan transform_means y RobotSplatTransform de un GaussianModel."""

    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self._xyz = torch.rand(n, 3, generator=g) * 2 - 1
        self._rotation = torch.randn(n, 4, generator=g)
        self.get_opacity_raw = torch.randn(n, 1, generator=g)
        self._features_dc = torch.randn(n, 1, 3, generator=g)
        self._features_rest = torch.randn(n, 15, 3, generator=g)

    # como en GaussianModel: propiedades que devuelven tensores nuevos (transform_means modifica
    # en el lugar lo que recibe de get_rotation)
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)


class TestRobotSplatTransform(unittest.TestCase):
    """Mismo resultado que las funciones por link de robot_splat_render_utils."""

    def setUp(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)

    def tearDown(self):
        p.disconnect(self.client)

    def test_matches_legacy_transform_means(self):
        pc = FakeGaussians(3000)
        T = torch.eye(4)
        T[:3, :3] = 1.7 * o3.angles_to_matrix(torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(0.5))
        T[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
        cache = {"r": {"transformation": T, "transformation_scale": torch.pow(torch.linalg.det(T[:3, :3]), 1 / 3),
                       "inv_transformation": torch.inverse(T)}}
        aabb = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
        inside = int(((cache["r"]["transformation"][:3, :3] @ pc.get_xyz.T).T + T[:3, 3]).abs().lt(1).all(1).sum())
        # algunas etiquetas fuera de rango (no se mueven), como en los labels reales
        labels = torch.randint(-1, self.num_joints + 1, (inside,), generator=torch.Generator().manual_seed(1))

        initial = get_curr_link_states(self.robot)
        for j in range(self.num_joints):
            p.resetJointState(self.robot, j, 0.4 * (j + 1) * (-1) ** j)
        current = get_curr_link_states(self.robot)

        transforms = [tuple(torch.from_numpy(x).float() for x in compute_transformation(
            (*a["pos"], np.array(a["q"])), (*b["pos"], np.array(b["q"])))) for a, b in zip(initial, current)]
        segmented, xyz = get_segmented_indices(self.robot, pc, None, aabb, "r", robot_labels=labels,
                                               transformations_cache=cache)
        expected = transform_means(self.robot, pc, xyz, segmented, transforms, None, "r",
                                   transformations_cache=cache)

        new = RobotSplatTransform(self.robot, pc, labels, T, cache["r"]["transformation_scale"],
                                  cache["r"]["inv_transformation"], aabb)(initial, current)
        xyz_e, rot_e, op_e, rest_e, dc_e = expected
        xyz_n, rot_n, op_n, rest_n, dc_n = new
        self.assertLess((xyz_e - xyz_n).abs().max().item(), 1e-5)
        same_rotation = torch.minimum((rot_e - rot_n).abs().amax(1), (rot_e + rot_n).abs().amax(1))
        self.assertLess(same_rotation.max().item(), 1e-3)
        self.assertLess((rest_e - rest_n).abs().max().item(), 1e-4)
        self.assertTrue(torch.equal(op_e, op_n) and torch.equal(dc_e, dc_n))
        # los que no tienen link válido no se mueven (salvo el redondeo de ida y vuelta)
        self.assertGreater((xyz_n - pc.get_xyz).abs().amax(1).gt(1e-3).sum().item(), 0)


class FakeServer:
    path = "unused"
    record_state_only = False

    def __init__(self):
        self.t = 0.0

    def clock(self):
        return self.t


class TestThreadedRecording(unittest.TestCase):
    def test_trial_frames_match_synchronous_save(self):
        server = FakeServer()
        rec = TrajectoryRecorder(server, max_seconds=100)
        obs = [{"base_rgb": torch.rand(3, 8, 10), "wrist_rgb": torch.rand(3, 8, 10),
                "state": np.arange(7.0) + i} for i in range(5)]
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as ref:
            folder = rec.start(root=root, images_in_pkl=False)
            for o in obs:
                rec.on_command()
                rec.on_observation(o)
            rec.stop()
            for i, o in enumerate(obs, start=1):
                save_frame(ref, i, o, images_in_pkl=False)
            for name in sorted(os.listdir(ref)):
                self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
            for i in range(1, 6):
                with open(os.path.join(folder, f"{i:05d}.pkl"), "rb") as f:
                    got = pickle.load(f)
                self.assertEqual(set(got), {"state"})
                np.testing.assert_array_equal(got["state"], obs[i - 1]["state"])
                for cam in ("base_rgb", "wrist_rgb"):
                    a = Path(folder, "image_1", f"{cam}_{i:05d}.png").read_bytes()
                    b = Path(ref, "image_1", f"{cam}_{i:05d}.png").read_bytes()
                    self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
