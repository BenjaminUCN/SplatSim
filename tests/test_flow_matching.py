"""Política de flujo con la red de la Diffusion Policy (splatsim/policies/flow_matching.py), en CPU y sin imágenes.

    python -m unittest discover -s tests -p 'test_flow_matching.py'
"""
import os
import tempfile
import unittest

import torch

from splatsim.policies.flow_matching import TIME_SCALE, checkpoint_objective, use_flow


class Fn(torch.nn.Module):
    """Una función como U-Net (nn.Module no acepta otra cosa en .unet)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, t, global_cond=None):
        return self.fn(x, t, global_cond)


def tiny_policy(horizon: int = 8):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
    cfg = DiffusionConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (2,)),
                        "observation.environment_state": PolicyFeature(FeatureType.ENV, (2,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        n_obs_steps=2, horizon=horizon, n_action_steps=4, down_dims=(16, 32), kernel_size=3, n_groups=4,
        diffusion_step_embed_dim=16, device="cpu")
    return DiffusionPolicy(cfg)


def batch(n: int, target: torch.Tensor, horizon: int = 8) -> dict:
    return {"observation.state": torch.zeros(n, 2, 2), "observation.environment_state": torch.zeros(n, 2, 2),
            "action": target.expand(n, horizon, 2).clone(), "action_is_pad": torch.zeros(n, horizon, dtype=torch.bool)}


class FlowTest(unittest.TestCase):
    def test_euler_follows_a_straight_path_exactly(self):
        policy = tiny_policy()
        self.assertEqual(use_flow(policy, 5), 5)
        model = policy.diffusion
        noise = torch.randn(3, 8, 2)
        target = torch.full((3, 8, 2), 0.4)
        times = []

        def unet(x, t, global_cond=None):
            times.append(t[0].item())
            return target - noise                     # velocidad constante del camino recto
        model.unet = Fn(unet)
        out = model.conditional_sample(3, noise=noise)
        torch.testing.assert_close(out, target)
        self.assertEqual(times, [k / 5 * TIME_SCALE for k in range(5)])

    def test_loss_is_the_velocity_error(self):
        policy = tiny_policy()
        use_flow(policy, 4)
        model = policy.diffusion
        target = torch.tensor([0.3, -0.2])
        seen = {}

        def unet(x, t, global_cond=None):
            seen["x"], seen["t"] = x, t / TIME_SCALE
            return torch.zeros_like(x)
        model.unet = Fn(unet)
        torch.manual_seed(0)
        loss = model.compute_loss(batch(4, target))
        # x_t = (1 - t) ruido + t acción, así que la velocidad es (acción - x_t) / (1 - t)
        t = seen["t"][:, None, None]
        velocity = (target - seen["x"]) / (1 - t)
        self.assertAlmostEqual(loss.item(), (velocity ** 2).mean().item(), places=4)

    def test_padding_is_masked(self):
        policy = tiny_policy()
        use_flow(policy)
        model = policy.diffusion
        model.unet = Fn(lambda x, t, global_cond=None: torch.zeros_like(x))
        b = batch(2, torch.tensor([0.0, 0.0]))
        b["action_is_pad"][:, 4:] = True
        b["action"][:, 4:] = 100.0                    # relleno
        self.assertGreater(model.compute_loss(b).item(), 50)   # como la difusión: por defecto cuenta
        policy.config.do_mask_loss_for_padding = True
        self.assertLess(model.compute_loss(b).item(), 50)

    def test_a_tiny_model_learns_a_constant_action(self):
        torch.manual_seed(0)
        policy = tiny_policy()
        use_flow(policy, 8)
        target = torch.tensor([0.5, -0.3])
        opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
        for _ in range(300):
            loss, _ = policy.forward(batch(16, target))
            opt.zero_grad()
            loss.backward()
            opt.step()
        policy.eval()
        with torch.no_grad():
            out = policy.diffusion.conditional_sample(
                8, global_cond=policy.diffusion._prepare_global_conditioning(batch(8, target)))
        self.assertLess((out - target).abs().mean().item(), 0.1)

    def test_steps_default_and_objective_file(self):
        policy = tiny_policy()
        self.assertEqual(use_flow(policy), 10)
        self.assertEqual(policy.config.num_inference_steps, 10)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(checkpoint_objective(d), "diffusion")
            with open(os.path.join(d, "train_config.yaml"), "w") as f:
                f.write("objective: flow\nflow_steps: 10\n")
            self.assertEqual(checkpoint_objective(d), "flow")


if __name__ == "__main__":
    unittest.main()
