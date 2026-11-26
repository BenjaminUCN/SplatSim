"""set_sampler y time_chunks de lerobot_diffusion_agent.py: DDIM de 16 pasos sobre un modelo de DDPM, sin reentrenar.

    python -m unittest discover -s tests -p "test_sampler.py"
Corre en CPU con un Diffusion Policy diminuto (solo estado), así no necesita la GPU ni un checkpoint.
"""
import unittest

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from splatsim.agents.lerobot_diffusion_agent import set_sampler, time_chunks


def tiny_policy() -> DiffusionPolicy:
    cfg = DiffusionConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        horizon=16, n_action_steps=8, down_dims=(16, 32), diffusion_step_embed_dim=8, device="cpu")
    return DiffusionPolicy(cfg).eval()


def batch():
    torch.manual_seed(0)
    return {"observation.state": torch.randn(1, 2, 7), "observation.environment_state": torch.randn(1, 2, 3)}


class SamplerTest(unittest.TestCase):
    def test_default_is_the_models_ddpm(self):
        policy = tiny_policy()
        self.assertEqual(set_sampler(policy), ("DDPM", 100))
        self.assertIsInstance(policy.diffusion.noise_scheduler, DDPMScheduler)

    def test_ddim_16_keeps_the_training_schedule(self):
        policy = tiny_policy()
        betas = policy.diffusion.noise_scheduler.betas.clone()
        self.assertEqual(set_sampler(policy, "DDIM", 16), ("DDIM", 16))
        scheduler = policy.diffusion.noise_scheduler
        self.assertIsInstance(scheduler, DDIMScheduler)
        torch.testing.assert_close(scheduler.betas, betas)   # mismo beta schedule que en el entrenamiento
        self.assertEqual(scheduler.config.prediction_type, "epsilon")
        self.assertEqual(policy.config.noise_scheduler_type, "DDIM")

    def test_ddim_samples_16_steps_and_is_timed(self):
        policy = tiny_policy()
        set_sampler(policy, "DDIM", 16)
        calls = []
        policy.diffusion.unet.register_forward_hook(lambda *_: calls.append(1))
        latencies = []
        time_chunks(policy, latencies)
        with torch.no_grad():
            actions = policy.diffusion.generate_actions(batch())
        self.assertEqual(len(calls), 16)                     # un paso de la U-Net por paso de DDIM
        self.assertEqual(tuple(actions.shape), (1, 8, 7))   # n_action_steps acciones
        self.assertEqual(len(latencies), 1)
        self.assertGreater(latencies[0], 0)

    def test_rejects_unknown_sampler_and_bad_steps(self):
        policy = tiny_policy()
        with self.assertRaises(ValueError):
            set_sampler(policy, "Euler")
        with self.assertRaises(ValueError):
            set_sampler(policy, "DDIM", 0)


if __name__ == "__main__":
    unittest.main()
