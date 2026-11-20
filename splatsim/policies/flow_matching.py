"""Política de flujo (flow matching, rectified flow) con la misma red que la Diffusion Policy de LeRobot.

La U-Net condicional, los encoders de imágenes, la normalización y el guardado son los de DiffusionPolicy; solo
cambian la pérdida y el muestreo:
  - entrenamiento: x_t = (1 - t) ruido + t acción con t ~ U(0, 1), y la red predice la velocidad acción - ruido;
  - inferencia: desde el ruido (t = 0) hasta la acción (t = 1) con num_inference_steps pasos de Euler.
La red recibe t * TIME_SCALE como "paso de difusión", la misma escala que los 100 pasos de entrenamiento de la
difusión, para que el embedding sinusoidal se use igual.

train_diffusion.py --objective flow lo aplica al modelo nuevo y lo anota en train_config.yaml; el agente
(LeRobotDiffusionAgent) lo lee de ahí y lo aplica al cargar. El config.json guardado sigue siendo un
DiffusionConfig, con num_inference_steps = los pasos de Euler.
"""
import types

import torch
import torch.nn.functional as F  # noqa: N812

OBJECTIVES = ("diffusion", "flow")
TIME_SCALE = 1000.0
DEFAULT_STEPS = 10


def flow_loss(model, batch: dict) -> torch.Tensor:
    """Reemplazo de DiffusionModel.compute_loss: error de la velocidad predicha (mismo enmascarado del relleno)."""
    from lerobot.utils.constants import ACTION
    global_cond = model._prepare_global_conditioning(batch)
    action = batch[ACTION]
    noise = torch.randn_like(action)
    t = torch.rand(action.shape[0], device=action.device, dtype=action.dtype)
    x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * action
    pred = model.unet(x_t, t * TIME_SCALE, global_cond=global_cond)
    loss = F.mse_loss(pred, action - noise, reduction="none")
    if model.config.do_mask_loss_for_padding:
        mask = (~batch["action_is_pad"]).unsqueeze(-1)
        return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
    return loss.mean()


def flow_sample(model, batch_size: int, global_cond=None, generator=None, noise=None) -> torch.Tensor:
    """Reemplazo de DiffusionModel.conditional_sample: Euler de t = 0 (ruido) a t = 1 (acción)."""
    if noise is None:
        p = next(model.parameters())
        shape = (batch_size, model.config.horizon, model.config.action_feature.shape[0])
        noise = torch.randn(shape, dtype=p.dtype, device=p.device, generator=generator)
    x = noise
    n = model.num_inference_steps
    for k in range(n):
        t = torch.full((batch_size,), k / n, dtype=x.dtype, device=x.device)
        x = x + model.unet(x, t * TIME_SCALE, global_cond=global_cond) / n
    if model.config.clip_sample:
        x = x.clamp(-model.config.clip_sample_range, model.config.clip_sample_range)
    return x


def use_flow(policy, steps: int | None = None) -> int:
    """Convierte un DiffusionPolicy (nuevo o cargado) en política de flujo. steps: pasos de Euler (None: los
    del config, o DEFAULT_STEPS si no los fija). Devuelve los pasos."""
    model = policy.diffusion
    model.compute_loss = types.MethodType(flow_loss, model)
    model.conditional_sample = types.MethodType(flow_sample, model)
    steps = steps or policy.config.num_inference_steps or DEFAULT_STEPS
    if steps < 1:
        raise ValueError("los pasos de Euler deben ser >= 1")
    policy.config.num_inference_steps = steps
    model.num_inference_steps = steps
    return steps


def checkpoint_objective(checkpoint: str) -> str:
    """objective de train_config.yaml del modelo ("diffusion" si no está o no hay archivo, p. ej. del Hub)."""
    import os

    import yaml
    path = os.path.join(checkpoint, "train_config.yaml")
    if not os.path.isfile(path):
        return "diffusion"
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("objective", "diffusion")
