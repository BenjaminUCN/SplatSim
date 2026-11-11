"""Entrena una Diffusion Policy de LeRobot sobre un dataset de SplatSim.

Los parámetros se leen de un YAML y cualquiera se puede sobrescribir por CLI:
    python scripts/xarm5/train_diffusion.py --config configs/train/diffusion.yaml
    python scripts/xarm5/train_diffusion.py --config configs/train/diffusion.yaml --horizon 32 --cameras base_rgb

La configuración final se guarda en <output_root>/<model_name>/train_config.yaml.

Con wandb: true (por defecto) cada entrenamiento es una corrida de Weights & Biases en
--wandb-project: la config, la pérdida, el learning rate y la velocidad cada --wandb-log-freq pasos,
y los checkpoints guardados. Hace falta haber hecho `wandb login` (o WANDB_API_KEY); si no, la
corrida se guarda offline en <output_root>/<model_name>/wandb y se sube después con
`wandb sync <esa carpeta>/offline-run-*`. La URL de la corrida queda en train_config.yaml.
"""

import argparse
import netrc
import os
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import torch
import tyro
import yaml
from torch.amp import GradScaler, autocast

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

from splatsim.utils.data.relative_actions import to_relative
from splatsim.utils.render.camera_setup import arrangement_cameras
from splatsim.utils.paths import expand

DEFAULT_CONFIG = "configs/train/diffusion.yaml"


@dataclass
class TrainConfig:
    # run
    model_name: str | None = None      # None -> diffusion_<MMDD_HHMM> (carpeta de salida)
    name: str = ""                     # nombre corto para configs/registry.yaml
    description: str = ""              # qué intenta mejorar este modelo
    output_root: str = "~/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial"
    device: str = "cuda"
    # dataset
    dataset_id: str = "LuEduSoHu/splatsim_xarm5_lite_10fps_261020_2010"
    dataset_root: str = "~/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010"
    cameras: tuple[str, ...] | None = None  # subconjunto de cámaras del dataset; None -> todas
    arrangement: str | None = None      # arreglo de configs/camera_poses.yaml (en vez de cameras)
    # policy
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32
    # (alto, ancho) al que el modelo reescala las imágenes antes de la ResNet; None -> tamaño del dataset.
    # Entrena más rápido, y el agente le sigue pasando las del sim tal cual. Con un dataset ya reescalado
    # (scripts/xarm5/resize_dataset.py) se pone el mismo tamaño, para que el modelo reescale las del sim
    resize_shape: tuple[int, int] | None = None
    # optimización
    training_steps: int = 50000
    batch_size: int = 32
    num_workers: int = 12               # 16 causaba OOM con 56GB en WSL
    prefetch_factor: int = 3
    image_augmentation: bool = True
    # acciones relativas a la pose del estado en el último paso observado (pos y euler; el gripper
    # queda absoluto). Ver splatsim/utils/data/relative_actions.py
    relative_actions: bool = False
    torch_compile: bool = False         # compila la U-Net y los encoders (no cambia el modelo guardado)
    # logging / guardado
    log_freq: int = 1
    save_freq: int = 10000
    push_to_hub: bool = True
    hub_user: str = "LuEduSoHu"
    # Weights & Biases
    wandb: bool = True
    wandb_project: str = "splatsim-diffusion"
    wandb_entity: str | None = None     # None -> la entidad por defecto de la cuenta
    wandb_mode: str = "online"          # online / offline / disabled
    wandb_log_freq: int = 10            # pasos entre registros


def load_config(argv: list[str]) -> TrainConfig:
    """Lee --config (YAML) y aplica encima los demás argumentos de la CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    known, rest = parser.parse_known_args(argv)

    with open(known.config) as f:
        values = yaml.safe_load(f) or {}
    unknown = set(values) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise SystemExit(f"{known.config}: campos desconocidos {sorted(unknown)}")
    if values.get("cameras") is not None:
        values["cameras"] = tuple(values["cameras"])

    print(f"Config: {known.config}")
    cfg = tyro.cli(TrainConfig, default=TrainConfig(**values), args=rest)
    return replace(cfg, output_root=expand(cfg.output_root), dataset_root=expand(cfg.dataset_root))


def make_image_transforms() -> ImageTransforms:
    return ImageTransforms(
        ImageTransformsConfig(
            enable=True,
            max_num_transforms=2,
            random_order=True,
            tfs={
                "brightness": ImageTransformConfig(
                    weight=3.0,
                    type="ColorJitter",
                    kwargs={"brightness": (0.8, 1.2)},
                ),
                "contrast": ImageTransformConfig(
                    weight=3.0,
                    type="ColorJitter",
                    kwargs={"contrast": (0.8, 1.2)},
                ),
                "erasing": ImageTransformConfig(
                    weight=1.0,
                    type="RandomErasing",
                    kwargs={
                        "p": 1.0,
                        "scale": (0.02, 0.2),
                        "ratio": (0.3, 3.3),
                        "value": "random",
                    },
                ),
                "noise": ImageTransformConfig(
                    weight=2.0,
                    type="GaussianNoise",
                    kwargs={
                        "mean": 0.0,
                        "sigma": 0.02,
                    },
                ),
            },
        )
    )


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def select_cameras(input_features: dict, cameras: tuple[str, ...] | None) -> dict:
    """Deja solo las imágenes observation.images.<cam> de `cameras` (None -> todas)."""
    if cameras is None:
        return input_features
    prefix = "observation.images."
    available = [k.removeprefix(prefix) for k in input_features if k.startswith(prefix)]
    missing = set(cameras) - set(available)
    if missing:
        raise SystemExit(f"Cámaras {sorted(missing)} no están en el dataset (hay {available})")
    return {k: ft for k, ft in input_features.items()
            if not k.startswith(prefix) or k.removeprefix(prefix) in cameras}


def wandb_logged_in() -> bool:
    """Hay API key de W&B (variable de entorno o ~/.netrc de `wandb login`)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def start_wandb(cfg: TrainConfig, model_name: str, output_directory: Path, extra: dict):
    """Corrida de W&B o None (wandb: false o sin el paquete). Sin login pasa a offline."""
    if not cfg.wandb or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[W&B] Falta el paquete wandb (pip install wandb); se entrena sin registrar")
        return None
    mode = cfg.wandb_mode
    if mode == "online" and not wandb_logged_in():
        mode = "offline"
        print("[W&B] Sin `wandb login` (ni WANDB_API_KEY): la corrida se guarda offline en "
              f"{output_directory / 'wandb'}; súbela después con `wandb sync`")
    run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=model_name,
                     notes=cfg.description or None, tags=[cfg.name] if cfg.name else None,
                     dir=str(output_directory), mode=mode,
                     config={**asdict(cfg), "model_name": model_name, **extra})
    print(f"[W&B] {run.url or f'offline: {run.dir}'}")
    return run


def main(cfg: TrainConfig):
    model_name = cfg.model_name or datetime.now().strftime("diffusion_%m%d_%H%M")
    output_directory = Path(cfg.output_root) / model_name
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "train_config.yaml", "w") as f:
        yaml.safe_dump({**asdict(cfg), "model_name": model_name,
                        "cameras": list(cfg.cameras) if cfg.cameras else None}, f, sort_keys=False)
    print(f"Output: {output_directory}")

    device = torch.device(cfg.device)

    print(f"Setting optimizations for device: {device}")
    # Optimizaciones RTX 5080
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Autotuning para tamaños fijos de imagen

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(cfg.dataset_id, root=cfg.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    if cfg.arrangement and cfg.cameras:
        raise SystemExit("Usa cameras o arrangement, no ambos")
    cameras = tuple(arrangement_cameras(cfg.arrangement)) if cfg.arrangement else cfg.cameras
    input_features = select_cameras(input_features, cameras)

    policy_cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        resize_shape=tuple(cfg.resize_shape) if cfg.resize_shape else None,
    )
    print(f"Cameras: {list(policy_cfg.image_features)}")
    policy = DiffusionPolicy(policy_cfg)
    stats = dataset_metadata.stats
    if cfg.relative_actions:
        from splatsim.utils.data.relative_actions import relative_action_stats, save_stats
        ref = policy_cfg.n_obs_steps - 1     # índice del frame de referencia en observation.state
        offsets = [i - (ref + policy_cfg.observation_delta_indices[0]) for i in policy_cfg.action_delta_indices]
        action_stats = relative_action_stats(cfg.dataset_root, offsets, stats["action"])
        save_stats(action_stats, output_directory)
        stats = {**stats, "action": action_stats}
        print(f"Acciones relativas: offsets {offsets[0]}..{offsets[-1]}, "
              f"min {action_stats['min'][:6].round(3)}, max {action_stats['max'][:6].round(3)}")
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=stats)

    policy.train()
    policy.to(device)
    print("Setting memory format to channels_last for CNN optimizations")
    policy.to(memory_format=torch.channels_last)  # Optimizado para CNNs con imágenes
    if cfg.torch_compile:
        # module.compile() en vez de torch.compile(module) o compile_model de DiffusionConfig: esos
        # agregan _orig_mod. a las claves guardadas. Se compilan las submódulos porque aquí se
        # llama a policy.forward directamente, sin pasar por el __call__ que compile() reemplaza.
        encoders = policy.diffusion.rgb_encoder
        encoders = encoders if isinstance(encoders, torch.nn.ModuleList) else [encoders]
        for module in (policy.diffusion.unet, *encoders):
            module.compile()

    # To perform action chunking, the policy expects a given number of actions as targets
    delta_timestamps = {
        "observation.state": make_delta_timestamps(policy_cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(policy_cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(policy_cfg.observation_delta_indices, dataset_metadata.fps)
        for k in policy_cfg.image_features
    }

    # Instantiate the dataset
    dataset = LeRobotDataset(
        cfg.dataset_id,
        root=cfg.dataset_root,
        delta_timestamps=delta_timestamps,
        image_transforms=make_image_transforms() if cfg.image_augmentation else None,
    )

    run = start_wandb(cfg, model_name, output_directory, {
        "cameras_used": list(policy_cfg.image_features), "dataset_episodes": dataset.num_episodes,
        "dataset_frames": dataset.num_frames, "fps": dataset_metadata.fps})
    if run is not None:
        with open(output_directory / "train_config.yaml", "a") as f:
            yaml.safe_dump({"wandb_run": run.url or run.dir, "wandb_id": run.id}, f, sort_keys=False)

    # Create the optimizer and dataloader for offline training
    print("Creating optimizer and dataloader")
    optimizer = policy_cfg.get_optimizer_preset().build(policy.parameters())
    scaler = GradScaler()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True,
    )

    checkpoint_root = output_directory / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Run training loop
    start_time = datetime.now()
    print(f"[{start_time.strftime('%y%m%d %H:%M')}] Starting training loop")
    step = 0
    done = False
    last_log = (time.monotonic(), 0)    # para la velocidad entre registros de W&B
    while not done:
        for batch in dataloader:
            if cfg.relative_actions:
                batch["action"] = to_relative(batch["action"], batch["observation.state"], policy_cfg.n_obs_steps - 1)
            batch = preprocessor(batch)

            # Mixed precision con bfloat16 (óptimo para RTX 50xx)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Más eficiente en memoria

            if step % cfg.log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            if run is not None and step % cfg.wandb_log_freq == 0:
                now = time.monotonic()
                rate = (step - last_log[1]) / max(now - last_log[0], 1e-6)
                last_log = (now, step)
                run.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"],
                         "train/steps_per_s": rate}, step=step)

            # Save checkpoint
            if step > 0 and step % cfg.save_freq == 0:
                checkpoint_dir = checkpoint_root / f"step_{step:06d}_loss_{loss.item():.4f}"

                print(f"Saving checkpoint to {checkpoint_dir}")

                policy.save_pretrained(checkpoint_dir / "policy")
                preprocessor.save_pretrained(checkpoint_dir / "preprocessor")
                postprocessor.save_pretrained(checkpoint_dir / "postprocessor")
                if run is not None:
                    run.log({"checkpoint/loss": loss.item()}, step=step)

            step += 1
            if step >= cfg.training_steps:
                done = True
                break

    end_time = datetime.now()
    print(f"Time of start: {start_time.strftime('%y%m%d %H:%M')}")
    print(f"Time of end: {end_time.strftime('%y%m%d %H:%M')}")
    print(f"\n Training time: {end_time - start_time}")

    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if cfg.push_to_hub:
        repo_id = f"{cfg.hub_user}/robot_learning_tutorial_{model_name}"
        policy.push_to_hub(repo_id)
        preprocessor.push_to_hub(repo_id)
        postprocessor.push_to_hub(repo_id)
    if run is not None:
        run.summary.update({"training_minutes": round((end_time - start_time).total_seconds() / 60, 1),
                            "final_loss": loss.item(), "model_dir": str(output_directory)})
        run.finish()


if __name__ == "__main__":
    main(load_config(sys.argv[1:]))
