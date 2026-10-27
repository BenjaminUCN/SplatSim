"""Entrena una Diffusion Policy de LeRobot sobre un dataset de SplatSim.

Los parámetros se leen de un YAML y cualquiera se puede sobrescribir por CLI:
    python scripts/train_diffusion.py --config configs/train/diffusion.yaml
    python scripts/train_diffusion.py --config configs/train/diffusion.yaml --horizon 32 --cameras base_rgb

La configuración final se guarda en <output_root>/<model_name>/train_config.yaml.
"""

import argparse
import sys
from dataclasses import asdict, dataclass, fields
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

from splatsim.utils.camera_setup import arrangement_cameras

DEFAULT_CONFIG = "configs/train/diffusion.yaml"


@dataclass
class TrainConfig:
    # run
    model_name: str | None = None      # None -> diffusion_<MMDD_HHMM> (carpeta de salida)
    name: str = ""                     # nombre corto para configs/registry.yaml
    description: str = ""              # qué intenta mejorar este modelo
    output_root: str = "/home/magister/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial"
    device: str = "cuda"
    # dataset
    dataset_id: str = "LuEduSoHu/splatsim_xarm5_lite_10fps_261020_2010"
    dataset_root: str = "/home/magister/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010"
    cameras: tuple[str, ...] | None = None  # subconjunto de cámaras del dataset; None -> todas
    arrangement: str | None = None      # arreglo de configs/camera_poses.yaml (en vez de cameras)
    # policy
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32
    # optimización
    training_steps: int = 50000
    batch_size: int = 32
    num_workers: int = 12               # 16 causaba OOM con 56GB en WSL
    prefetch_factor: int = 3
    image_augmentation: bool = True
    # logging / guardado
    log_freq: int = 1
    save_freq: int = 10000
    push_to_hub: bool = True
    hub_user: str = "LuEduSoHu"


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
    return tyro.cli(TrainConfig, default=TrainConfig(**values), args=rest)


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
    )
    print(f"Cameras: {list(policy_cfg.image_features)}")
    policy = DiffusionPolicy(policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)
    print("Setting memory format to channels_last for CNN optimizations")
    policy.to(memory_format=torch.channels_last)  # Optimizado para CNNs con imágenes

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
    while not done:
        for batch in dataloader:
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

            # Save checkpoint
            if step > 0 and step % cfg.save_freq == 0:
                checkpoint_dir = checkpoint_root / f"step_{step:06d}_loss_{loss.item():.4f}"

                print(f"Saving checkpoint to {checkpoint_dir}")

                policy.save_pretrained(checkpoint_dir / "policy")
                preprocessor.save_pretrained(checkpoint_dir / "preprocessor")
                postprocessor.save_pretrained(checkpoint_dir / "postprocessor")

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


if __name__ == "__main__":
    main(load_config(sys.argv[1:]))
