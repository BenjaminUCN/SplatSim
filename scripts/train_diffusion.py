"""Entrena una Diffusion Policy de LeRobot sobre un dataset de SplatSim (copiado de ~/lerobot/examples/tutorial/diffusion/diffusion_training_test_v2.py)."""

from datetime import datetime
from pathlib import Path

import torch
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

transforms = ImageTransforms(
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


def main():
    model_name = "diffusion_1021_1640"
    output_directory = Path(f"/home/magister/lerobot/examples/tutorial/diffusion/outputs/robot_learning_tutorial/{model_name}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    print(f"Setting optimizations for device: {device}")
    # Optimizaciones RTX 5080
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Autotuning para tamaños fijos de imagen

    user = "LuEduSoHu"
    dataset_id = f"{user}/splatsim_xarm5_lite_10fps_261020_2010"
    dataset_root = "/home/magister/lerobot/splatsim_test_data/splatsim_xarm5_lite_10fps_261020_2010"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id,root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)
    print("Setting memory format to channels_last for CNN optimizations")
    policy.to(memory_format=torch.channels_last)  # Optimizado para CNNs con imágenes

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # Instantiate the dataset
    dataset = LeRobotDataset(
        dataset_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        image_transforms=transforms
    )

    # Create the optimizer and dataloader for offline training
    print("Creating optimizer and dataloader with optimized settings for RTX 5080")
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    scaler = GradScaler()
    batch_size = 32  # Aumentado para aprovechar la VRAM de la RTX 5080
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,           # Seguro con 56GB en WSL (16 causaba OOM)
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3,        # Buen balance velocidad/RAM con 2 cámaras
        persistent_workers=True,
    )

    # Number of training steps and logging frequency
    training_steps = 50000
    log_freq = 1  # Loggear menos frecuente reduce overhead
    save_freq = 10000

    checkpoint_root = Path(output_directory) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Run training loop
    start_time = datetime.now()
    print(f"[{start_time.strftime('%y%m%d %H:%M')}] Starting training loop")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)

            # print("Moving batch to device with optimized settings")
            # Mixed precision con bfloat16 (óptimo para RTX 50xx)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)

            # print("Performing backward pass and optimizer step with mixed precision")
            # print("scaler.scale(loss).backward()")
            scaler.scale(loss).backward()
            # print("scaler.step(optimizer)")
            scaler.step(optimizer)
            # print("scaler.update()")
            scaler.update()
            # print("optimizer.zero_grad(set_to_none=True)")
            optimizer.zero_grad(set_to_none=True)  # Más eficiente en memoria

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            # Save checkpoint
            if step > 0 and step % save_freq == 0:
                checkpoint_dir = (
                    checkpoint_root
                    / f"step_{step:06d}_loss_{loss.item():.4f}"
                )

                print(f"Saving checkpoint to {checkpoint_dir}")

                policy.save_pretrained(checkpoint_dir / "policy")
                preprocessor.save_pretrained(checkpoint_dir / "preprocessor")
                postprocessor.save_pretrained(checkpoint_dir / "postprocessor")

            step += 1
            if step >= training_steps:
                done = True
                break

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time of start: {start_time.strftime('%y%m%d %H:%M')}")
    print(f"Time of end: {end_time.strftime('%y%m%d %H:%M')}")
    print(f"\n Training time: {duration}")

    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    policy.push_to_hub(f"{user}/robot_learning_tutorial_{model_name}")
    preprocessor.push_to_hub(f"{user}/robot_learning_tutorial_{model_name}")
    postprocessor.push_to_hub(f"{user}/robot_learning_tutorial_{model_name}")






if __name__ == "__main__":
    main()