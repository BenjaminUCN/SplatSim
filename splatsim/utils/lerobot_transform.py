"""
lerobot_transform.py
=========================

Convert SplatSim demonstrations into a VALID LeRobot v3 dataset
using the OFFICIAL LeRobot dataset writer API.

This version:
- uses latest LeRobot API
- creates REAL v3 datasets
- dynamically handles:
    - arbitrary image resolutions
    - arbitrary image keys
    - arbitrary DoFs
- supports:
    - base_rgb
    - wrist_rgb
    - any future cameras
- preserves original image resolution
- stores videos properly
- computes dataset stats

Requirements
------------
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

pip install imageio imageio-ffmpeg

Usage
-----
python splatsim_to_lerobot_v3.py \
    --input_dir ~/SplatSim/bc_data/gello \
    --output_dir ~/datasets/splatsim_v3
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# LeRobot imports
# ============================================================

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================


@dataclass
class ConversionConfig:
    input_dir: Path
    output_dir: Path

    fps: int = 10

    robot_type: str = "lerobot_splatsim"

    wrist_camera: bool = True
    wrist_offset: int | None = None
    use_videos: bool = True


# ============================================================
# Utils
# ============================================================


def safe_to_numpy(obj: Any) -> np.ndarray:

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)

    if np.isscalar(obj):
        return np.asarray([obj])

    raise TypeError(f"Unsupported type: {type(obj)}")


def load_pickle(path: Path) -> dict:

    with open(path, "rb") as f:

        try:
            data = pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")

    return data


def load_image(path: Path) -> np.ndarray:
    """
    Returns:
        uint8 CHW image
    """

    img = Image.open(path).convert("RGB")

    arr = np.asarray(img, dtype=np.uint8)

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    return arr


def find_image(image_dir: Path, prefix: str, index: int) -> Path | None:

    candidates = [
        f"{prefix}_{index:05d}.png",
        f"{prefix}_{index:04d}.png",
        f"{prefix}_{index}.png",
    ]

    for c in candidates:
        p = image_dir / c

        if p.exists():
            return p

    return None


def discover_episodes(input_dir: Path) -> list[Path]:

    episodes = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and any(d.glob("*.pkl"))
    )

    if not episodes:
        raise RuntimeError(f"No episodes found in {input_dir}")

    return episodes


def detect_wrist_offset(
    episode_dir: Path,
    num_samples: int = 5,
) -> int | None:

    image_dir = episode_dir / "image_1"

    if not image_dir.exists():
        return None

    pkls = sorted(episode_dir.glob("*.pkl"))[:num_samples]

    matches = []

    for pkl_path in pkls:

        idx = int(pkl_path.stem)

        if find_image(image_dir, "wrist_rgb", idx):
            matches.append(0)

        elif find_image(image_dir, "wrist_rgb", idx + 1):
            matches.append(1)

    if not matches:
        return None

    if all(m == matches[0] for m in matches):
        return matches[0]

    return None


# ============================================================
# Dynamic feature inference
# ============================================================


def infer_image_keys(
    episode_dir: Path,
    wrist_camera: bool,
) -> list[str]:

    image_dir = episode_dir / "image_1"

    keys = []

    if list(image_dir.glob("base_rgb*.png")):
        keys.append("base_rgb")

    if wrist_camera and list(image_dir.glob("wrist_rgb*.png")):
        keys.append("wrist_rgb")

    if not keys:
        raise RuntimeError(f"No image streams found in {image_dir}")

    return keys


def infer_image_shape(
    episode_dir: Path,
    image_key: str,
) -> tuple[int, int, int]:

    image_dir = episode_dir / "image_1"

    img_path = next(
        image_dir.glob(f"{image_key}*.png")
    )

    img = Image.open(img_path).convert("RGB")

    w, h = img.size

    # CHW
    return (3, h, w)


def infer_dofs(episode_dir: Path) -> tuple[int, int]:

    pkl_path = sorted(
        episode_dir.glob("*.pkl"),
        key=lambda p: int(p.stem)
    )[0]

    data = load_pickle(pkl_path)

    joint_positions = safe_to_numpy(
        data["joint_positions"]
    ).flatten()

    action = safe_to_numpy(
        data["action"]
    ).flatten()

    return len(joint_positions), len(action)


# ============================================================
# LeRobot feature spec
# ============================================================


def build_lerobot_features(
    image_keys: list[str],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
    frame_type: str = "video",
):

    features = {}

    for key in image_keys:

        features[f"observation.images.{key}"] = {
            "dtype": frame_type,
            "shape": image_shapes[key],
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"state_{i}" for i in range(state_dim)],
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"action_{i}" for i in range(action_dim)],
    }

    return features


# ============================================================
# Dataset creation
# ============================================================


def create_dataset(
    config: ConversionConfig,
    image_keys: list[str],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
):

    features = build_lerobot_features(
        image_keys=image_keys,
        image_shapes=image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
        frame_type="video" if config.use_videos else "image",
    )

    dataset = LeRobotDataset.create(
        repo_id="LuEduSoHu/25051601",
        root=config.output_dir,
        robot_type=config.robot_type,
        fps=config.fps,
        features=features,
        use_videos=config.use_videos,
    )

    return dataset

def push_lerobot_to_hub(dataset: "LeRobotDataset") -> None:
    """Push dataset to hub, retrying with a new repo_id on failure.

    Loops until the push succeeds or the user presses Enter to skip.
    """
    while True:
        repo_id = dataset.repo_id
        print(f"[LeRobot] Pushing dataset to hub as '{repo_id}'...")
        try:
            dataset.push_to_hub()
            print(f"[LeRobot] Successfully pushed to hub as '{repo_id}'.")
            return
        except Exception as e:
            print(f"[LeRobot] ERROR: Failed to push to hub: {e}")
            print("[LeRobot] Repo ID should be in 'username/dataset_name' format.")
            print("[LeRobot] Make sure you are authenticated with `huggingface-cli login`.")
            new_repo_id = input("[LeRobot] Enter a new repo_id to retry (or press Enter to skip): ").strip()
            if not new_repo_id:
                print("[LeRobot] Skipping push to hub. Dataset is saved locally.")
                return
            dataset.repo_id = new_repo_id

# ============================================================
# Frame builder
# ============================================================


def build_frame(
    pkl_data: dict,
    images: dict[str, np.ndarray],
    image_keys: list[str],
    task: str = "",
) -> dict:
    """
    Build a frame compatible with LeRobotDataset.add_frame().

    Expected conventions:
    - observation.state -> float32
    - action            -> float32
    - images            -> float32 CHW in [0, 1]

    Args:
        pkl_data:
            Dict containing:
                - "joint_positions"
                - "action"

        images:
            Dict[str, np.ndarray]
            Raw uint8 CHW or HWC images.

        image_keys:
            Keys to include in frame.

        task:
            Optional task description.
    """

    state = safe_to_numpy(
        pkl_data["joint_positions"]
    ).astype(np.float32).flatten()

    action = safe_to_numpy(
        pkl_data["action"]
    ).astype(np.float32).flatten()

    frame: dict = {
        "observation.state": state,
        "action": action,
        "task": task,
    }

    for key in image_keys:
        if key not in images:
            continue

        img = images[key]
        img = np.asarray(img)  # Ensure numpy

        # Convert HWC -> CHW if necessary
        if img.ndim != 3:
            raise ValueError(f"Image '{key}' must be 3D, got shape {img.shape}")

        # HWC -> CHW
        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))

        # Convert uint8 -> float32 [0,1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        # Final validation
        if img.shape[0] != 3:
            raise ValueError(f"Image '{key}' must be CHW with 3 channels, got {img.shape}")

        frame[f"observation.images.{key}"] = img

    return frame


# ============================================================
# Episode processing
# ============================================================


def process_episode(
    dataset: LeRobotDataset,
    episode_dir: Path,
    image_keys: list[str],
    config: ConversionConfig,
):

    image_dir = episode_dir / "image_1"

    pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    if not pkl_files:
        return

    wrist_offset = config.wrist_offset

    if "wrist_rgb" in image_keys and wrist_offset is None:
        detected = detect_wrist_offset(episode_dir)

        wrist_offset = (
            detected
            if detected is not None
            else 0
        )
        # print(f"Using wrist camera offset: {wrist_offset}")

    episode_frames = []

    for pkl_path in pkl_files:
        idx = int(pkl_path.stem)

        try:
            pkl_data = load_pickle(pkl_path)
        except Exception as e:
            warnings.warn(f"Failed loading {pkl_path}: {e}")
            continue

        images = {}
        valid = True

        for key in image_keys:
            image_idx = idx

            if key == "wrist_rgb":
                image_idx += wrist_offset

            img_path = find_image(image_dir, key, image_idx)

            if img_path is None:
                valid = False
                print(f"Missing image for key '{key}' at index {image_idx} in {image_dir}")
                break

            try:
                images[key] = load_image(img_path)
            except Exception as e:
                warnings.warn(f"Failed image {img_path}: {e}")

                valid = False
                break

        if not valid:
            continue

        try:
            frame = build_frame(pkl_data, images, images.keys(), task="aproach_apple") 
            # print(f"Built frame for index {idx} with keys: {list(frame.keys())}")
            # episode_frames.append(frame)
            dataset.add_frame(frame)
            # print(f"Added frame for index {idx} to dataset.")
        except Exception as e:
            warnings.warn(f"Failed building frame: {e}")

    # if not episode_frames:
    #     warnings.warn(f"No valid frames in {episode_dir}")
    #     return

    dataset.save_episode()


# ============================================================
# Main conversion
# ============================================================


def run_conversion(config: ConversionConfig):

    print("=" * 70)
    print("SplatSim -> LeRobot v3")
    print("=" * 70)

    episode_dirs = discover_episodes(config.input_dir)

    first_episode = episode_dirs[0]

    image_keys = infer_image_keys(first_episode, config.wrist_camera)

    image_shapes = {
        k: infer_image_shape(first_episode, k)
        for k in image_keys
    }

    state_dim, action_dim = infer_dofs(
        first_episode
    )

    print("\nDetected configuration:")
    print(f"Image keys : {image_keys}")

    for k, shape in image_shapes.items():
        print(f"{k:12s}: {shape}")

    print(f"State dim  : {state_dim}")
    print(f"Action dim : {action_dim}")

    dataset = create_dataset(
        config=config,
        image_keys=image_keys,
        image_shapes=image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    print(f"\nFound {len(episode_dirs)} episodes\n")

    for ep_dir in tqdm(episode_dirs):

        print(f"\nProcessing {ep_dir.name}")

        process_episode(
            dataset=dataset,
            episode_dir=ep_dir,
            image_keys=image_keys,
            config=config,
        )

    print("\nFinalizing dataset...")

    try:
        dataset.clear_episode_buffer()
    except Exception:
        pass
    dataset.finalize()

    print("\nDone.")
    print(
        f"\nDataset saved to:\n"
        f"{config.output_dir.resolve()}"
    )
    
    push_lerobot_to_hub(dataset)


# ============================================================
# CLI
# ============================================================


def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--robot_type", type=str, default="lerobot_splatsim")
    p.add_argument("--wrist_offset", type=int, default=None, choices=[0, 1])
    p.add_argument("--no_wrist_camera", action="store_true")
    p.add_argument("--use_images", action="store_true")

    return p


def main():

    args = build_parser().parse_args()

    config = ConversionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,

        fps=args.fps,
        robot_type=args.robot_type,

        wrist_offset=args.wrist_offset,
        wrist_camera=not args.no_wrist_camera,
        use_videos=not args.use_images,
    )
    
    print(f"args.use_images = {args.use_images}")
    print(f"config.use_videos = {config.use_videos}")
    print("TERMINATING!!!!!!!")
    exit()

    run_conversion(config)


if __name__ == "__main__":
    main()