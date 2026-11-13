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
import copy
import math
import netrc
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
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
    # Encoder de imágenes. Por defecto (LeRobot desde may-2026) ResNet18 de ImageNet con BatchNorm;
    # Diffusion Policy lo entrena desde cero con GroupNorm: pretrained_backbone: null, group_norm: true
    pretrained_backbone: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    group_norm: bool = False
    # optimización
    seed: int | None = None             # None -> otra corrida cada vez
    # Learning rate coseno con warmup (el preset de DiffusionConfig: warmup 500, termina en ~0).
    # False -> constante 1e-4, como todos los modelos hasta el 13-nov
    lr_scheduler: bool = False
    lr_warmup_steps: int = 500
    # Promedio exponencial de los pesos, como Diffusion Policy (power 0.75, máximo 0.9999). Se guarda
    # y se evalúa el modelo promediado. Con BatchNorm conviene group_norm (las estadísticas se copian)
    ema: bool = False
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999
    training_steps: int = 50000
    batch_size: int = 32
    # Cada worker del DataLoader deja prefetch_factor batches en memoria compartida, así que hay
    # num_workers × prefetch_factor batches en RAM a la vez. Ver loader_memory_bytes
    num_workers: int = 12
    prefetch_factor: int = 1
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """Semilla de cada worker del DataLoader a partir de la del proceso (para random y numpy)."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class EMA:
    """Copia de la política con el promedio exponencial de los pesos, como EMAModel de
    diffusion_policy: decay = 1 - (1 + step)^-power, recortado a [0, max_decay]. Los buffers
    (p. ej. estadísticas de BatchNorm) se copian tal cual."""

    def __init__(self, model: torch.nn.Module, power: float = 0.75, max_decay: float = 0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power = power
        self.max_decay = max_decay
        self.step = 0

    def decay(self) -> float:
        value = 1 - (1 + self.step) ** -self.power
        return min(max(value, 0.0), self.max_decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        decay = self.decay()
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1 - decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)
        self.step += 1


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


class Uint8Images(torch.utils.data.Dataset):
    """Devuelve las imágenes en uint8 en vez de float32 (4 veces menos memoria compartida).

    Los workers decodifican y aplican el aumento de datos en float, y cuantizan al final; el bucle de
    entrenamiento las pasa a float en la GPU con images_to_float. Los valores que el aumento deja fuera
    de [0, 1] (ruido, borrado aleatorio) quedan recortados. Las cámaras del dataset que el modelo no usa
    se descartan aquí para que no pasen por la memoria compartida."""

    def __init__(self, dataset, image_keys):
        self.dataset = dataset
        self.image_keys = list(image_keys)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        for key in [k for k in item if k.startswith("observation.images.")
                    and k.removesuffix("_is_pad") not in self.image_keys]:
            del item[key]
        for key in self.image_keys:
            item[key] = item[key].clamp(0, 1).mul(255).round().to(torch.uint8)
        return item


def images_to_float(batch: dict, image_keys, device: torch.device) -> dict:
    """Pasa las imágenes uint8 de Uint8Images a float en [0, 1], ya en `device`.

    Devuelve un dict nuevo: el DataLoader conserva el que entregó, y si se modificara en el lugar esa
    referencia mantendría viva en la GPU la copia sin normalizar (~0,7 GB con 3 cámaras y batch 32)."""
    return batch | {key: batch[key].to(device, non_blocking=True).float().div_(255) for key in image_keys}


def release_step_inputs(pipeline):
    """Suelta la última transición que cada paso del preprocesador de LeRobot guarda en
    _current_transition. Los primeros pasos (renombrar, agregar dimensión de batch) reciben las imágenes
    de images_to_float ya en la GPU, y sin esto esa copia sin normalizar seguiría ocupando VRAM durante
    el forward y el backward. Algunos pasos contienen sub-pasos (AddBatchDimensionProcessorStep)."""
    for step in pipeline.steps:
        for obj in (step, *vars(step).values()):
            if getattr(obj, "_current_transition", None) is not None:
                obj._current_transition = None


def loader_memory_bytes(image_frames: dict[str, tuple[int, int]], batch_size: int,
                        num_workers: int, prefetch_factor: int, bytes_per_value: int = 1) -> int:
    """RAM de los batches de imágenes que tiene el DataLoader a la vez.

    image_frames: {cámara: (frames por muestra, valores por frame)}. Hay num_workers × prefetch_factor
    batches en memoria compartida, más la copia de pin_memory y el que se está usando."""
    batch = batch_size * sum(frames * size for frames, size in image_frames.values()) * bytes_per_value
    in_flight = num_workers * prefetch_factor + 2 if num_workers else 1
    return batch * in_flight


def total_ram_bytes() -> int:
    """RAM de la máquina, o el límite del contenedor si es menor (cgroup v2, pods de la nube)."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            total = min(total, int(limit))
    except OSError:
        pass
    return total


# Fracción de la RAM que pueden ocupar los batches en vuelo. Los workers y el proceso principal ocupan
# varios GB más; con 56 GB en WSL, 3 cámaras 480x640 en float32 (~27 GB, 48%) terminó en OOM y
# 2 cámaras (~18 GB, 32%) entrenaban bien
MAX_LOADER_RAM_FRACTION = 0.4


def check_loader_memory(cfg: TrainConfig, image_frames: dict[str, tuple[int, int]]):
    """Imprime la RAM estimada del DataLoader y aborta antes de entrenar si no cabe."""
    need = loader_memory_bytes(image_frames, cfg.batch_size, cfg.num_workers, cfg.prefetch_factor)
    total = total_ram_bytes()
    print(f"DataLoader: {cfg.num_workers} workers × prefetch {cfg.prefetch_factor}, "
          f"~{need / 1e9:.1f} GB de imágenes en RAM ({need / total:.0%} de {total / 1e9:.0f} GB)")
    if need > MAX_LOADER_RAM_FRACTION * total:
        raise SystemExit(f"El DataLoader ocuparía ~{need / 1e9:.1f} GB, más del "
                         f"{MAX_LOADER_RAM_FRACTION:.0%} de la RAM: baja --prefetch-factor, "
                         "--num-workers o --batch-size, o reescala el dataset (resize_dataset.py)")


def warn_if_vram_overflows(device: torch.device):
    """Avisa si PyTorch reservó más VRAM que la que tiene la GPU. En WSL/Windows el driver no falla:
    usa RAM del sistema como memoria de GPU y cada paso se vuelve ~10 veces más lento (3 cámaras
    480x640 con batch 32: 19,6 GB en una RTX 5080 de 16 GB, 4,5 s por paso; con batch 16, 0,3 s)."""
    if device.type != "cuda":
        return
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if reserved > total:
        print(f"AVISO: el entrenamiento ocupa {reserved / 1e9:.1f} GB de VRAM y la GPU tiene "
              f"{total / 1e9:.1f} GB; el resto va a la RAM del sistema y es mucho más lento. "
              "Baja --batch-size o usa --resize-shape (p. ej. 240 320)")


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
    if cfg.seed is not None:
        seed_everything(cfg.seed)
        print(f"Semilla: {cfg.seed}")

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
        pretrained_backbone_weights=cfg.pretrained_backbone,
        use_group_norm=cfg.group_norm,
        scheduler_warmup_steps=cfg.lr_warmup_steps,
    )
    print(f"Encoder: {cfg.pretrained_backbone or 'desde cero'}, "
          f"{'GroupNorm' if cfg.group_norm else 'BatchNorm'}")
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
    ema = EMA(policy, cfg.ema_power, cfg.ema_max_decay) if cfg.ema else None   # antes de compilar
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

    image_keys = list(policy_cfg.image_features)
    check_loader_memory(cfg, {k: (len(delta_timestamps[k]), math.prod(policy_cfg.image_features[k].shape))
                              for k in image_keys})

    run = start_wandb(cfg, model_name, output_directory, {
        "cameras_used": image_keys, "dataset_episodes": dataset.num_episodes,
        "dataset_frames": dataset.num_frames, "fps": dataset_metadata.fps})
    if run is not None:
        with open(output_directory / "train_config.yaml", "a") as f:
            yaml.safe_dump({"wandb_run": run.url or run.dir, "wandb_id": run.id}, f, sort_keys=False)

    # Create the optimizer and dataloader for offline training
    print("Creating optimizer and dataloader")
    optimizer = policy_cfg.get_optimizer_preset().build(policy.parameters())
    lr_scheduler = (policy_cfg.get_scheduler_preset().build(optimizer, cfg.training_steps)
                    if cfg.lr_scheduler else None)
    print(f"Learning rate: {policy_cfg.optimizer_lr:g}, "
          + (f"coseno con warmup {cfg.lr_warmup_steps}" if lr_scheduler else "constante")
          + (", EMA" if ema else ""))
    scaler = GradScaler()
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
    dataloader = torch.utils.data.DataLoader(
        Uint8Images(dataset, image_keys),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator,
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
            batch = images_to_float(batch, image_keys, device)
            if cfg.relative_actions:
                batch["action"] = to_relative(batch["action"], batch["observation.state"], policy_cfg.n_obs_steps - 1)
            batch = preprocessor(batch)
            release_step_inputs(preprocessor)

            # Mixed precision con bfloat16 (óptimo para RTX 50xx)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Más eficiente en memoria
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(policy)

            if step == 2:                    # el primer paso todavía no reservó todo
                warn_if_vram_overflows(device)
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

                (ema.model if ema else policy).save_pretrained(checkpoint_dir / "policy")
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
    final = ema.model if ema else policy      # con EMA se guarda y se evalúa el modelo promediado
    final.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if cfg.push_to_hub:
        repo_id = f"{cfg.hub_user}/robot_learning_tutorial_{model_name}"
        final.push_to_hub(repo_id)
        preprocessor.push_to_hub(repo_id)
        postprocessor.push_to_hub(repo_id)
    if run is not None:
        run.summary.update({"training_minutes": round((end_time - start_time).total_seconds() / 60, 1),
                            "final_loss": loss.item(), "model_dir": str(output_directory)})
        run.finish()


if __name__ == "__main__":
    main(load_config(sys.argv[1:]))
