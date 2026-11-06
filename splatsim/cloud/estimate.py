"""Costo estimado de un trabajo en cada oferta, a partir de velocidades medidas.

La velocidad se toma del historial: cada trabajo terminado agrega un registro. Sin una medición
comparable no se inventa un número (rate = None, confianza "sin datos").
"""
from dataclasses import dataclass

from splatsim.cloud.providers import Offer

IMAGE_GB = 9.0                # tamaño de splatsim-train; se paga en cada pod que la descarga
CHECKPOINT_GB = 1.1           # model.safetensors del modelo de 2 cámaras
CONFIDENCE_ORDER = {"alta": 0, "baja": 1, "sin datos": 2}

# Medido en vast con 12 workers y en RunPod (entrenamiento completo), noviembre de 2026.
SEED_HISTORY = [
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD Ryzen 9 7950X 16-Core Processor", "rate": 6.30},
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD Ryzen 5 7600 6-Core Processor", "rate": 4.49},    # 50k pasos en 185.7 min
]


@dataclass
class Workload:
    kind: str
    profile: str         # lo que cambia la velocidad además del hardware
    work: float          # pasos o frames
    down_gb: float       # imagen + entradas
    up_gb: float         # salidas


@dataclass
class Estimate:
    offer: Offer
    rate: float | None
    confidence: str
    source: str
    hours: float | None
    compute_cost: float | None
    transfer_cost: float
    total: float | None


def train_profile(cfg: dict) -> str:
    cameras = cfg.get("cameras")
    if cameras:
        count = len(cameras)
    elif cfg.get("arrangement"):
        digits = "".join(c for c in str(cfg["arrangement"]) if c.isdigit())
        count = digits or "?"
    else:
        count = "all"
    return (f"bs{cfg.get('batch_size', '?')}-cam{count}"
            f"-{'aug' if cfg.get('image_augmentation') else 'noaug'}"
            f"-{'compile' if cfg.get('torch_compile') else 'nocompile'}")


def train_workload(cfg: dict, dataset_gb: float) -> Workload:
    steps = int(cfg["training_steps"])
    checkpoints = steps // int(cfg.get("save_freq") or steps) + 1
    return Workload("train", train_profile(cfg), steps, IMAGE_GB + dataset_gb, checkpoints * CHECKPOINT_GB)


def lookup_rate(history: list[dict], work: Workload, offer: Offer) -> tuple[float | None, str, str]:
    """(velocidad, confianza, de dónde sale). Con otra CPU se toma la medición más lenta de esa GPU."""
    same_gpu = [r for r in history if r["kind"] == work.kind and r["profile"] == work.profile
                and r["gpu"] == offer.gpu_name]
    same_cpu = [r["rate"] for r in same_gpu if r["cpu"] == offer.cpu_name]
    if same_cpu:
        return sum(same_cpu) / len(same_cpu), "alta", f"{len(same_cpu)} medición(es) con este hardware"
    if same_gpu:
        return min(r["rate"] for r in same_gpu), "baja", "misma GPU, otra CPU"
    return None, "sin datos", "sin mediciones comparables"


def estimate(history: list[dict], work: Workload, offer: Offer) -> Estimate:
    rate, confidence, source = lookup_rate(history, work, offer)
    transfer_s = (work.down_gb * 8000 / max(offer.inet_down_mbps, 1)
                  + work.up_gb * 8000 / max(offer.inet_up_mbps, 1))
    transfer_cost = work.down_gb * offer.down_cost_gb + work.up_gb * offer.up_cost_gb
    if rate is None:
        return Estimate(offer, None, confidence, source, None, None, transfer_cost, None)
    hours = (work.work / rate + transfer_s) / 3600
    compute = hours * offer.dph
    return Estimate(offer, rate, confidence, source, hours, compute, transfer_cost, compute + transfer_cost)


def rank(history: list[dict], work: Workload, offers: list[Offer]) -> list[Estimate]:
    """Primero las estimaciones más fiables; dentro de cada grupo, las más baratas."""
    estimates = [estimate(history, work, o) for o in offers]
    return sorted(estimates, key=lambda e: (CONFIDENCE_ORDER[e.confidence],
                                            e.total if e.total is not None else e.offer.dph))
