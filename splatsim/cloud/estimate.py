"""Costo estimado de un trabajo en cada oferta, a partir de velocidades medidas.

La velocidad se toma del historial: cada trabajo terminado agrega un registro. Sin una medición
comparable no se inventa un número (rate = None, confianza "sin datos").
"""
from dataclasses import dataclass

from splatsim.cloud.providers import Offer

IMAGE_GB = 4.82               # splatsim-train comprimida: lo que cada pod descarga y paga
IMAGE_DISK_GB = 9.06          # lo que ocupa ya extraída
RENDER_IMAGE_GB = 5.12        # splatsim-render comprimida
RENDER_IMAGE_DISK_GB = 9.71
ASSETS_GB = 1.1               # escena mínima, URDF y tomate
PNG_GB = 0.44e-3              # 53 GB / 122264 PNG de planned_261103_10hz
RENDER_STARTUP_S = 120        # cargar la escena en la GPU
LEAF_FIT_S = 10               # sin medir en un pod: unas 40 mediciones de oclusión de 0.2 s por episodio
STARTUP_S = 260               # GPU, dataset, carga y torch.compile, medidos en la prueba de 500 pasos
CHECKPOINT_GB = 1.1           # model.safetensors del modelo de 2 cámaras
CONVERT_STARTUP_S = 60        # sin GPU ni torch.compile
DATASET_GB_PER_FRAME_CAMERA = 14.6e-6   # 13.0-14.6 kB en los datasets de 640x480 del lab: el mayor
SCENE_GB_PER_FRAME = 2.8e-3   # images/ y sparse/ de 260729_2003: 3.3 GB para 1189 fotos 4K sin distorsión
SPLAT_GB = 0.4                # output/ de 260729_2003: 363 MB
SPLAT_STARTUP_S = 60          # extraer las fotos del video
CONFIDENCE_ORDER = {"alta": 0, "baja": 1, "sin datos": 2}
# Con la imagen ya en la máquina, el pod arrancó en 18 s en vez de 7-25 min. La espera no se cobra, así
# que esto es lo que vale ahorrarla: un trabajo largo no acaba en una máquina mucho más lenta por esto.
PREFER_CACHED_USD = 0.15
PREFER_CACHED_DPH = 1.25      # sin estimaciones que comparar, cuánto más cara por hora

# Medido en vast con 12 workers y en RunPod (entrenamiento completo), noviembre de 2026.
SEED_HISTORY = [
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD Ryzen 9 7950X 16-Core Processor", "rate": 6.30},
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD Ryzen 5 7600 6-Core Processor", "rate": 4.49},    # 50k pasos en 185.7 min
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD EPYC 7452 32-Core Processor", "rate": 3.26},    # últimos 326 pasos de la prueba, PCIe 26
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD EPYC 9355 32-Core Processor", "rate": 7.2},    # últimos 600 de 1200 pasos, 16 núcleos
    {"kind": "train", "profile": "bs32-cam2-noaug-compile", "gpu": "RTX 5090",
     "cpu": "AMD Ryzen 9 7950X 16-Core Processor", "rate": 4.03},    # otra máquina que la de 6.30: varían
    {"kind": "convert", "profile": "convert-cam2", "gpu": "", "cpu": "AMD Ryzen 9 9950X 16-Core Processor", "rate": 15.1},    # lab, 3 episodios; usa ~1.5 núcleos
    {"kind": "convert", "profile": "convert-cam2", "gpu": "RTX 4060",
     "cpu": "AMD EPYC 7502 32-Core Processor", "rate": 7.96},    # pod, 790 frames incluido el arranque
    {"kind": "convert", "profile": "convert-cam4", "gpu": "Quadro P4000",
     "cpu": "Xeon® E5-2690 0 ", "rate": 3.89},    # pod, 2122 frames en 545 s
    {"kind": "render", "profile": "render-cam4-w1", "gpu": "RTX 5080",
     "cpu": "AMD Ryzen 9 9950X 16-Core Processor", "rate": 4.25},    # lab, 117 frames sin pantalla
    {"kind": "render", "profile": "render-cam4-w1", "gpu": "RTX 5090",
     "cpu": "AMD EPYC 9355 32-Core Processor", "rate": 7.0},    # pod, 1061 frames con un hilo
    {"kind": "mirror", "profile": "mirror-cam4-w3", "gpu": "RTX 5090",
     "cpu": "XEON® GOLD 6530", "rate": 11.73},    # pod, 2122 frames en 181 s
]


@dataclass
class Workload:
    kind: str
    profile: str         # lo que cambia la velocidad además del hardware
    work: float          # pasos o frames
    down_gb: float       # imagen + entradas
    up_gb: float         # salidas
    startup_s: float = STARTUP_S
    uses_gpu: bool = True
    image_gb: float = IMAGE_GB
    image_disk_gb: float = IMAGE_DISK_GB


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
    cached: bool = False      # la máquina ya corrió una de nuestras imágenes


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


def convert_workload(frames: int, cameras: int, recordings_gb: float) -> Workload:
    return Workload("convert", f"convert-cam{cameras}", frames, IMAGE_GB + recordings_gb,
                    predicted_dataset_gb(frames, cameras), CONVERT_STARTUP_S, uses_gpu=False)


def render_workload(kind: str, frames: int, cameras: int, workers: int, input_gb: float,
                    leaf_episodes: int = 0) -> Workload:
    """work = frames que deja la etapa; un espejo con originales renderiza el doble."""
    return Workload(kind, f"{kind}-cam{cameras}-w{workers}", frames, RENDER_IMAGE_GB + ASSETS_GB + input_gb,
                    frames * cameras * PNG_GB, RENDER_STARTUP_S + leaf_episodes * LEAF_FIT_S / workers,
                    image_gb=RENDER_IMAGE_GB,
                    image_disk_gb=RENDER_IMAGE_DISK_GB)


def splat_workload(frames: int, matcher: str, iterations: int, video_gb: float) -> Workload:
    """work = fotos. Emparejar todas con todas crece con su cuadrado: la velocidad solo vale para
    escenas de un tamaño parecido."""
    return Workload("splat", f"splat-{matcher}-it{iterations}", frames, RENDER_IMAGE_GB + video_gb,
                    frames * SCENE_GB_PER_FRAME + SPLAT_GB, SPLAT_STARTUP_S, image_gb=RENDER_IMAGE_GB,
                    image_disk_gb=RENDER_IMAGE_DISK_GB)


def predicted_dataset_gb(frames: int, cameras: int) -> float:
    return frames * cameras * DATASET_GB_PER_FRAME_CAMERA


def lookup_rate(history: list[dict], work: Workload, offer: Offer) -> tuple[float | None, str, str]:
    """(velocidad, confianza, de dónde sale). Con otra CPU se toma la medición más lenta de esa GPU;
    si el trabajo no usa GPU, de cualquiera."""
    similar = [r for r in history if r["kind"] == work.kind and r["profile"] == work.profile
               and (not work.uses_gpu or r["gpu"] == offer.gpu_name)]
    same_cpu = [r["rate"] for r in similar if r["cpu"] == offer.cpu_name]
    if same_cpu:
        return sum(same_cpu) / len(same_cpu), "alta", f"{len(same_cpu)} medición(es) con este hardware"
    if similar:
        return min(r["rate"] for r in similar), "baja", "misma GPU, otra CPU" if work.uses_gpu else "otra CPU"
    return None, "sin datos", "sin mediciones comparables"


def estimate(history: list[dict], work: Workload, offer: Offer) -> Estimate:
    rate, confidence, source = lookup_rate(history, work, offer)
    # la descarga de la imagen se cobra por GB, no por hora: la instancia aún no corre
    transfer_s = ((work.down_gb - work.image_gb) * 8000 / max(offer.inet_down_mbps, 1)
                  + work.up_gb * 8000 / max(offer.inet_up_mbps, 1))
    transfer_cost = work.down_gb * offer.down_cost_gb + work.up_gb * offer.up_cost_gb
    if rate is None:
        return Estimate(offer, None, confidence, source, None, None, transfer_cost, None)
    hours = (work.startup_s + work.work / rate + transfer_s) / 3600
    compute = hours * offer.dph
    return Estimate(offer, rate, confidence, source, hours, compute, transfer_cost, compute + transfer_cost)


def rank(history: list[dict], work: Workload, offers: list[Offer]) -> list[Estimate]:
    """Primero las estimaciones más fiables; dentro de cada grupo, las más baratas. Sin velocidad no hay
    total: se compara una hora más la transferencia, que en algunas máquinas vale más que el cómputo."""
    estimates = [estimate(history, work, o) for o in offers]
    return sorted(estimates, key=lambda e: (CONFIDENCE_ORDER[e.confidence],
                                            e.total if e.total is not None else e.offer.dph + e.transfer_cost))


def prefer_machines(ranked: list[Estimate], cached: set[int], failed: set[int]) -> list[Estimate]:
    """Primero las máquinas que ya tienen la imagen, si no salen mucho más caras que la mejor; al final,
    las que fallaron hace poco."""
    usable = [e for e in ranked if e.offer.machine_id not in failed]
    best = usable[0] if usable else None

    def worth_it(e: Estimate) -> bool:
        if not e.cached or best is None:
            return False
        if e.total is not None and best.total is not None:
            return e.total <= best.total + PREFER_CACHED_USD
        return (e.total is None and best.total is None and e.offer.dph <= best.offer.dph * PREFER_CACHED_DPH
                and e.transfer_cost <= best.transfer_cost + PREFER_CACHED_USD)

    for e in ranked:
        e.cached = e.offer.machine_id in cached
    return sorted(ranked, key=lambda e: (e.offer.machine_id in failed, not worth_it(e)))
