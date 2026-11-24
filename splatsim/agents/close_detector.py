"""Detector de cierre: probabilidad, desde un cuadro de wrist_rgb, de que cerrar la cortadora ahora corte el tallo.

Se entrena con scripts/xarm5/close_detector.py (etiquetas peduncle_cuttable de los ensayos guardados). En la
evaluación, LeRobotDiffusionAgent(close_detector=...) cierra cuando el detector lo dice, además de cuando lo
ordena la política.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIZE = (96, 128)            # alto, ancho de la entrada (wrist_rgb es 480x640)
OPEN_MAX = 0.05             # solo decide con la cortadora abierta
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_model(pretrained: bool = True):
    import torch
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def resize(image: np.ndarray) -> np.ndarray:
    """H x W x 3 uint8 -> SIZE."""
    from PIL import Image
    return np.array(Image.fromarray(image[..., :3]).resize(SIZE[::-1], Image.BILINEAR))


def as_uint8_hwc(image) -> np.ndarray:
    """La imagen de la observación (H x W x 3 o 3 x H x W, uint8 o float en [0, 1]) como H x W x 3 uint8."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.max() <= 1.0 else a.clip(0, 255).astype(np.uint8)
    return a[..., :3]


def to_tensor(images, device):
    """uint8 N x H x W x 3 (ya en SIZE) -> tensor normalizado como ImageNet."""
    import torch
    x = torch.as_tensor(images, device=device).permute(0, 3, 1, 2).float() / 255
    return (x - torch.as_tensor(MEAN, device=device)[:, None, None]) / torch.as_tensor(STD, device=device)[:, None, None]


class CloseDetector:
    """Dispara cuando la probabilidad llega a threshold en consecutive cuadros seguidos."""

    def __init__(self, folder: str, threshold: float = 0.9, consecutive: int = 2, device: str = "cuda"):
        import torch
        folder = Path(folder).expanduser()
        info = json.loads((folder / "detector.json").read_text())
        if tuple(info["size"]) != SIZE:
            raise ValueError(f"{folder}: entrenado con tamaño {info['size']} y el código usa {SIZE}")
        self.model = make_model(pretrained=False)
        self.model.load_state_dict(torch.load(folder / "detector.pt", map_location="cpu"))
        self.model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.consecutive = consecutive
        self.last_prob = float("nan")
        self._run = 0

    def reset(self) -> None:
        self._run = 0
        self.last_prob = float("nan")

    def prob(self, image) -> float:
        import torch
        with torch.no_grad():
            x = to_tensor(resize(as_uint8_hwc(image))[None], self.device)
            return float(torch.sigmoid(self.model(x)).item())

    def fire(self, image) -> bool:
        self.last_prob = self.prob(image)
        self._run = self._run + 1 if self.last_prob >= self.threshold else 0
        return self._run >= self.consecutive
