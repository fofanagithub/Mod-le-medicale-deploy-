"""Architecture du modèle et format de checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES: tuple[str, str] = ("normal", "pneumonia")
POSITIVE_CLASS = "pneumonia"
ARCHITECTURE = "resnet18"
CHECKPOINT_FORMAT_VERSION = 1


class XRayClassifier(nn.Module):
    """ResNet18 découpé explicitement en `features` (convolutions) et `head` (classifieur).

    Sortie : un unique logit, P(pneumonie) = sigmoid(logit). Un seul logit plutôt qu'un
    softmax à 2 classes rend explicite le fait que la décision repose sur un *seuil*, que
    l'on choisit ensuite selon l'objectif clinique (voir `training/metrics.py`).

    Le découpage features/head permet un Grad-CAM sans hook (voir `gradcam.py`).
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,  # couche cible de Grad-CAM : (B, 512, 7, 7) pour une entrée 224
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits: Tensor = self.head(self.features(x))
        return logits.squeeze(1)


@dataclass(frozen=True)
class ModelMetadata:
    """Tout ce qu'il faut pour savoir *quel* modèle tourne en production, et comment l'utiliser.

    Le seuil de décision fait partie du modèle : il est choisi sur la validation et voyage
    avec les poids, jamais codé en dur dans l'API.
    """

    model_version: str
    threshold: float
    image_size: int = 224
    architecture: str = ARCHITECTURE
    class_names: tuple[str, ...] = CLASS_NAMES
    dataset: str = "PneumoniaMNIST (MedMNIST v2), 224x224"
    trained_at: str = ""
    torch_version: str = str(torch.__version__)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        if "class_names" in values:
            values["class_names"] = tuple(values["class_names"])
        return cls(**values)


def save_checkpoint(path: Path, model: XRayClassifier, metadata: ModelMetadata) -> None:
    """Sauvegarde poids + métadonnées. Uniquement des tenseurs et des types primitifs,
    afin que le fichier soit relisible avec `weights_only=True`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = asdict(metadata)
    meta["class_names"] = list(metadata.class_names)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "metadata": meta,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[XRayClassifier, ModelMetadata]:
    """Charge un checkpoint de façon sûre.

    `weights_only=True` est indispensable : un `torch.load` classique désérialise du pickle
    arbitraire, donc un fichier de poids malveillant pourrait exécuter du code au démarrage
    du service.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Format de checkpoint non supporté : {version!r}")
    metadata = ModelMetadata.from_dict(payload["metadata"])
    model = XRayClassifier(pretrained=False)
    model.load_state_dict(payload["state_dict"])  # strict=True : toute divergence échoue
    model.to(device).eval()
    return model, metadata


def file_sha256(path: Path) -> str:
    """Empreinte du fichier de poids, exposée par l'API pour la traçabilité."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
