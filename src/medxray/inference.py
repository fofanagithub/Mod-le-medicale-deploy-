"""Couche d'inférence : indépendante de FastAPI, réutilisable en batch, en CLI ou en test."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from medxray.gradcam import grad_cam
from medxray.model import (
    POSITIVE_CLASS,
    ModelMetadata,
    XRayClassifier,
    file_sha256,
    load_checkpoint,
)
from medxray.preprocessing import display_crop, to_model_input

ClassName = Literal["normal", "pneumonia"]
MAX_DISPLAY_SIZE = 512


@dataclass(frozen=True)
class Prediction:
    label: ClassName
    probability_pneumonia: float
    threshold: float


@dataclass(frozen=True)
class Explanation:
    prediction: Prediction
    explained_class: ClassName
    heatmap: np.ndarray  # (H, W) float32 dans [0, 1], à la résolution d'entrée du modèle
    overlay: Image.Image  # radio recadrée + heatmap superposée (RGB)

    def overlay_png(self) -> bytes:
        buffer = io.BytesIO()
        self.overlay.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


class Predictor:
    """Encapsule modèle + métadonnées + prétraitement. Thread-safe : aucun état mutable."""

    def __init__(
        self,
        model: XRayClassifier,
        metadata: ModelMetadata,
        sha256: str = "",
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.sha256 = sha256
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> Predictor:
        model, metadata = load_checkpoint(path, device=device)
        return cls(model, metadata, sha256=file_sha256(path), device=device)

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return to_model_input(image, self.metadata.image_size).to(self.device)

    def _to_prediction(self, logit: float) -> Prediction:
        probability = float(torch.sigmoid(torch.tensor(logit)))
        threshold = self.metadata.threshold
        label: ClassName = "pneumonia" if probability >= threshold else "normal"
        return Prediction(label, probability, threshold)

    def predict(self, image: Image.Image) -> Prediction:
        with torch.inference_mode():
            logit = float(self.model(self._prepare(image))[0])
        return self._to_prediction(logit)

    def explain(
        self, image: Image.Image, target: ClassName | Literal["predicted"] = "predicted"
    ) -> Explanation:
        """Prédiction + carte Grad-CAM de la classe `target` (par défaut, la classe prédite).

        Expliquer la classe prédite est le cas d'usage normal ; expliquer l'autre classe
        permet de voir quelles zones « plaident contre » la décision.
        """
        x = self._prepare(image)
        if target == "predicted":
            with torch.inference_mode():
                prediction = self._to_prediction(float(self.model(x)[0]))
            explained: ClassName = prediction.label
        else:
            explained = target
        sign = 1.0 if explained == POSITIVE_CLASS else -1.0

        logits, cams = grad_cam(self.model.features, self.model.head, x, sign=sign)
        prediction = self._to_prediction(float(logits[0]))
        heatmap = cams[0].cpu().numpy().astype(np.float32)
        return Explanation(prediction, explained, heatmap, render_overlay(image, heatmap))


def render_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpose la heatmap (colormap « jet ») sur la radio, avec le recadrage du modèle.

    L'opacité est proportionnelle à l'intensité : les zones sans évidence restent la radio
    d'origine, ce qui évite le « tout rouge-bleu » illisible des overlays uniformes.
    """
    size = min(MAX_DISPLAY_SIZE, *image.size)
    base = np.asarray(display_crop(image, size), dtype=np.float32) / 255.0
    heat = np.asarray(
        Image.fromarray(heatmap.astype(np.float32)).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).clip(0.0, 1.0)

    weight = (alpha * heat)[..., None]
    blended = base[..., None] * (1.0 - weight) + _jet(heat) * weight
    return Image.fromarray((blended * 255.0).round().astype(np.uint8))


def _jet(x: np.ndarray) -> np.ndarray:
    """Colormap « jet » vectorisée (évite d'embarquer matplotlib dans l'image Docker)."""
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)
