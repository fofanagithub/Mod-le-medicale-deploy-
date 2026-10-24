"""Contrats d'entrée/sortie de l'API. Ils génèrent la documentation OpenAPI (/docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Outil de démonstration à visée pédagogique. Ce n'est pas un dispositif médical : "
    "il ne doit pas être utilisé pour poser un diagnostic."
)

ClassName = Literal["normal", "pneumonia"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_version: str


class PredictionResponse(BaseModel):
    label: ClassName = Field(description="Classe prédite, après application du seuil.")
    probability_pneumonia: float = Field(ge=0, le=1, description="P(pneumonie) du modèle.")
    threshold: float = Field(ge=0, le=1, description="Seuil de décision du modèle.")
    model_version: str
    inference_ms: float = Field(description="Temps de calcul du modèle (hors réseau).")
    disclaimer: str = DISCLAIMER

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "pneumonia",
                    "probability_pneumonia": 0.93,
                    "threshold": 0.41,
                    "model_version": "1.0.0",
                    "inference_ms": 38.2,
                    "disclaimer": DISCLAIMER,
                }
            ]
        }
    }


class ExplanationResponse(PredictionResponse):
    explained_class: ClassName = Field(description="Classe dont Grad-CAM montre l'évidence.")
    heatmap_png_base64: str = Field(description="Radio + heatmap superposée, PNG encodé base64.")


class ModelInfo(BaseModel):
    model_version: str
    architecture: str
    class_names: list[str]
    image_size: int
    threshold: float
    dataset: str
    trained_at: str
    torch_version: str
    sha256: str = Field(description="Empreinte du fichier de poids servi.")
    metrics: dict[str, Any]
    notes: str


class ErrorResponse(BaseModel):
    detail: str
