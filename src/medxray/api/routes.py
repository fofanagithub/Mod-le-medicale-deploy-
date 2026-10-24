"""Routes HTTP.

Les endpoints de calcul sont déclarés en `def` (et non `async def`) : l'inférence PyTorch est
CPU-bound et bloquante. FastAPI exécute les `def` dans un pool de threads, ce qui garde la
boucle événementielle libre pour /health et les autres requêtes. Un `async def` qui
appellerait le modèle directement bloquerait tout le serveur pendant le calcul.
"""

from __future__ import annotations

import base64
import time
from importlib.resources import files
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from medxray.api.schemas import (
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    ReadinessResponse,
)
from medxray.config import Settings
from medxray.inference import Predictor
from medxray.preprocessing import InvalidImageError, load_image

router = APIRouter()

ALLOWED_CONTENT_TYPES = frozenset({"image/png", "image/jpeg"})
UPLOAD_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux."},
    415: {"model": ErrorResponse, "description": "Type de fichier non supporté."},
    422: {"model": ErrorResponse, "description": "Image illisible ou invalide."},
}


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé.")
    return predictor


SettingsDep = Annotated[Settings, Depends(get_settings)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]
UploadDep = Annotated[UploadFile, File(description="Radiographie thoracique (PNG ou JPEG).")]


def read_image(upload: UploadFile, settings: Settings) -> Image.Image:
    """Validation défensive de l'upload : type déclaré, taille, puis contenu réel."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Type {upload.content_type!r} non supporté. Envoyez une image PNG ou JPEG.",
        )
    # On lit au plus limite + 1 octet : un fichier de 2 Go n'est jamais chargé en mémoire.
    data = upload.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Fichier trop volumineux (max {settings.max_upload_mb:g} Mo).",
        )
    try:
        return load_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.get("/", include_in_schema=False)
def demo_page() -> HTMLResponse:
    html = files("medxray.api").joinpath("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness : le processus répond. Ne touche pas au modèle (doit rester instantané)."""
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
def ready(predictor: PredictorDep) -> ReadinessResponse:
    """Readiness : le modèle est chargé, le service peut recevoir du trafic."""
    return ReadinessResponse(model_version=predictor.metadata.model_version)


@router.get("/v1/model", response_model=ModelInfo, tags=["model"])
def model_info(predictor: PredictorDep) -> ModelInfo:
    """Quel modèle est servi : version, seuil, métriques de test, empreinte des poids."""
    meta = predictor.metadata
    return ModelInfo(
        model_version=meta.model_version,
        architecture=meta.architecture,
        class_names=list(meta.class_names),
        image_size=meta.image_size,
        threshold=meta.threshold,
        dataset=meta.dataset,
        trained_at=meta.trained_at,
        torch_version=meta.torch_version,
        sha256=predictor.sha256,
        metrics=meta.metrics,
        notes=meta.notes,
    )


@router.post(
    "/v1/predict", response_model=PredictionResponse, responses=UPLOAD_ERRORS, tags=["model"]
)
def predict(file: UploadDep, predictor: PredictorDep, settings: SettingsDep) -> PredictionResponse:
    """Classe une radiographie thoracique : normal ou pneumonie."""
    image = read_image(file, settings)
    start = time.perf_counter()
    result = predictor.predict(image)
    return PredictionResponse(
        label=result.label,
        probability_pneumonia=round(result.probability_pneumonia, 6),
        threshold=round(result.threshold, 6),
        model_version=predictor.metadata.model_version,
        inference_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@router.post(
    "/v1/explain",
    response_model=ExplanationResponse,
    responses={
        **UPLOAD_ERRORS,
        200: {"content": {"image/png": {}}, "description": "JSON, ou PNG si format=png."},
    },
    tags=["model"],
)
def explain(
    file: UploadDep,
    predictor: PredictorDep,
    settings: SettingsDep,
    target: Annotated[
        Literal["predicted", "normal", "pneumonia"],
        Query(description="Classe à expliquer (par défaut : la classe prédite)."),
    ] = "predicted",
    response_format: Annotated[
        Literal["json", "png"],
        Query(alias="format", description="`png` renvoie directement l'image."),
    ] = "json",
) -> ExplanationResponse | Response:
    """Prédiction + carte Grad-CAM : les zones de la radio qui ont pesé dans la décision."""
    image = read_image(file, settings)
    start = time.perf_counter()
    result = predictor.explain(image, target=target)
    png = result.overlay_png()
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    pred = result.prediction

    if response_format == "png":
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Prediction-Label": pred.label,
                "X-Probability-Pneumonia": f"{pred.probability_pneumonia:.6f}",
                "X-Explained-Class": result.explained_class,
                "X-Model-Version": predictor.metadata.model_version,
            },
        )
    return ExplanationResponse(
        label=pred.label,
        probability_pneumonia=round(pred.probability_pneumonia, 6),
        threshold=round(pred.threshold, 6),
        model_version=predictor.metadata.model_version,
        inference_ms=elapsed_ms,
        explained_class=result.explained_class,
        heatmap_png_base64=base64.b64encode(png).decode("ascii"),
    )
