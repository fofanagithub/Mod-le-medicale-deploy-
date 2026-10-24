"""Application FastAPI (pattern *app factory*).

Lancement :
    uvicorn --factory medxray.api.main:create_app --host 0.0.0.0 --port 8000

La factory permet d'injecter une configuration ou un `Predictor` de test sans variable
globale ni monkey-patching : les tests construisent leur propre application isolée.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from medxray import __version__
from medxray.api.log_config import configure_logging
from medxray.api.routes import router
from medxray.config import Settings
from medxray.inference import Predictor

logger = logging.getLogger("medxray.api")
# Un X-Request-ID fourni par le client n'est repris que s'il est inoffensif (pas d'injection
# dans les logs ou les en-têtes).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Les sondes (healthcheck toutes les 30 s) noieraient les logs utiles : niveau DEBUG.
_PROBE_PATHS = frozenset({"/health", "/ready"})


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if settings.torch_threads:
            torch.set_num_threads(settings.torch_threads)
        # Le modèle est chargé UNE fois au démarrage, pas à chaque requête. Si le fichier
        # est absent ou corrompu, le démarrage échoue immédiatement (fail fast) : mieux
        # vaut un conteneur qui redémarre en erreur qu'un service « up » qui renvoie des 500.
        app.state.predictor = predictor or Predictor.from_checkpoint(settings.model_path)
        meta = app.state.predictor.metadata
        logger.info(
            "model_loaded",
            extra={
                "model_version": meta.model_version,
                "sha256": app.state.predictor.sha256[:12],
                "threshold": meta.threshold,
                "torch_threads": torch.get_num_threads(),
            },
        )
        yield
        app.state.predictor = None

    app = FastAPI(
        title="MedXRay API",
        summary="Détection de pneumonie sur radiographie thoracique, avec explications Grad-CAM.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(router)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Identifiant de requête (corrélation des logs) + latence de bout en bout."""
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled_error", extra={"request_id": request_id})
            raise
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        level = logging.DEBUG if request.url.path in _PROBE_PATHS else logging.INFO
        logger.log(
            level,
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        return response

    return app
