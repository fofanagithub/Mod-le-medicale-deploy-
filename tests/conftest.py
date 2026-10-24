"""Fixtures partagées.

Les tests n'utilisent ni les données ni le modèle entraîné : un `XRayClassifier` aux poids
aléatoires suffit à vérifier le *contrat* du code (formes, types, erreurs, HTTP). La CI
tourne ainsi en quelques secondes, sans téléchargement.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from medxray.api.main import create_app
from medxray.config import Settings
from medxray.inference import Predictor
from medxray.model import ModelMetadata, XRayClassifier, save_checkpoint


def make_image_bytes(
    size: tuple[int, int] = (320, 280), mode: str = "L", fmt: str = "PNG", seed: int = 0
) -> bytes:
    """Fausse radio : gradient + bruit, dans le mode PIL demandé."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.linspace(0, 1, w)[None, :] * np.linspace(0.3, 1, h)[:, None]
    arr = np.clip(base + 0.1 * rng.standard_normal((h, w)), 0, 1)
    if mode == "I;16":
        img = Image.fromarray((arr * 65535).astype(np.uint16))
    elif mode == "RGB":
        noise = rng.random((h, w, 3))
        img = Image.fromarray(
            (np.clip(arr[..., None] * 0.7 + noise * 0.3, 0, 1) * 255).astype(np.uint8)
        )
    else:
        img = Image.fromarray((arr * 255).astype(np.uint8)).convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("models") / "model.pt"
    metadata = ModelMetadata(model_version="0.0.0-test", threshold=0.5, notes="test")
    save_checkpoint(path, XRayClassifier(pretrained=False), metadata)
    return path


@pytest.fixture(scope="session")
def predictor(checkpoint_path: Path) -> Predictor:
    return Predictor.from_checkpoint(checkpoint_path)


@pytest.fixture
def settings(checkpoint_path: Path) -> Settings:
    return Settings(model_path=checkpoint_path, max_upload_mb=1)


@pytest.fixture
def client(settings: Settings, predictor: Predictor) -> Iterator[TestClient]:
    # Le `with` déclenche le lifespan (startup/shutdown), comme en production.
    with TestClient(create_app(settings, predictor)) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes()
