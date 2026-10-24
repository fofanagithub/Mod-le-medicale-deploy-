from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from medxray.inference import Predictor
from medxray.preprocessing import load_image
from tests.conftest import make_image_bytes


@pytest.fixture
def image() -> Image.Image:
    return load_image(make_image_bytes(size=(700, 600)))


def test_predict_label_follows_threshold(predictor: Predictor, image: Image.Image) -> None:
    pred = predictor.predict(image)
    assert 0.0 <= pred.probability_pneumonia <= 1.0
    expected = "pneumonia" if pred.probability_pneumonia >= pred.threshold else "normal"
    assert pred.label == expected


def test_explain_is_consistent_with_predict(predictor: Predictor, image: Image.Image) -> None:
    explanation = predictor.explain(image)
    pred = predictor.predict(image)
    assert explanation.prediction.probability_pneumonia == pytest.approx(
        pred.probability_pneumonia, abs=1e-6
    )
    assert explanation.explained_class == pred.label


@pytest.mark.parametrize("target", ["normal", "pneumonia"])
def test_explain_forced_target(
    predictor: Predictor, image: Image.Image, target: Literal["normal", "pneumonia"]
) -> None:
    assert predictor.explain(image, target=target).explained_class == target


def test_explanation_outputs(predictor: Predictor, image: Image.Image) -> None:
    explanation = predictor.explain(image)
    assert explanation.heatmap.shape == (224, 224)
    assert explanation.heatmap.dtype == np.float32
    assert explanation.overlay.mode == "RGB"
    assert explanation.overlay.size == (512, 512)  # min(512, côté court = 600)
    assert Image.open(io.BytesIO(explanation.overlay_png())).format == "PNG"
