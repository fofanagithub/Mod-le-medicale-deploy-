from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from medxray.preprocessing import (
    IMAGE_SIZE,
    InvalidImageError,
    display_crop,
    eval_transform,
    load_image,
    to_model_input,
)
from tests.conftest import make_image_bytes


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "I;16"])
def test_load_image_accepts_common_modes(mode: str) -> None:
    image = load_image(make_image_bytes(mode=mode))
    assert image.mode == "L"
    assert image.size == (320, 280)


def test_load_image_accepts_jpeg() -> None:
    assert load_image(make_image_bytes(mode="RGB", fmt="JPEG")).mode == "L"


def test_16bit_image_is_rescaled_not_saturated() -> None:
    # Un convert("L") naïf sur du 16 bits donnerait une image quasi uniforme à 255.
    arr = np.asarray(load_image(make_image_bytes(mode="I;16")))
    assert arr.min() < 50
    assert arr.max() > 200


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"pas une image", "illisible"),
        (b"", "illisible"),
        (make_image_bytes(fmt="BMP"), "Format non supporté"),
        (make_image_bytes(size=(40, 300)), "trop petite"),
    ],
    ids=["texte", "vide", "bmp", "trop-petite"],
)
def test_load_image_rejects_invalid_input(data: bytes, message: str) -> None:
    with pytest.raises(InvalidImageError, match=message):
        load_image(data)


def test_load_image_rejects_truncated_file() -> None:
    data = make_image_bytes()
    with pytest.raises(InvalidImageError):
        load_image(data[: len(data) // 2])


def test_load_image_applies_exif_orientation() -> None:
    img = Image.new("RGB", (300, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # l'appareil demande une rotation de 90°
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=exif)
    assert load_image(buffer.getvalue()).size == (200, 300)


def test_model_input_shape() -> None:
    x = to_model_input(load_image(make_image_bytes(size=(500, 300))))
    assert x.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_eval_transform_is_deterministic() -> None:
    image = load_image(make_image_bytes())
    assert torch.equal(eval_transform()(image), eval_transform()(image))


def test_display_crop_matches_model_geometry() -> None:
    """L'image affichée doit être exactement le recadrage vu par le modèle (sinon la heatmap
    Grad-CAM serait décalée par rapport à l'anatomie)."""
    image = load_image(make_image_bytes(size=(600, 400)))
    crop = display_crop(image, IMAGE_SIZE)
    assert crop.size == (IMAGE_SIZE, IMAGE_SIZE)
    model_view = eval_transform()(image)[0]
    expected = (np.asarray(crop, dtype=np.float32) / 255 - 0.485) / 0.229
    np.testing.assert_allclose(model_view.numpy(), expected, atol=1e-5)
