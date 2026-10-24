from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medxray.api.main import create_app
from medxray.config import Settings
from tests.conftest import make_image_bytes


def _upload(data: bytes, content_type: str = "image/png") -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("xray.png", data, content_type)}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_model_version(client: TestClient) -> None:
    assert client.get("/ready").json() == {"status": "ready", "model_version": "0.0.0-test"}


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["model_version"] == "0.0.0-test"
    assert body["class_names"] == ["normal", "pneumonia"]
    assert len(body["sha256"]) == 64


def test_predict(client: TestClient, png_bytes: bytes) -> None:
    response = client.post("/v1/predict", files=_upload(png_bytes))
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"normal", "pneumonia"}
    assert 0 <= body["probability_pneumonia"] <= 1
    assert body["threshold"] == 0.5
    assert "dispositif médical" in body["disclaimer"]


def test_explain_json(client: TestClient, png_bytes: bytes) -> None:
    response = client.post("/v1/explain", files=_upload(png_bytes))
    assert response.status_code == 200
    body = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"])))
    assert image.format == "PNG"
    assert body["explained_class"] == body["label"]


def test_explain_png(client: TestClient, png_bytes: bytes) -> None:
    response = client.post(
        "/v1/explain", params={"format": "png", "target": "pneumonia"}, files=_upload(png_bytes)
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-explained-class"] == "pneumonia"
    assert Image.open(io.BytesIO(response.content)).format == "PNG"


def test_explain_rejects_unknown_target(client: TestClient, png_bytes: bytes) -> None:
    response = client.post("/v1/explain", params={"target": "covid"}, files=_upload(png_bytes))
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("data", "content_type", "expected"),
    [
        (b"%PDF-1.4", "application/pdf", 415),
        (b"pas une image", "image/png", 422),
        (make_image_bytes(fmt="BMP"), "image/png", 422),  # Content-Type mensonger
        (make_image_bytes(size=(1500, 1500), mode="RGB", seed=1), "image/png", 413),  # > 1 Mo
    ],
    ids=["mauvais-type", "corrompu", "type-mensonger", "trop-gros"],
)
def test_predict_rejects_bad_uploads(
    client: TestClient, data: bytes, content_type: str, expected: int
) -> None:
    response = client.post("/v1/predict", files=_upload(data, content_type))
    assert response.status_code == expected
    assert "detail" in response.json()


def test_predict_requires_a_file(client: TestClient) -> None:
    assert client.post("/v1/predict").status_code == 422


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"
    assert float(response.headers["x-process-time-ms"]) >= 0


def test_malicious_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": 'x" injected=1'})
    assert response.headers["x-request-id"] != 'x" injected=1'


def test_demo_page_and_openapi(client: TestClient) -> None:
    assert "MedXRay" in client.get("/").text
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/predict", "/v1/explain", "/v1/model", "/health", "/ready"} <= set(paths)


def test_app_fails_fast_when_model_is_missing(tmp_path: Path) -> None:
    app = create_app(Settings(model_path=tmp_path / "absent.pt"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass
