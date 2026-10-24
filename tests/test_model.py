from __future__ import annotations

from pathlib import Path

import pytest
import torch

from medxray.model import ModelMetadata, XRayClassifier, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = XRayClassifier(pretrained=False).eval()
    meta = ModelMetadata(model_version="9.9.9", threshold=0.37, metrics={"test": {"auroc": 0.9}})
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, meta)

    loaded, loaded_meta = load_checkpoint(path)
    assert loaded_meta == meta
    assert not loaded.training
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


def test_checkpoint_with_arbitrary_object_is_refused(tmp_path: Path) -> None:
    """`weights_only=True` : un checkpoint contenant un objet Python arbitraire est rejeté
    (un pickle malveillant pourrait sinon exécuter du code au chargement)."""
    path = tmp_path / "evil.pt"
    torch.save({"format_version": 1, "payload": Path("x")}, path)
    with pytest.raises(Exception, match="Weights only load failed"):
        load_checkpoint(path)


def test_unknown_checkpoint_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"format_version": 0}, path)
    with pytest.raises(ValueError, match="Format de checkpoint"):
        load_checkpoint(path)


def test_model_outputs_one_logit_per_image() -> None:
    model = XRayClassifier(pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(4, 3, 224, 224)).shape == (4,)
