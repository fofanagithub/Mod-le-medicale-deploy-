from __future__ import annotations

import torch
from torch import nn

from medxray.gradcam import grad_cam
from medxray.model import XRayClassifier


def _linear_head(weights: list[float]) -> nn.Module:
    linear = nn.Linear(len(weights), 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([weights]))
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), linear)


def _two_region_activations(batch: int = 1) -> torch.Tensor:
    """Canal 0 actif en haut à gauche, canal 1 actif en bas à droite."""
    acts = torch.zeros(batch, 2, 8, 8)
    acts[:, 0, :4, :4] = 1.0
    acts[:, 1, 4:, 4:] = 1.0
    return acts


def test_cam_localizes_the_evidence() -> None:
    """Cas analytique : seul le canal 0 pèse positivement, Grad-CAM doit allumer son coin.

    C'est un test de *correction* de l'algorithme, pas seulement de forme du résultat.
    """
    _, cam = grad_cam(nn.Identity(), _linear_head([1.0, -1.0]), _two_region_activations())
    assert cam.shape == (1, 8, 8)
    assert cam[0, :4, :4].min() > 0.99
    assert cam[0, 4:, 4:].max() < 1e-6


def test_negative_sign_explains_the_other_class() -> None:
    acts = _two_region_activations()
    _, cam = grad_cam(nn.Identity(), _linear_head([1.0, -1.0]), acts, sign=-1.0)
    assert cam[0, 4:, 4:].min() > 0.99
    assert cam[0, :4, :4].max() < 1e-6


def test_per_sample_sign() -> None:
    acts = _two_region_activations(batch=2)
    sign = torch.tensor([1.0, -1.0])
    _, cam = grad_cam(nn.Identity(), _linear_head([1.0, -1.0]), acts, sign=sign)
    assert cam[0, :4, :4].min() > 0.99
    assert cam[1, 4:, 4:].min() > 0.99


def test_cam_without_positive_evidence_is_zero() -> None:
    _, cam = grad_cam(nn.Identity(), _linear_head([-1.0]), torch.ones(1, 1, 8, 8))
    assert torch.count_nonzero(cam) == 0


def test_cam_on_real_model_shape_range_and_logits() -> None:
    torch.manual_seed(0)
    model = XRayClassifier(pretrained=False).eval()
    x = torch.randn(3, 3, 224, 224)
    logits, cams = grad_cam(model.features, model.head, x)

    assert cams.shape == (3, 224, 224)
    assert cams.min() >= 0
    assert cams.max() <= 1
    with torch.no_grad():
        torch.testing.assert_close(logits, model(x))


def test_cam_is_side_effect_free() -> None:
    """Pas de gradient accumulé dans les poids, pas de hook : appelable en concurrence."""
    torch.manual_seed(0)
    model = XRayClassifier(pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    _, first = grad_cam(model.features, model.head, x)
    _, second = grad_cam(model.features, model.head, x)

    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks and not m._backward_hooks for m in model.modules())
    torch.testing.assert_close(first, second)
