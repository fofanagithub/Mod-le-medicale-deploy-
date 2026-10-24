"""Grad-CAM (Selvaraju et al., 2017), implémenté sans hook.

Principe : pour la carte d'activations A ∈ R^{K×h×w} de la dernière couche convolutive et
un score cible y,

    α_k  = moyenne_{i,j} ∂y / ∂A_k[i, j]          (importance du canal k)
    CAM  = ReLU( Σ_k α_k · A_k )                    (zones qui *augmentent* le score)

puis la carte h×w est ré-échantillonnée à la taille de l'entrée et normalisée dans [0, 1].

Pourquoi sans hook ? Les implémentations classiques posent des `register_forward_hook` /
`register_full_backward_hook` sur le modèle et stockent les activations dans un attribut.
En production, le même modèle sert des requêtes concurrentes (pool de threads de FastAPI) :
deux requêtes peuvent alors écraser mutuellement leurs activations. Ici on calcule
explicitement `A = features(x)` puis `y = head(A)`, et on dérive y par rapport à A avec
`torch.autograd.grad`. C'est une fonction pure : pas d'état partagé, donc thread-safe.

Bonus : comme A est détaché du graphe du backbone, la rétropropagation ne traverse que la
tête (pooling + linéaire). Le surcoût de Grad-CAM par rapport à une prédiction est minime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def grad_cam(
    features: nn.Module,
    head: nn.Module,
    x: Tensor,
    sign: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor]:
    """Calcule les cartes Grad-CAM d'un lot d'images.

    Args:
        features: partie convolutive ; renvoie des activations (B, K, h, w).
        head: tête de classification ; renvoie un logit par image (B,) ou (B, 1).
        x: lot d'images (B, C, H, W), déjà prétraité.
        sign: +1 pour expliquer la classe positive (pneumonie), -1 pour la classe négative
            (normal). Scalaire ou tenseur (B,) pour choisir par image.

    Returns:
        (logits (B,), cams (B, H, W) dans [0, 1]).

    Ne pas appeler sous `torch.inference_mode()` : Grad-CAM a besoin de l'autograd.
    """
    with torch.no_grad():
        activations = features(x)
    activations = activations.detach().requires_grad_(True)

    with torch.enable_grad():
        logits = head(activations).reshape(-1)
        scores = logits * torch.as_tensor(sign, dtype=logits.dtype, device=logits.device)
        # Les images d'un lot sont indépendantes en mode eval : le gradient de la somme
        # donne, pour chaque image, le gradient de son propre score.
        (grads,) = torch.autograd.grad(scores.sum(), activations)

    weights = grads.mean(dim=(2, 3), keepdim=True)  # α_k : (B, K, 1, 1)
    cam = F.relu((weights * activations.detach()).sum(dim=1, keepdim=True))  # (B, 1, h, w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits.detach(), _normalize_per_sample(cam.squeeze(1))


def _normalize_per_sample(cam: Tensor, eps: float = 1e-8) -> Tensor:
    """Ramène chaque carte dans [0, 1]. Une carte nulle (aucune évidence) reste nulle."""
    flat = cam.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1)
    span = hi - lo
    return torch.where(span > eps, (cam - lo) / span.clamp_min(eps), torch.zeros_like(cam))
