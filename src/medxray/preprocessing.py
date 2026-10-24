"""Prétraitement des images : source unique de vérité pour l'entraînement ET le serving.

Le *training/serving skew* (un prétraitement subtilement différent entre l'entraînement et
la production) est l'une des causes les plus fréquentes de dégradation silencieuse d'un
modèle déployé. Toute transformation appliquée à une image passe donc par ce module, et
les deux pipelines (`train_transform`, `eval_transform`) partagent la même géométrie et la
même normalisation.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor
from torchvision import transforms as T

IMAGE_SIZE = 224
# Le backbone est pré-entraîné sur ImageNet : on réutilise sa normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ALLOWED_FORMATS = frozenset({"PNG", "JPEG"})
MIN_IMAGE_SIDE = 64
# Garde-fou contre les "decompression bombs" (un PNG de quelques Ko qui se décompresse en
# plusieurs Go). Vérifié sur l'en-tête, AVANT le décodage des pixels.
MAX_IMAGE_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """L'entrée n'est pas une image exploitable (format, taille, contenu)."""


def load_image(data: bytes) -> Image.Image:
    """Décode des octets en image PIL en niveaux de gris ('L'), après validation.

    On se fie au contenu réel du fichier (signature décodée par Pillow), jamais à
    l'extension ou au Content-Type déclarés par le client.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    f"Format non supporté : {img.format}. Formats acceptés : PNG, JPEG."
                )
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(f"Image trop grande ({width}x{height} pixels).")
            if min(width, height) < MIN_IMAGE_SIDE:
                raise InvalidImageError(
                    f"Image trop petite ({width}x{height}), minimum {MIN_IMAGE_SIDE} px de côté."
                )
            # Respecte l'orientation EXIF (photos de radios prises au smartphone).
            oriented = ImageOps.exif_transpose(img)
            return to_grayscale(oriented)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise InvalidImageError("Fichier illisible ou corrompu.") from exc


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convertit n'importe quel mode PIL en 'L' (8 bits), en gérant le 16 bits.

    Les radiographies sont souvent exportées en PNG 16 bits : un simple `convert("L")`
    saturerait toutes les valeurs > 255 et produirait une image quasi blanche. On fait donc
    une remise à l'échelle min-max explicite.
    """
    if img.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        scaled = (arr - lo) / max(hi - lo, 1e-6) * 255.0
        return Image.fromarray(scaled.round().astype(np.uint8))
    return img.convert("L")


def _normalize() -> list[T.ToTensor | T.Normalize]:
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def eval_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Pipeline déterministe utilisé en validation, test ET en production.

    Resize(côté court) + CenterCrop reproduit le recadrage carré centré appliqué par
    MedMNIST lors de la création du jeu de données : une radio plein format envoyée à
    l'API subit donc la même géométrie que les images d'entraînement.
    """
    return T.Compose(
        [
            T.Grayscale(num_output_channels=3),  # 1 canal -> 3 canaux pour le backbone ImageNet
            T.Resize(image_size),
            T.CenterCrop(image_size),
            *_normalize(),
        ]
    )


def train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Augmentations légères et *plausibles cliniquement*.

    Pas de flip horizontal : l'anatomie thoracique n'est pas symétrique (cœur à gauche) et
    un flip créerait des images qui n'existent pas en pratique. Rotations et translations
    restent faibles : elles simulent un mauvais positionnement du patient, pas plus.
    """
    return T.Compose(
        [
            T.Grayscale(num_output_channels=3),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomAffine(degrees=7, translate=(0.03, 0.03)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            *_normalize(),
        ]
    )


def to_model_input(image: Image.Image, image_size: int = IMAGE_SIZE) -> Tensor:
    """Image PIL -> tenseur (1, 3, H, W) prêt pour le modèle."""
    tensor: Tensor = eval_transform(image_size)(image)
    return tensor.unsqueeze(0)


def display_crop(image: Image.Image, size: int) -> Image.Image:
    """Recadrage identique à `eval_transform` (sans normalisation), pour l'affichage.

    Garantit que la heatmap Grad-CAM se superpose pixel pour pixel à ce que le modèle a vu.
    """
    crop: Image.Image = T.Compose([T.Resize(size), T.CenterCrop(size)])(image.convert("L"))
    return crop
