"""Chargement de PneumoniaMNIST (MedMNIST v2).

Jeu choisi pour sa reproductibilité : téléchargeable sans compte, licence CC BY 4.0, et
splits train/val/test *officiels* (pas de fuite entre patients introduite par un split
fait maison). Images d'origine : Kermany et al., 2018 (enfants de 1 à 5 ans, Guangzhou).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader, Dataset, Subset

from medxray.preprocessing import IMAGE_SIZE, eval_transform, train_transform

Split = Literal["train", "val", "test"]


def label_to_float(target: np.ndarray) -> float:
    """MedMNIST renvoie un tableau (1,) : on le convertit en flottant pour BCEWithLogits.

    Fonction de module (et non lambda) pour rester sérialisable par les workers du DataLoader.
    """
    return float(target[0])


def make_dataset(
    split: Split, data_dir: Path, augment: bool = False, raw: bool = False
) -> PneumoniaMNIST:
    """`raw=True` renvoie des images PIL non transformées (pour l'affichage / Grad-CAM)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = None if raw else (train_transform() if augment else eval_transform())
    return PneumoniaMNIST(
        split=split,
        size=IMAGE_SIZE,
        root=str(data_dir),
        download=True,
        transform=transform,
        target_transform=None if raw else label_to_float,
    )


def labels_of(dataset: PneumoniaMNIST) -> np.ndarray:
    return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    limit: int | None = None,
) -> DataLoader[Any]:
    """`limit` restreint le jeu aux N premiers exemples (smoke test rapide du pipeline)."""
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))  # type: ignore[arg-type]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
