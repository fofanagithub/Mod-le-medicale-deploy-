"""Crée un checkpoint aux poids aléatoires (CI : tester le build Docker sans entraîner).

python scripts/make_dummy_checkpoint.py models/model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from medxray.model import ModelMetadata, XRayClassifier, save_checkpoint

if __name__ == "__main__":
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "models/model.pt")
    torch.manual_seed(0)
    metadata = ModelMetadata(
        model_version="0.0.0-dummy",
        threshold=0.5,
        notes="Poids aléatoires générés pour la CI. NE PAS DÉPLOYER.",
    )
    save_checkpoint(output, XRayClassifier(pretrained=False), metadata)
    print(f"Checkpoint factice écrit : {output}")
