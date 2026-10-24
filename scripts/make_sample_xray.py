"""Génère une image PNG de test (fausse radio) pour les smoke tests de l'API.

python scripts/make_sample_xray.py sample.png
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:512, 0:512] / 512.0
    lungs = np.exp(-(((x - 0.3) / 0.15) ** 2 + ((y - 0.5) / 0.3) ** 2))
    lungs += np.exp(-(((x - 0.7) / 0.15) ** 2 + ((y - 0.5) / 0.3) ** 2))
    image = np.clip(0.8 - 0.6 * lungs + 0.05 * rng.standard_normal(x.shape), 0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(output)
    print(f"Image écrite : {output}")
