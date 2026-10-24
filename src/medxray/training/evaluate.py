"""Évaluation d'un checkpoint sur le jeu de test et génération des rapports.

Usage :
    medxray-evaluate --checkpoint models/model.pt

Produit dans `reports/` : metrics.json, roc_curve.png, confusion_matrix.png,
gradcam_examples.png. Exécutable indépendamment de l'entraînement : n'importe qui peut
re-vérifier les chiffres annoncés à partir du checkpoint publié.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # rendu sans affichage (serveur, CI)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve
from torch import nn

from medxray.gradcam import grad_cam
from medxray.inference import Predictor
from medxray.model import XRayClassifier, load_checkpoint
from medxray.preprocessing import to_model_input
from medxray.training.data import make_dataset, make_loader
from medxray.training.metrics import binary_metrics, bootstrap_ci

logger = logging.getLogger("medxray.evaluate")


@torch.inference_mode()
def predict_probs(
    model: nn.Module, loader: torch.utils.data.DataLoader[Any]
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x)).numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def evaluate_model(
    model: nn.Module,
    threshold: float,
    data_dir: Path,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    loader = make_loader(make_dataset("test", data_dir), batch_size, False, limit=limit)
    y, p = predict_probs(model, loader)
    return {
        "metrics": binary_metrics(y, p, threshold),
        "ci95": bootstrap_ci(y, p, threshold),
        "y_true": y,
        "y_prob": p,
    }


def write_reports(checkpoint: Path, data_dir: Path, out_dir: Path, test: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor.from_checkpoint(checkpoint)
    y, p = test["y_true"], test["y_prob"]
    threshold = predictor.metadata.threshold

    raw_test = make_dataset("test", data_dir, raw=True)
    sanity = gradcam_randomization_check(predictor.model, raw_test, n=min(32, len(y)))

    report = {
        "model_version": predictor.metadata.model_version,
        "sha256": predictor.sha256,
        "threshold": threshold,
        "test": test["metrics"],
        "test_ci95": test["ci95"],
        "validation": predictor.metadata.metrics.get("validation"),
        "gradcam_sanity_check": sanity,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_roc(y, p, test["metrics"], out_dir / "roc_curve.png")
    _plot_confusion(test["metrics"]["confusion_matrix"], out_dir / "confusion_matrix.png")
    _plot_score_distribution(y, p, threshold, out_dir / "score_distribution.png")
    history = out_dir / "training_history.json"
    if history.exists():
        plot_training_history(history, out_dir / "training_curves.png")
    _plot_gradcam_gallery(predictor, raw_test, y, p, threshold, out_dir / "gradcam_examples.png")
    logger.info("Rapports écrits dans %s", out_dir)


def gradcam_randomization_check(
    model: XRayClassifier, raw_test: Any, n: int = 32
) -> dict[str, Any]:
    """Test de randomisation des paramètres (Adebayo et al., 2018, « Sanity Checks for
    Saliency Maps »).

    Si les cartes Grad-CAM restent quasi identiques quand on ré-initialise aléatoirement
    les dernières couches, elles reflètent la structure de l'image (contours) et non ce que
    le modèle a appris : l'explication serait alors trompeuse. On attend une corrélation de
    rang *faible* entre cartes du modèle entraîné et du modèle randomisé.
    """
    randomized = copy.deepcopy(model)
    torch.manual_seed(0)
    for module in [*randomized.features[-1].modules(), *randomized.head.modules()]:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    randomized.eval()

    x = torch.cat([to_model_input(raw_test[i][0]) for i in range(n)])
    _, cams_trained = grad_cam(model.features, model.head, x)
    _, cams_random = grad_cam(randomized.features, randomized.head, x)
    rhos = []
    for a, b in zip(cams_trained, cams_random, strict=True):
        a_flat, b_flat = a.flatten().numpy(), b.flatten().numpy()
        if a_flat.std() == 0 or b_flat.std() == 0:  # carte constante : aucune corrélation
            rhos.append(0.0)
            continue
        rhos.append(float(spearmanr(a_flat, b_flat).statistic))
    return {
        "method": "randomisation de layer4 + tête, corrélation de Spearman des cartes",
        "n_images": n,
        "mean_spearman": float(np.mean(rhos)),
        "interpretation": "proche de 0 = la carte dépend des poids appris (attendu)",
    }


def _plot_roc(y: np.ndarray, p: np.ndarray, metrics: dict[str, Any], path: Path) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.scatter(
        [1 - metrics["specificity"]],
        [metrics["sensitivity"]],
        color="crimson",
        zorder=3,
        label=f"seuil {metrics['threshold']:.2f}",
    )
    ax.set(xlabel="1 - spécificité", ylabel="Sensibilité", title="Courbe ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(cm: dict[str, int], path: Path) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontsize=14,
            color="white" if v > matrix.max() / 2 else "black",
        )
    labels = ["normal", "pneumonie"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Prédit",
        ylabel="Réel",
        title="Matrice de confusion (test)",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_score_distribution(y: np.ndarray, p: np.ndarray, threshold: float, path: Path) -> None:
    """Histogramme des scores par classe réelle : montre la séparation et les cas ambigus."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, 1, 41).tolist()
    ax.hist(p[y == 0], bins=bins, alpha=0.7, label="réel : normal", color="#1e8449")
    ax.hist(p[y == 1], bins=bins, alpha=0.7, label="réel : pneumonie", color="#c0392b")
    ax.axvline(threshold, color="black", ls="--", lw=1, label=f"seuil {threshold:.2f}")
    ax.set_yscale("log")
    ax.set(
        xlabel="P(pneumonie)",
        ylabel="Nombre d'images (log)",
        title="Distribution des scores (test)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_history(history_json: Path, path: Path) -> None:
    """Perte d'entraînement et AUROC de validation par époque."""
    history = json.loads(history_json.read_text(encoding="utf-8"))["history"]
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-")
    ax1.set(xlabel="Époque", ylabel="Perte BCE pondérée", title="Perte d'entraînement")
    ax2.plot(epochs, [h["val_auroc"] for h in history], "o-", color="darkorange")
    best = max(history, key=lambda h: h["val_auroc"])
    ax2.scatter(
        [best["epoch"]],
        [best["val_auroc"]],
        s=120,
        facecolors="none",
        edgecolors="crimson",
        label=f"retenue (époque {best['epoch']})",
    )
    ax2.set(xlabel="Époque", ylabel="AUROC", title="AUROC de validation")
    ax2.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gradcam_gallery(
    predictor: Predictor, raw_test: Any, y: np.ndarray, p: np.ndarray, threshold: float, path: Path
) -> None:
    """Exemples représentatifs : succès ET erreurs (les erreurs sont les plus instructives)."""
    pred = (p >= threshold).astype(int)
    groups = {
        "Vrai positif": np.where((y == 1) & (pred == 1))[0],
        "Vrai négatif": np.where((y == 0) & (pred == 0))[0],
        "Faux négatif": np.where((y == 1) & (pred == 0))[0],
        "Faux positif": np.where((y == 0) & (pred == 1))[0],
    }
    chosen = [(name, int(i)) for name, idx in groups.items() for i in idx[:2]]
    if not chosen:
        return
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.6 * len(chosen), 5.6), squeeze=False)
    for col, (name, i) in enumerate(chosen):
        image = raw_test[i][0]
        explanation = predictor.explain(image)
        axes[0, col].imshow(image, cmap="gray")
        axes[0, col].set_title(f"{name}\np={p[i]:.2f}", fontsize=9)
        axes[1, col].imshow(explanation.overlay)
        axes[1, col].set_title(f"Grad-CAM : {explanation.explained_class}", fontsize=9)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, metadata = load_checkpoint(args.checkpoint)
    test = evaluate_model(model, metadata.threshold, args.data_dir)
    m = test["metrics"]
    logger.info(
        "Test : AUROC=%.3f sensibilité=%.3f spécificité=%.3f",
        m["auroc"],
        m["sensitivity"],
        m["specificity"],
    )
    write_reports(args.checkpoint, args.data_dir, args.reports_dir, test)


if __name__ == "__main__":
    main()
