"""Métriques et choix du seuil de décision.

En dépistage, rater une pneumonie (faux négatif) coûte bien plus cher qu'une seconde
lecture inutile (faux positif). On fixe donc une **sensibilité cible** sur la validation,
puis on choisit, parmi les seuils qui l'atteignent, celui qui maximise la spécificité. Le
seuil est ensuite figé et appliqué tel quel au jeu de test : le test n'intervient dans
aucune décision, sinon les métriques rapportées seraient optimistes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float) -> float:
    """Plus grand seuil atteignant la sensibilité cible (donc spécificité maximale)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # thresholds[0] vaut +inf par convention sklearn : on l'exclut.
    candidates = [
        (f, -t, t)
        for f, s, t in zip(fpr[1:], tpr[1:], thresholds[1:], strict=True)
        if s >= target_sensitivity
    ]
    if not candidates:
        return 0.5
    _, _, best = min(candidates)
    return float(min(best, 1.0))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    sensitivity = ratio(tp, tp + fn)
    ppv = ratio(tp, tp + fp)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": ratio(tp + tn, len(y_true)),
        "sensitivity": sensitivity,
        "specificity": ratio(tn, tn + fp),
        "ppv": ppv,
        "npv": ratio(tn, tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Intervalles de confiance à 95 % par bootstrap (percentiles).

    624 images de test, c'est peu : un point estimé « 92 % de sensibilité » sans intervalle
    ne permet pas de juger si le modèle est réellement meilleur qu'un autre.
    """
    rng = np.random.default_rng(seed)
    keys = ("auroc", "sensitivity", "specificity", "accuracy")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # AUROC indéfinie sans les deux classes
            continue
        m = binary_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in samples.items()
    }
