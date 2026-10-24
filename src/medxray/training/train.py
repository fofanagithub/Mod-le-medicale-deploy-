"""Entraînement : fine-tuning d'un ResNet18 pré-entraîné ImageNet.

Usage :
    medxray-train --config configs/train.yaml
    medxray-train --config configs/train.yaml --limit 64 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from medxray.model import ModelMetadata, XRayClassifier, save_checkpoint
from medxray.training.data import labels_of, make_dataset, make_loader
from medxray.training.evaluate import evaluate_model, predict_probs, write_reports
from medxray.training.metrics import binary_metrics, select_threshold

logger = logging.getLogger("medxray.train")


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    output: Path = Path("models/model.pt")
    reports_dir: Path = Path("reports")
    model_version: str = "1.0.0"
    epochs: int = 8
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    target_sensitivity: float = 0.95
    num_workers: int = 0
    seed: int = 42
    limit: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Clés inconnues dans {path} : {sorted(unknown)}")
        for key in ("data_dir", "output", "reports_dir"):
            if key in raw:
                raw[key] = Path(raw[key])
        return cls(**raw)


def set_seed(seed: int) -> None:
    """Reproductibilité : mêmes données + même config => mêmes poids (sur CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total, count = 0.0, 0
    for step, (x, y) in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y.float())
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
        if step % 20 == 0:
            logger.info("  step %d/%d  loss=%.4f", step, len(loader), total / count)
    return total / max(count, 1)


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    started = time.perf_counter()

    train_ds = make_dataset("train", cfg.data_dir, augment=True)
    val_ds = make_dataset("val", cfg.data_dir)
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.limit)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.limit)

    # Déséquilibre (≈ 74 % de pneumonies en train) : pos_weight = n_neg / n_pos < 1
    # rééquilibre la contribution des deux classes à la perte.
    y_train = labels_of(train_ds)
    pos_weight = torch.tensor((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.float())
    logger.info("Train=%d Val=%d pos_weight=%.3f", len(train_ds), len(val_ds), pos_weight)

    model = XRayClassifier(pretrained=True)
    # Taux d'apprentissage différenciés : le backbone pré-entraîné bouge peu, la tête
    # (initialisée aléatoirement) apprend 10x plus vite.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": cfg.lr_backbone},
            {"params": model.head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_auc, best_state, best_epoch, stale = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        y_val, p_val = predict_probs(model, val_loader)
        val_auc = binary_metrics(y_val, p_val, 0.5)["auroc"]
        history.append({"epoch": epoch, "train_loss": loss, "val_auroc": val_auc})
        logger.info(
            "Epoch %d/%d  loss=%.4f  val_auroc=%.4f  (%.0fs)",
            epoch,
            cfg.epochs,
            loss,
            val_auc,
            time.perf_counter() - t0,
        )
        # Sélection de modèle sur l'AUROC de validation (indépendante du seuil).
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("Early stopping (pas d'amélioration depuis %d époques)", stale)
                break

    model.load_state_dict(best_state)
    model.eval()

    y_val, p_val = predict_probs(model, val_loader)
    threshold = select_threshold(y_val, p_val, cfg.target_sensitivity)
    val_metrics = binary_metrics(y_val, p_val, threshold)
    logger.info("Seuil retenu (sensibilité cible %.2f) : %.4f", cfg.target_sensitivity, threshold)

    # Le test n'est évalué qu'UNE fois, avec le modèle et le seuil déjà figés.
    test = evaluate_model(model, threshold, cfg.data_dir, cfg.batch_size, cfg.limit)
    metadata = ModelMetadata(
        model_version=cfg.model_version,
        threshold=threshold,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        metrics={
            "best_epoch": best_epoch,
            "target_sensitivity": cfg.target_sensitivity,
            "validation": val_metrics,
            "test": test["metrics"],
            "test_ci95": test["ci95"],
        },
        notes="Smoke test (--limit) : ne pas déployer." if cfg.limit else "",
    )
    save_checkpoint(cfg.output, model, metadata)
    logger.info("Checkpoint écrit : %s", cfg.output)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "training_history.json").write_text(
        json.dumps({"config": asdict(cfg), "history": history}, indent=2, default=str),
        encoding="utf-8",
    )
    write_reports(cfg.output, cfg.data_dir, cfg.reports_dir, test)
    logger.info("Terminé en %.1f min", (time.perf_counter() - started) / 60)
    return cfg.output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int, help="N exemples max par split (smoke test)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(args.config) if args.config.exists() else TrainConfig()
    for key in ("epochs", "limit", "output"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    train(cfg)


if __name__ == "__main__":
    main()
