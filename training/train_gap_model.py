"""
train_gap_model.py
==================
Trains the Multi-Task Research Gap Detection Network in PyTorch.
Evaluates evidence features, limitation frequencies, coverage ratios, and relationship scores.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.gap_detection_model import ResearchGapDetectionNetwork, GAP_TYPE_CLASSES
from app.features.gap_features import GAP_FEATURE_COLUMNS
from app.logging_config import logger

MODELS_DIR = PROJECT_ROOT / "models" / "gap_detection_model"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_gap_training_data(n_samples: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes labeled candidate gap training samples based on domain heuristics.
    Features: (N, 12)
    Targets:
      - gap_score: (N, 1) in [0, 1]
      - gap_type: (N,) int in [0, len(GAP_TYPE_CLASSES)-1]
      - confidence: (N, 1) in [0, 1]
    """
    np.random.seed(42)
    features = []
    gap_scores = []
    gap_types = []
    confidences = []

    for i in range(n_samples):
        # Sample type
        t_idx = i % (len(GAP_TYPE_CLASSES) - 1)  # reserve last class for insufficient evidence
        if i % 10 == 0:
            t_idx = 5  # INSUFFICIENT_EVIDENCE

        # Formulate realistic feature distributions per type
        if t_idx == 0:  # TECHNICAL_GAP
            passage_cnt = np.random.uniform(0.3, 0.9)
            conf = np.random.uniform(0.8, 0.98)
            p_lim = np.random.uniform(0.5, 1.0)
            pt_lim = np.random.uniform(0.1, 0.4)
            rigid = np.random.uniform(0.1, 0.5)
            cov_ratio = np.random.uniform(0.4, 0.8)
            tech_nov = np.random.uniform(0.7, 0.95)
            mean_rel = np.random.uniform(0.6, 0.9)
            min_rel = np.random.uniform(0.4, 0.7)
            bm25 = np.random.uniform(0.6, 0.9)
            faiss = np.random.uniform(0.65, 0.92)
            breadth = np.random.uniform(0.2, 0.5)
            score = np.random.uniform(0.80, 0.95)
        elif t_idx == 1:  # RESEARCH_TO_APPLICATION_GAP
            passage_cnt = np.random.uniform(0.4, 0.8)
            conf = np.random.uniform(0.75, 0.95)
            p_lim = np.random.uniform(0.3, 0.7)
            pt_lim = np.random.uniform(0.1, 0.3)
            rigid = np.random.uniform(0.1, 0.3)
            cov_ratio = np.random.uniform(0.8, 1.0)  # high academic, low patent
            tech_nov = np.random.uniform(0.75, 0.95)
            mean_rel = np.random.uniform(0.4, 0.7)
            min_rel = np.random.uniform(0.2, 0.5)
            bm25 = np.random.uniform(0.5, 0.8)
            faiss = np.random.uniform(0.6, 0.85)
            breadth = np.random.uniform(0.3, 0.7)
            score = np.random.uniform(0.75, 0.92)
        elif t_idx == 2:  # COMBINATION_GAP
            passage_cnt = np.random.uniform(0.3, 0.7)
            conf = np.random.uniform(0.75, 0.92)
            p_lim = np.random.uniform(0.2, 0.6)
            pt_lim = np.random.uniform(0.2, 0.6)
            rigid = np.random.uniform(0.2, 0.5)
            cov_ratio = np.random.uniform(0.3, 0.7)
            tech_nov = np.random.uniform(0.85, 1.0)  # very high novelty
            mean_rel = np.random.uniform(0.3, 0.6)
            min_rel = np.random.uniform(0.1, 0.4)
            bm25 = np.random.uniform(0.4, 0.75)
            faiss = np.random.uniform(0.5, 0.8)
            breadth = np.random.uniform(0.6, 1.0)
            score = np.random.uniform(0.82, 0.96)
        elif t_idx == 3:  # PATENT_COVERAGE_GAP
            passage_cnt = np.random.uniform(0.4, 0.85)
            conf = np.random.uniform(0.8, 0.96)
            p_lim = np.random.uniform(0.2, 0.5)
            pt_lim = np.random.uniform(0.6, 1.0)  # high patent limitation
            rigid = np.random.uniform(0.7, 1.0)   # high rigid claim presence
            cov_ratio = np.random.uniform(0.2, 0.5)
            tech_nov = np.random.uniform(0.6, 0.85)
            mean_rel = np.random.uniform(0.5, 0.8)
            min_rel = np.random.uniform(0.3, 0.6)
            bm25 = np.random.uniform(0.6, 0.85)
            faiss = np.random.uniform(0.65, 0.9)
            breadth = np.random.uniform(0.3, 0.6)
            score = np.random.uniform(0.85, 0.95)
        elif t_idx == 4:  # APPLICATION_GAP
            passage_cnt = np.random.uniform(0.3, 0.75)
            conf = np.random.uniform(0.75, 0.9)
            p_lim = np.random.uniform(0.4, 0.8)
            pt_lim = np.random.uniform(0.3, 0.7)
            rigid = np.random.uniform(0.3, 0.6)
            cov_ratio = np.random.uniform(0.4, 0.7)
            tech_nov = np.random.uniform(0.7, 0.9)
            mean_rel = np.random.uniform(0.5, 0.75)
            min_rel = np.random.uniform(0.3, 0.55)
            bm25 = np.random.uniform(0.55, 0.8)
            faiss = np.random.uniform(0.6, 0.82)
            breadth = np.random.uniform(0.4, 0.7)
            score = np.random.uniform(0.78, 0.90)
        else:  # INSUFFICIENT_EVIDENCE
            passage_cnt = np.random.uniform(0.0, 0.2)
            conf = np.random.uniform(0.3, 0.6)
            p_lim = np.random.uniform(0.0, 0.2)
            pt_lim = np.random.uniform(0.0, 0.2)
            rigid = np.random.uniform(0.0, 0.2)
            cov_ratio = np.random.uniform(0.1, 0.5)
            tech_nov = np.random.uniform(0.2, 0.5)
            mean_rel = np.random.uniform(0.1, 0.4)
            min_rel = np.random.uniform(0.0, 0.2)
            bm25 = np.random.uniform(0.1, 0.4)
            faiss = np.random.uniform(0.2, 0.45)
            breadth = np.random.uniform(0.1, 0.3)
            score = np.random.uniform(0.15, 0.40)

        feat_vec = [
            passage_cnt, conf, p_lim, pt_lim, rigid, cov_ratio,
            tech_nov, mean_rel, min_rel, bm25, faiss, breadth
        ]
        features.append(feat_vec)
        gap_scores.append(score)
        gap_types.append(t_idx)
        confidences.append(conf)

    X = np.array(features, dtype=np.float32)
    y_score = np.array(gap_scores, dtype=np.float32).reshape(-1, 1)
    y_type = np.array(gap_types, dtype=np.int64)
    y_conf = np.array(confidences, dtype=np.float32).reshape(-1, 1)

    return X, y_score, y_type, y_conf


def train_gap_detection_model(
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[ResearchGapDetectionNetwork, Dict[str, Any]]:
    """Trains the Multi-Task ResearchGapDetectionNetwork with PyTorch."""
    torch.manual_seed(42)
    np.random.seed(42)

    X, y_score, y_type, y_conf = generate_synthetic_gap_training_data(n_samples=200)

    # 70% Train, 30% Val
    n_train = int(len(X) * 0.75)
    X_train, X_val = torch.tensor(X[:n_train]), torch.tensor(X[n_train:])
    y_score_tr, y_score_val = torch.tensor(y_score[:n_train]), torch.tensor(y_score[n_train:])
    y_type_tr, y_type_val = torch.tensor(y_type[:n_train]), torch.tensor(y_type[n_train:])
    y_conf_tr, y_conf_val = torch.tensor(y_conf[:n_train]), torch.tensor(y_conf[n_train:])

    model = ResearchGapDetectionNetwork(num_features=12, num_classes=len(GAP_TYPE_CLASSES)).to(device)

    crit_score = nn.BCELoss()
    crit_type = nn.CrossEntropyLoss()
    crit_conf = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info(f"Starting Multi-Task Gap Detection Network training ({epochs} epochs)...")
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "val_type_acc": []}

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        p_scores, p_types, p_confs, _ = model(X_train)

        l_s = crit_score(p_scores, y_score_tr)
        l_t = crit_type(p_types, y_type_tr)
        l_c = crit_conf(p_confs, y_conf_tr)
        loss = l_s + (0.8 * l_t) + (0.5 * l_c)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_scores, v_types, v_confs, _ = model(X_val)
            vl_s = crit_score(v_scores, y_score_val)
            vl_t = crit_type(v_types, y_type_val)
            vl_c = crit_conf(v_confs, y_conf_val)
            val_loss = float(vl_s + (0.8 * vl_t) + (0.5 * vl_c))

            val_preds = torch.argmax(v_types, dim=-1).numpy()
            type_acc = float((val_preds == y_type_val.numpy()).mean())

        history["train_loss"].append(round(loss.item(), 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_type_acc"].append(round(type_acc, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                str(MODELS_DIR / "gap_detection_model.pt"),
                metadata={"best_epoch": epoch, "val_loss": val_loss, "val_type_acc": type_acc}
            )

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Type Acc: {type_acc:.3f}")

    training_time = time.time() - start_time
    history["training_time_s"] = round(training_time, 2)

    with open(REPORTS_DIR / "gap_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Gap Detection Network training complete in {training_time:.1f}s. Saved to {MODELS_DIR / 'gap_detection_model.pt'}")
    return model, history


if __name__ == "__main__":
    train_gap_detection_model()
