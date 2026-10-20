"""
train_relationship_model.py
===========================
Trains the Siamese / Dual-Tower + Multi-Head Cross-Attention Relationship Model in PyTorch.
Integrates RAG feature fusion, early stopping, and class-weighted BCE loss.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS
from app.retrieval.embeddings import EmbeddingGenerator
from app.logging_config import logger

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "relationship_model"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class PairwiseDataset(Dataset):
    """PyTorch Dataset for (Paper, Patent, RAG_features, Label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        paper_emb_dict: Dict[str, np.ndarray],
        patent_emb_dict: Dict[str, np.ndarray],
        scaler
    ):
        self.df = df.reset_index(drop=True)
        self.paper_emb_dict = paper_emb_dict
        self.patent_emb_dict = patent_emb_dict
        self.scaler = scaler

        raw_features = self.df[RELATIONSHIP_FEATURE_COLUMNS].values
        self.scaled_features = self.scaler.transform(raw_features).astype(np.float32)
        self.labels = self.df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p_id = row["paper_id"]
        pt_id = row["patent_id"]

        p_emb = self.paper_emb_dict.get(p_id, np.zeros(384, dtype=np.float32))
        pt_emb = self.patent_emb_dict.get(pt_id, np.zeros(384, dtype=np.float32))
        rag_feat = self.scaled_features[idx]
        label = self.labels[idx]

        return (
            torch.tensor(p_emb, dtype=torch.float32),
            torch.tensor(pt_emb, dtype=torch.float32),
            torch.tensor(rag_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32).unsqueeze(-1)
        )


def train_dual_tower_model(
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[DualTowerCrossAttentionRelationshipModel, Dict[str, Any]]:
    """Trains the DualTowerCrossAttentionRelationshipModel with PyTorch."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data
    train_df = pd.read_csv(DATA_TRAINING / "train.csv")
    val_df = pd.read_csv(DATA_TRAINING / "validation.csv")
    test_df = pd.read_csv(DATA_TRAINING / "test.csv")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")

    # 2. Compute embeddings
    encoder = EmbeddingGenerator()
    df_papers = pd.read_csv(DATA_PROCESSED / "papers_clean.csv")
    df_patents = pd.read_csv(DATA_PROCESSED / "patents_clean.csv")

    paper_emb_dict = {}
    for _, row in df_papers.iterrows():
        text = f"{row['title']}. {str(row.get('abstract', ''))}"
        paper_emb_dict[row["paper_id"]] = encoder.encode([text])[0]

    patent_emb_dict = {}
    for _, row in df_patents.iterrows():
        text = f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}"
        patent_emb_dict[row["patent_id"]] = encoder.encode([text])[0]

    train_ds = PairwiseDataset(train_df, paper_emb_dict, patent_emb_dict, scaler)
    val_ds = PairwiseDataset(val_df, paper_emb_dict, patent_emb_dict, scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model setup
    model = DualTowerCrossAttentionRelationshipModel(
        embed_dim=384,
        num_rag_features=len(RELATIONSHIP_FEATURE_COLUMNS),
        num_heads=4,
        hidden_dim=256,
        dropout=0.25
    ).to(device)

    # Handle class imbalance in loss
    n_pos = float(train_df["label"].sum())
    n_neg = float(len(train_df) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_auc": []
    }

    best_val_loss = float("inf")
    patience = 12
    patience_counter = 0

    logger.info(f"Starting Dual-Tower Relationship Model training ({epochs} epochs, {len(train_ds)} train, {len(val_ds)} val)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds, train_targets = [], []

        for p_emb, pt_emb, rag_feat, labels in train_loader:
            p_emb, pt_emb, rag_feat, labels = p_emb.to(device), pt_emb.to(device), rag_feat.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(p_emb, pt_emb, rag_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_targets.extend(labels.cpu().numpy().flatten())

        train_loss = float(np.mean(train_losses))
        train_acc = float(accuracy_score(np.array(train_targets) > 0.5, np.array(train_preds) > 0.5))
        train_f1 = float(f1_score(np.array(train_targets) > 0.5, np.array(train_preds) > 0.5, zero_division=0))

        # Validation
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for p_emb, pt_emb, rag_feat, labels in val_loader:
                p_emb, pt_emb, rag_feat, labels = p_emb.to(device), pt_emb.to(device), rag_feat.to(device), labels.to(device)
                outputs, _ = model(p_emb, pt_emb, rag_feat)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(labels.cpu().numpy().flatten())

        val_loss = float(np.mean(val_losses))
        val_acc = float(accuracy_score(np.array(val_targets) > 0.5, np.array(val_preds) > 0.5))
        val_f1 = float(f1_score(np.array(val_targets) > 0.5, np.array(val_preds) > 0.5, zero_division=0))
        val_auc = float(roc_auc_score(val_targets, val_preds)) if len(set(val_targets)) > 1 else 0.5

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["train_f1"].append(round(train_f1, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["val_auc"].append(round(val_auc, 4))

        logger.info(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} AUC: {val_auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_checkpoint(
                str(MODELS_DIR / "relationship_model.pt"),
                metadata={"best_epoch": epoch, "val_loss": val_loss, "val_auc": val_auc, "val_f1": val_f1}
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    training_time = time.time() - start_time
    history["total_training_time_s"] = round(training_time, 2)

    with open(REPORTS_DIR / "relationship_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Dual-Tower training complete in {training_time:.1f}s. Best val_loss={best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    train_dual_tower_model()
