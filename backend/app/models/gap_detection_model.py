"""
gap_detection_model.py
======================
Evidence-Grounded Multi-Task Research Gap Detection Network in PyTorch.

PRIMARY RESPONSIBILITY:
  Evaluates candidate innovation gaps from retrieved evidence, corpus-level coverage,
  unresolved technological limitations, underexplored technology combinations, and
  paper-patent relationship scores.

OUTPUT:
  - Gap Score in [0, 1]
  - Gap Type (Categorical: TECHNICAL_GAP, RESEARCH_TO_APPLICATION_GAP, COMBINATION_GAP, etc.)
  - Gap Confidence in [0, 1]
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from app.logging_config import logger

GAP_TYPE_CLASSES = [
    "TECHNICAL_GAP",
    "RESEARCH_TO_APPLICATION_GAP",
    "COMBINATION_GAP",
    "PATENT_COVERAGE_GAP",
    "APPLICATION_GAP",
    "INSUFFICIENT_EVIDENCE"
]


class FeatureAttention(nn.Module):
    """Learns normalized attention weights across input feature dimensions."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn_weights = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.attn_weights(x)
        attended = x * weights
        return attended, weights


class ResearchGapDetectionNetwork(nn.Module):
    """
    Multi-Task Neural Network for Evidence-Grounded Research Gap Detection and Classification.
    
    Structure:
      Gap Candidate Features (d_features)
               │
               ▼
        Feature Encoder (128, LayerNorm, ReLU)
               │
               ▼
        Feature Attention Layer
               │
               ▼
        Dense (128, ReLU) + Dropout(0.25)
               │
               ▼
        Dense (64, ReLU) + Dropout(0.20)
               │
        ┌──────┼──────┐
        ▼      ▼      ▼
       Head 1 Head 2 Head 3
      GapScore GapType Confidence
    """

    def __init__(
        self,
        num_features: int = 12,
        num_classes: int = len(GAP_TYPE_CLASSES),
        dropout: float = 0.25
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # 1. Feature Encoding & Normalization
        self.input_encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Dynamic Feature Attention
        self.feature_attention = FeatureAttention(dim=128)

        # 3. Shared Representation Backbone
        self.backbone = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8)
        )

        # 4. Multi-Task Heads
        # Head 1: Gap Score / Probability ∈ [0, 1]
        self.gap_score_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Head 2: Categorical Gap Type Classification
        self.gap_type_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # Head 3: Evidence Grounding Confidence ∈ [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor of shape (batch_size, num_features)
            
        Returns:
            gap_scores: (batch_size, 1)
            gap_type_logits: (batch_size, num_classes)
            confidences: (batch_size, 1)
            attention_weights: (batch_size, 128)
        """
        encoded = self.input_encoder(features)
        attended, weights = self.feature_attention(encoded)
        rep = self.backbone(attended)

        gap_scores = self.gap_score_head(rep)
        gap_type_logits = self.gap_type_head(rep)
        confidences = self.confidence_head(rep)

        return gap_scores, gap_type_logits, confidences, weights

    def predict(self, features: np.ndarray, device: str = "cpu") -> List[Dict[str, Any]]:
        """Evaluates batch of candidate gap features and returns structured prediction dicts."""
        self.eval()
        with torch.no_grad():
            t_feat = torch.tensor(features, dtype=torch.float32, device=device)
            scores, type_logits, confs, _ = self.forward(t_feat)

            scores_np = scores.cpu().numpy().flatten()
            types_idx = torch.argmax(type_logits, dim=-1).cpu().numpy().flatten()
            types_prob = torch.softmax(type_logits, dim=-1).cpu().numpy()
            confs_np = confs.cpu().numpy().flatten()

            results = []
            for i in range(len(scores_np)):
                idx = int(types_idx[i])
                results.append({
                    "gap_score": float(np.round(scores_np[i], 4)),
                    "gap_type": GAP_TYPE_CLASSES[idx] if idx < len(GAP_TYPE_CLASSES) else "UNKNOWN",
                    "gap_type_probabilities": {GAP_TYPE_CLASSES[c]: float(np.round(types_prob[i][c], 4)) for c in range(self.num_classes)},
                    "confidence": float(np.round(confs_np[i], 4))
                })
            return results

    def save_checkpoint(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves PyTorch state dict and model configuration metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "gap_types": GAP_TYPE_CLASSES,
            "metadata": metadata or {}
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved gap detection model checkpoint to {save_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "ResearchGapDetectionNetwork":
        """Loads model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(
            num_features=checkpoint.get("num_features", 12),
            num_classes=checkpoint.get("num_classes", len(GAP_TYPE_CLASSES))
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Loaded gap detection model from {checkpoint_path}")
        return model
