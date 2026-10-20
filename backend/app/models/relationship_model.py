"""
relationship_model.py
======================
Siamese / Dual-Tower Neural Network with Multi-Head Cross-Attention and RAG Feature Fusion.

PRIMARY RESPONSIBILITY:
  Learns the semantic and technical relationship between a research paper and a patent document.
  Input:  Paper text representations + Patent text representations + RAG-derived features
  Output: Paper-Patent Relationship Probability P(Related | Paper, Patent) in [0, 1]
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from app.logging_config import logger
from app.models.paper_encoder import PaperTowerEncoder
from app.models.patent_encoder import PatentTowerEncoder
from app.models.cross_attention import BiDirectionalCrossAttention


class DualTowerCrossAttentionRelationshipModel(nn.Module):
    """
    Siamese / Dual-Tower Neural Architecture with Multi-Head Cross-Attention & RAG Fusion.
    
    Architecture:
      Paper Tower (384)       Patent Tower (384)
            │                         │
            ▼                         ▼
      Paper Representation      Patent Representation
            │                         │
            └───────────┬─────────────┘
                        │
                        ▼
            Bi-Directional Cross-Attention
                        │
                        ▼
          Cross-Attention Representation (384 * 4)
                        │
         + RAG Features (10 normalized metrics -> 64-d)
                        │
                        ▼
                  Feature Fusion Layer
                        │
                  Dense (256, ReLU) + Dropout(0.35)
                        │
                  Dense (64, ReLU) + Dropout(0.20)
                        │
                  Dense (1, Sigmoid) -> P(Related)
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_rag_features: int = 10,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_rag_features = num_rag_features

        # 1. Dual Towers
        self.paper_tower = PaperTowerEncoder(embed_dim=embed_dim, hidden_dim=embed_dim, dropout=dropout)
        self.patent_tower = PatentTowerEncoder(embed_dim=embed_dim, hidden_dim=embed_dim, dropout=dropout)

        # 2. Multi-Head Cross-Attention
        self.cross_attention = BiDirectionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 3. RAG Feature Projection
        self.rag_projection = nn.Sequential(
            nn.Linear(num_rag_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. Fusion Dimensions
        # [z_p (384), z_pt (384), diff (384), prod (384)] = 384 * 4 = 1536
        # cross_repr = 384 * 4 = 1536
        # rag_repr = 64
        # Total = 1536 + 1536 + 64 = 3136
        fusion_dim = (embed_dim * 4) + self.cross_attention.output_dim + 64

        # 5. Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        paper_emb: torch.Tensor,
        patent_emb: torch.Tensor,
        rag_features: torch.Tensor,
        paper_seq: Optional[torch.Tensor] = None,
        patent_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            paper_emb: (batch_size, embed_dim)
            patent_emb: (batch_size, embed_dim)
            rag_features: (batch_size, num_rag_features)
            paper_seq: Optional (batch_size, seq_len_p, embed_dim)
            patent_seq: Optional (batch_size, seq_len_pt, embed_dim)
            
        Returns:
            probability: (batch_size, 1) in [0, 1]
            attention_weights: Tuple of (attn_p2pt, attn_pt2p) or None
        """
        batch_size = paper_emb.shape[0]

        # 1. Project through dedicated towers
        z_p = self.paper_tower(paper_emb)      # (B, D)
        z_pt = self.patent_tower(patent_emb)   # (B, D)

        diff_emb = torch.abs(z_p - z_pt)
        prod_emb = z_p * z_pt
        direct_fusion = torch.cat([z_p, z_pt, diff_emb, prod_emb], dim=-1)  # (B, 4*D)

        # 2. Cross-Attention
        # If sequence is not provided, expand pooled embedding to pseudo-sequence (B, 1, D)
        if paper_seq is None:
            paper_seq = z_p.unsqueeze(1)
        else:
            paper_seq = self.paper_tower(paper_seq)

        if patent_seq is None:
            patent_seq = z_pt.unsqueeze(1)
        else:
            patent_seq = self.patent_tower(patent_seq)

        cross_repr, attn_weights = self.cross_attention(paper_seq, patent_seq)

        # 3. Project RAG features
        rag_repr = self.rag_projection(rag_features)

        # 4. Concatenate all feature streams
        fused = torch.cat([direct_fusion, cross_repr, rag_repr], dim=-1)

        # 5. Output relationship probability
        prob = self.classifier(fused)
        return prob, attn_weights

    def predict_proba(
        self,
        paper_emb: np.ndarray,
        patent_emb: np.ndarray,
        rag_features: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        """Convenience method for evaluation and production inference."""
        self.eval()
        with torch.no_grad():
            t_p = torch.tensor(paper_emb, dtype=torch.float32, device=device)
            t_pt = torch.tensor(patent_emb, dtype=torch.float32, device=device)
            t_rag = torch.tensor(rag_features, dtype=torch.float32, device=device)

            probs, _ = self.forward(t_p, t_pt, t_rag)
            return probs.cpu().numpy().flatten()

    def save_checkpoint(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves PyTorch state dict and model configuration metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "num_rag_features": self.num_rag_features,
            "metadata": metadata or {}
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved relationship model checkpoint to {save_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "DualTowerCrossAttentionRelationshipModel":
        """Loads model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(
            embed_dim=checkpoint.get("embed_dim", 384),
            num_rag_features=checkpoint.get("num_rag_features", 10)
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Loaded relationship model from {checkpoint_path}")
        return model
