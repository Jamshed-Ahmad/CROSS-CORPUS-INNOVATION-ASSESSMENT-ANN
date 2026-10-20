"""
paper_encoder.py
================
Paper Tower Encoder for the Siamese / Dual-Tower Architecture.
Extracts both sequence-level concept representations and pooled document embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
import numpy as np


class PaperTowerEncoder(nn.Module):
    """
    Dedicated Neural Tower for scientific paper representations.
    Projects dense pretrained embeddings into the relationship learning manifold.
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Learnable adaptation and normalization projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
        Returns:
            Projected representation of identical rank.
        """
        return self.projection(x)

    def prepare_sequence_from_chunks(
        self,
        chunk_embeddings: List[np.ndarray],
        max_seq_len: int = 6
    ) -> torch.Tensor:
        """
        Converts variable-length chunk embeddings into a padded (batch_size, max_seq_len, embed_dim) tensor.
        """
        batch_size = len(chunk_embeddings)
        out = torch.zeros(batch_size, max_seq_len, self.embed_dim, dtype=torch.float32)
        for i, chunks in enumerate(chunk_embeddings):
            if len(chunks) == 0:
                continue
            usable = chunks[:max_seq_len]
            out[i, :len(usable), :] = torch.tensor(usable, dtype=torch.float32)
        return out
