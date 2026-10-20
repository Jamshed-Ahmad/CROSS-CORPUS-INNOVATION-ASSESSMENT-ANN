"""
cross_attention.py
==================
Bi-directional Multi-Head Cross-Attention mechanism for Paper-Patent relationship learning.

Conceptually:
    Paper tokens / concept chunks  <--->  Multi-Head Cross-Attention  <--->  Patent tokens / claims
Learns fine-grained technical and semantic interactions between scientific hypotheses and patent claims.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BiDirectionalCrossAttention(nn.Module):
    """
    Bi-directional Multi-Head Cross-Attention module.
    
    Computes:
      1. Paper-to-Patent Attention: How each paper token/chunk attends to all patent tokens/claims.
      2. Patent-to-Paper Attention: How each patent token/claim attends to all paper tokens/chunks.
      3. Residual connection + Layer Normalization.
      4. Interaction pooling: [z_paper, z_patent, |z_paper - z_patent|, z_paper * z_patent].
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Paper attends to Patent
        self.mha_paper_to_patent = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_paper = nn.LayerNorm(embed_dim)
        
        # Patent attends to Paper
        self.mha_patent_to_paper = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_patent = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = embed_dim * 4  # [z_p, z_pt, diff, prod]

    def forward(
        self,
        paper_seq: torch.Tensor,
        patent_seq: torch.Tensor,
        paper_mask: Optional[torch.Tensor] = None,
        patent_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            paper_seq: Tensor of shape (batch_size, seq_len_p, embed_dim)
            patent_seq: Tensor of shape (batch_size, seq_len_pt, embed_dim)
            paper_mask: Optional key padding mask (batch_size, seq_len_p)
            patent_mask: Optional key padding mask (batch_size, seq_len_pt)
            
        Returns:
            fused_cross_repr: Tensor of shape (batch_size, output_dim)
            (attn_p2pt, attn_pt2p): Raw attention weight matrices for explainability
        """
        # 1. Paper attends to Patent (Q=Paper, K=Patent, V=Patent)
        attended_paper, weights_p2pt = self.mha_paper_to_patent(
            query=paper_seq,
            key=patent_seq,
            value=patent_seq,
            key_padding_mask=patent_mask
        )
        norm_paper = self.ln_paper(paper_seq + self.dropout(attended_paper))
        
        # 2. Patent attends to Paper (Q=Patent, K=Paper, V=Paper)
        attended_patent, weights_pt2p = self.mha_patent_to_paper(
            query=patent_seq,
            key=paper_seq,
            value=paper_seq,
            key_padding_mask=paper_mask
        )
        norm_patent = self.ln_patent(patent_seq + self.dropout(attended_patent))
        
        # 3. Mean Pooling across sequence dimension
        z_paper = norm_paper.mean(dim=1)    # (batch_size, embed_dim)
        z_patent = norm_patent.mean(dim=1)  # (batch_size, embed_dim)
        
        # 4. Construct rich relational interaction vector
        diff = torch.abs(z_paper - z_patent)
        prod = z_paper * z_patent
        fused_cross_repr = torch.cat([z_paper, z_patent, diff, prod], dim=-1)
        
        return fused_cross_repr, (weights_p2pt, weights_pt2p)
