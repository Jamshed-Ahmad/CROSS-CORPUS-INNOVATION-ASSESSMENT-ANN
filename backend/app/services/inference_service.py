"""
inference_service.py
====================
Unified Inference Service for the Evidence-Grounded Cross-Corpus Innovation Assessment System.
Loads trained PyTorch checkpoints for Dual-Tower Cross-Attention and Research Gap Detection.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from app.config import settings
from app.logging_config import logger
from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
from app.models.gap_detection_model import ResearchGapDetectionNetwork


class InferenceService:
    """Production inference service orchestrating loaded PyTorch models."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, models_dir: Optional[str] = None, device: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self.device = device
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent.parent / "models"
        self.relationship_model: Optional[DualTowerCrossAttentionRelationshipModel] = None
        self.gap_model: Optional[ResearchGapDetectionNetwork] = None

        self._load_models()
        self._initialized = True

    def _load_models(self):
        # 1. Load Relationship Model
        rel_path = self.models_dir / "relationship_model" / "relationship_model.pt"
        if rel_path.exists():
            try:
                self.relationship_model = DualTowerCrossAttentionRelationshipModel.load_checkpoint(str(rel_path), device=self.device)
                logger.info(f"InferenceService: Successfully loaded relationship model from {rel_path}")
            except Exception as e:
                logger.warning(f"InferenceService: Failed to load relationship model checkpoint ({e}). Initializing fresh model.")
                self.relationship_model = DualTowerCrossAttentionRelationshipModel().to(self.device)
        else:
            logger.info("InferenceService: No relationship checkpoint found. Using initialized model.")
            self.relationship_model = DualTowerCrossAttentionRelationshipModel().to(self.device)

        # 2. Load Gap Detection Model
        gap_path = self.models_dir / "gap_detection_model" / "gap_detection_model.pt"
        if gap_path.exists():
            try:
                self.gap_model = ResearchGapDetectionNetwork.load_checkpoint(str(gap_path), device=self.device)
                logger.info(f"InferenceService: Successfully loaded gap detection model from {gap_path}")
            except Exception as e:
                logger.warning(f"InferenceService: Failed to load gap model checkpoint ({e}). Initializing fresh model.")
                self.gap_model = ResearchGapDetectionNetwork().to(self.device)
        else:
            logger.info("InferenceService: No gap checkpoint found. Using initialized model.")
            self.gap_model = ResearchGapDetectionNetwork().to(self.device)


def get_inference_service() -> InferenceService:
    """Returns singleton inference service instance."""
    return InferenceService()


def analyze_research_idea(query: str, domain: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Complete end-to-end inference function for evaluating a research proposal.
    Executes: Query Decomposition -> Hybrid RAG -> CrossEncoder Reranking -> 
    Dual-Tower Cross-Attention -> Limitation Extraction -> Gap Detection Network -> 
    Innovation Scoring -> Evidence Validation.
    """
    from app.services.analysis_pipeline import AnalysisPipelineService
    from app.api.schemas import AnalysisRequest

    pipeline = AnalysisPipelineService()
    request = AnalysisRequest(idea_text=query, domain=domain, top_k=top_k)
    response = pipeline.run_analysis(request)

    return {
        "analysis_id": response.analysis_id,
        "status": response.status,
        "research_idea": response.parsed_idea.model_dump(),
        "recommended_papers": [p.model_dump() for p in response.papers],
        "recommended_patents": [pt.model_dump() for pt in response.patents],
        "paper_patent_relationships": [r.model_dump() for r in response.relationships],
        "research_gaps": [g.model_dump() for g in response.gaps],
        "recommendations": [rec.model_dump() for rec in response.recommendations],
        "evidence": [e.model_dump() for e in response.evidence],
        "scores": response.scores,
        "warnings": response.warnings
    }

