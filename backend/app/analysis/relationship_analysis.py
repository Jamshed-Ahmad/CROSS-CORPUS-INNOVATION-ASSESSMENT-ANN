"""
relationship_analysis.py
========================
Relationship Analysis Service using the Siamese / Dual-Tower + Cross-Attention Model.

RESPONSIBILITY:
  Evaluates semantic and technical relationships between retrieved scientific papers
  and patent records using the trained DualTowerCrossAttentionRelationshipModel.
"""

import os
import uuid
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from app.api.schemas import ResearchPaper, Patent, CrossCorpusRelationship, EvidenceRecord
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.tfidf import TFIDFRetriever
from app.features.relationship_features import compute_pairwise_rag_features, RELATIONSHIP_FEATURE_COLUMNS
from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
from app.logging_config import logger


class RelationshipAnalyzer:
    """Orchestrates relationship prediction between papers and patents using the Dual-Tower model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        tfidf_retriever: Optional[TFIDFRetriever] = None,
        device: str = "cpu"
    ):
        self.encoder = embedding_generator or EmbeddingGenerator()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.tfidf = tfidf_retriever or TFIDFRetriever()
        self.device = device
        self.model: Optional[DualTowerCrossAttentionRelationshipModel] = None
        self.model_path = model_path

        self._load_model()

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = DualTowerCrossAttentionRelationshipModel.load_checkpoint(self.model_path, device=self.device)
                logger.info(f"RelationshipAnalyzer: Loaded neural relationship model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"RelationshipAnalyzer: Could not load checkpoint ({e}). Running in deterministic fallback mode.")

        # Default fallback initialization
        self.model = DualTowerCrossAttentionRelationshipModel(embed_dim=self.encoder.dimension, num_rag_features=10)
        self.model.eval()

    def analyze_relationships(
        self,
        papers: List[ResearchPaper],
        patents: List[Patent],
        similarity_threshold: float = 0.20
    ) -> Tuple[List[CrossCorpusRelationship], List[EvidenceRecord]]:
        """
        Computes pairwise relationships across papers and patents.
        Uses Dual-Tower + Cross-Attention model for paper-to-patent relationship prediction.
        """
        relationships: List[CrossCorpusRelationship] = []
        evidence_list: List[EvidenceRecord] = []

        paper_texts = [f"{p.title}. {p.abstract or ''}" for p in papers]
        patent_texts = [f"{pt.title}. {pt.abstract or ''} {' '.join(pt.claims or [])}" for pt in patents]

        paper_vecs = self.encoder.encode(paper_texts) if paper_texts else np.zeros((0, self.encoder.dimension))
        patent_vecs = self.encoder.encode(patent_texts) if patent_texts else np.zeros((0, self.encoder.dimension))

        # 1. Paper-to-Paper Relationships
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                p1, p2 = papers[i], papers[j]
                sim = float(np.dot(paper_vecs[i], paper_vecs[j]))
                is_cited = (p2.paper_id in (p1.references or [])) or (p1.paper_id in (p2.references or []))

                if sim >= similarity_threshold or is_cited:
                    ev_id = f"EV-PP-{uuid.uuid4().hex[:6]}"
                    ev = EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=p1.paper_id,
                        document_type="paper",
                        passage_text=f"Conceptual overlap between '{p1.title}' and '{p2.title}' (Cosine: {sim:.3f})",
                        source_url=p1.source_url,
                        evidence_type="method",
                        extraction_method="cross_corpus_embedding_similarity",
                        confidence=round(min(1.0, max(0.5, sim)), 2)
                    )
                    evidence_list.append(ev)

                    rel = CrossCorpusRelationship(
                        relationship_id=f"REL-PP-{uuid.uuid4().hex[:6]}",
                        source_entity_id=p1.paper_id,
                        target_entity_id=p2.paper_id,
                        relationship_type="paper-to-paper",
                        score=round(sim, 3),
                        supporting_evidence_ids=[ev_id],
                        extraction_method="citation_and_semantic_similarity",
                        confidence=round(sim, 2)
                    )
                    relationships.append(rel)

        # 2. Patent-to-Patent Relationships
        for i in range(len(patents)):
            for j in range(i + 1, len(patents)):
                pt1, pt2 = patents[i], patents[j]
                sim = float(np.dot(patent_vecs[i], patent_vecs[j]))
                shared_cpc = set(pt1.classifications or []).intersection(set(pt2.classifications or []))

                if sim >= similarity_threshold or shared_cpc:
                    ev_id = f"EV-TT-{uuid.uuid4().hex[:6]}"
                    ev = EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=pt1.patent_id,
                        document_type="patent",
                        passage_text=f"Patent overlap in classifications {list(shared_cpc)} (Cosine: {sim:.3f})",
                        source_url=pt1.source_url,
                        evidence_type="solution",
                        extraction_method="classification_and_claims_similarity",
                        confidence=round(min(1.0, max(0.5, sim)), 2)
                    )
                    evidence_list.append(ev)

                    rel = CrossCorpusRelationship(
                        relationship_id=f"REL-TT-{uuid.uuid4().hex[:6]}",
                        source_entity_id=pt1.patent_id,
                        target_entity_id=pt2.patent_id,
                        relationship_type="patent-to-patent",
                        score=round(sim, 3),
                        supporting_evidence_ids=[ev_id],
                        extraction_method="cpc_classification_and_claims_matching",
                        confidence=round(sim, 2)
                    )
                    relationships.append(rel)

        # 3. Paper-to-Patent Relationships (using Dual-Tower + Cross-Attention Network)
        candidate_features = []
        candidate_pairs = []

        for i in range(len(papers)):
            for j in range(len(patents)):
                p = papers[i]
                pt = patents[j]
                tfidf_sim = self.tfidf.compute_pair_similarity(paper_texts[i], patent_texts[j])
                bm25_val = 0.5  # normalized baseline for pair
                
                feat_dict = compute_pairwise_rag_features(
                    paper=p,
                    patent=pt,
                    paper_emb=paper_vecs[i],
                    patent_emb=patent_vecs[j],
                    tfidf_sim=tfidf_sim,
                    bm25_score=bm25_val
                )
                feat_vec = [feat_dict[col] for col in RELATIONSHIP_FEATURE_COLUMNS]
                candidate_features.append(feat_vec)
                candidate_pairs.append((i, j, feat_dict))

        if candidate_pairs and self.model is not None:
            feat_arr = np.array(candidate_features, dtype=np.float32)
            p_embs = np.array([paper_vecs[p_idx] for p_idx, _, _ in candidate_pairs], dtype=np.float32)
            pt_embs = np.array([patent_vecs[pt_idx] for _, pt_idx, _ in candidate_pairs], dtype=np.float32)

            predicted_probs = self.model.predict_proba(p_embs, pt_embs, feat_arr, device=self.device)
        else:
            predicted_probs = np.full(len(candidate_pairs), 0.5, dtype=np.float32)

        for idx, (p_idx, pt_idx, feat_dict) in enumerate(candidate_pairs):
            p = papers[p_idx]
            pt = patents[pt_idx]
            rel_prob = float(predicted_probs[idx])
            sim = feat_dict["semantic_similarity"]
            is_cited = (p.paper_id in (pt.citations or [])) or (pt.patent_id in (p.references or []))

            if rel_prob >= similarity_threshold or sim >= similarity_threshold or is_cited:
                ev_id = f"EV-PT-{uuid.uuid4().hex[:6]}"
                ev = EvidenceRecord(
                    evidence_id=ev_id,
                    document_id=pt.patent_id,
                    document_type="patent",
                    passage_text=f"Cross-corpus linkage between '{p.title}' and patent '{pt.title}' (Predicted Rel: {rel_prob:.3f}, Cosine: {sim:.3f})",
                    source_url=pt.source_url,
                    evidence_type="solution",
                    extraction_method="dual_tower_cross_attention_prediction",
                    confidence=round(min(1.0, max(0.5, rel_prob)), 2)
                )
                evidence_list.append(ev)

                rel = CrossCorpusRelationship(
                    relationship_id=f"REL-PT-{uuid.uuid4().hex[:6]}",
                    source_entity_id=p.paper_id,
                    target_entity_id=pt.patent_id,
                    relationship_type="paper-to-patent",
                    score=round(rel_prob, 3),
                    supporting_evidence_ids=[ev_id],
                    extraction_method="siamese_dual_tower_cross_attention",
                    confidence=round(rel_prob, 2),
                    ann_relevance_score=round(rel_prob, 4)
                )
                relationships.append(rel)

        logger.info(f"RelationshipAnalyzer extracted {len(relationships)} relationships across {len(papers)} papers and {len(patents)} patents.")
        return relationships, evidence_list
