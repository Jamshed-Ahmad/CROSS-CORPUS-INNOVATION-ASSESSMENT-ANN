"""
cross_corpus.py
===============
Cross-Corpus relationship analyzer.
Delegates to the Siamese / Dual-Tower + Cross-Attention relationship model.
"""

from typing import List, Tuple, Optional
from app.api.schemas import ResearchPaper, Patent, CrossCorpusRelationship, EvidenceRecord
from app.retrieval.embeddings import EmbeddingGenerator
from app.analysis.relationship_analysis import RelationshipAnalyzer


class CrossCorpusAnalyzer(RelationshipAnalyzer):
    """
    Analyzes cross-corpus paper-paper, patent-patent, and paper-patent relationships
    using the Dual-Tower Cross-Attention neural model.
    """
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None, model_path: Optional[str] = None):
        super().__init__(
            model_path=model_path,
            embedding_generator=embedding_generator
        )
