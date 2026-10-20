"""
__init__.py
===========
Feature engineering package for relationship learning and research gap detection.
"""

from app.features.relationship_features import (
    RELATIONSHIP_FEATURE_COLUMNS,
    compute_pairwise_rag_features,
    assign_independent_ground_truth_label
)
from app.features.gap_features import (
    GAP_FEATURE_COLUMNS,
    extract_gap_candidate_features
)

__all__ = [
    "RELATIONSHIP_FEATURE_COLUMNS",
    "compute_pairwise_rag_features",
    "assign_independent_ground_truth_label",
    "GAP_FEATURE_COLUMNS",
    "extract_gap_candidate_features"
]
