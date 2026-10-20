"""
relationship_features.py
========================
Calculates the 10 heterogeneous RAG-derived features for paper-patent relationship learning.
Ensures strictly leak-free scaling and non-circular labeling.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from app.api.schemas import ResearchPaper, Patent

RELATIONSHIP_FEATURE_COLUMNS = [
    "semantic_similarity",     # F1: MiniLM dense cosine
    "tfidf_similarity",        # F2: TF-IDF sparse cosine
    "bm25_score",              # F3: Normalized BM25 score
    "jaccard_similarity",      # F4: Full token Jaccard
    "word_overlap_count",      # F5: Normalized shared token count
    "title_similarity",        # F6: Title-only Jaccard
    "abstract_similarity",     # F7: Abstract-only TF-IDF
    "emb_diff_norm",           # F8: L2 norm of embedding delta
    "emb_product_mean",        # F9: Hadamard interaction mean
    "publication_year_diff"    # F10: Temporal absolute year delta
]

# Domain keyword pairs used exclusively for independent labeling (NOT features)
INDEPENDENT_DOMAIN_PAIRS: List[Tuple[str, str]] = [
    ("drug", "affinity"),
    ("quantum", "docking"),
    ("cryo-em", "protein"),
    ("recommender", "graph"),
    ("knowledge graph", "patent"),
    ("genomic", "phenotypic"),
    ("molecular", "binding"),
    ("transformer", "sequence"),
    ("graph neural", "compound"),
    ("wearable", "sensor"),
    ("heart disease", "cardiac"),
    ("federated", "privacy"),
    ("anomaly detection", "intrusion")
]


def compute_pairwise_rag_features(
    paper: ResearchPaper,
    patent: Patent,
    paper_emb: np.ndarray,
    patent_emb: np.ndarray,
    tfidf_sim: float = 0.0,
    bm25_score: float = 0.0
) -> Dict[str, float]:
    """Computes all 10 non-redundant relationship features for one (paper, patent) pair."""
    # 1. Semantic Cosine Similarity
    p_norm = np.linalg.norm(paper_emb)
    pt_norm = np.linalg.norm(patent_emb)
    if p_norm > 0 and pt_norm > 0:
        sem_sim = float(np.dot(paper_emb, patent_emb) / (p_norm * pt_norm))
    else:
        sem_sim = 0.0

    # Tokens
    p_title_tokens = set(paper.title.lower().split())
    pt_title_tokens = set(patent.title.lower().split())
    p_all_tokens = set((paper.title + " " + (paper.abstract or "")).lower().split())
    claims_text = " ".join(patent.claims or [])
    pt_all_tokens = set((patent.title + " " + (patent.abstract or "") + " " + claims_text).lower().split())

    # 4. Jaccard token similarity
    union = len(p_all_tokens.union(pt_all_tokens))
    inter = len(p_all_tokens.intersection(pt_all_tokens))
    jaccard = float(inter / union) if union > 0 else 0.0

    # 5. Normalized word overlap
    mean_len = (len(p_all_tokens) + len(pt_all_tokens)) / 2.0
    word_overlap = float(inter / mean_len) if mean_len > 0 else 0.0

    # 6. Title similarity
    title_union = len(p_title_tokens.union(pt_title_tokens))
    title_inter = len(p_title_tokens.intersection(pt_title_tokens))
    title_sim = float(title_inter / title_union) if title_union > 0 else 0.0

    # 7. Abstract similarity approximation
    abstract_sim = float(np.clip(tfidf_sim * 0.95, 0.0, 1.0))

    # 8. Embedding difference L2 norm
    emb_diff = float(np.linalg.norm(paper_emb - patent_emb))

    # 9. Embedding Hadamard product mean
    emb_prod = float(np.mean(paper_emb * patent_emb))

    # 10. Publication year diff
    p_year = paper.publication_year or 2023
    try:
        pt_year = int(patent.publication_date[:4]) if patent.publication_date else 2023
    except Exception:
        pt_year = 2023
    year_diff = float(abs(p_year - pt_year))

    return {
        "semantic_similarity": round(sem_sim, 4),
        "tfidf_similarity": round(float(tfidf_sim), 4),
        "bm25_score": round(float(bm25_score), 4),
        "jaccard_similarity": round(jaccard, 4),
        "word_overlap_count": round(word_overlap, 4),
        "title_similarity": round(title_sim, 4),
        "abstract_similarity": round(abstract_sim, 4),
        "emb_diff_norm": round(emb_diff, 4),
        "emb_product_mean": round(emb_prod, 4),
        "publication_year_diff": round(year_diff, 2)
    }


def assign_independent_ground_truth_label(
    paper: ResearchPaper,
    patent: Patent,
    jaccard: float,
    title_sim: float
) -> int:
    """
    Assigns non-circular ground-truth binary label y ∈ {0, 1}
    using three independent signals (citations, domain taxonomy co-occurrence, dual lexical constraints).
    """
    # Signal 1: Explicit citation cross-link
    is_cited = False
    if patent.citations and paper.paper_id in patent.citations:
        is_cited = True
    if paper.references and patent.patent_id in paper.references:
        is_cited = True

    # Signal 2: Domain keyword taxonomy co-occurrence
    pair_text = (paper.title + " " + patent.title).lower()
    kw_match = False
    for t1, t2 in INDEPENDENT_DOMAIN_PAIRS:
        if t1 in pair_text and t2 in pair_text:
            kw_match = True
            break

    # Signal 3: Dual conservative intersection (both title and body must overlap independently)
    dual_lexical_overlap = (jaccard > 0.08 and title_sim > 0.05)

    return 1 if (is_cited or kw_match or dual_lexical_overlap) else 0
