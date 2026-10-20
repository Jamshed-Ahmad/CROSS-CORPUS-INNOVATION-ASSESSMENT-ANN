from typing import List, Tuple, Dict
from app.logging_config import logger

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float, str]]],
    k: int = 60,
    top_n: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Combine multiple rank-ordered lists of (doc_id, score, doc_type) using Reciprocal Rank Fusion.
    RRF_score(d) = sum_{m} 1 / (k + r_m(d))
    """
    rrf_scores: Dict[str, float] = {}
    doc_types: Dict[str, str] = {}

    for rank_list in rankings:
        for rank, item in enumerate(rank_list, start=1):
            if len(item) >= 3:
                doc_id, _, doc_type = item[0], item[1], item[2]
            else:
                doc_id, _ = item[0], item[1]
                doc_type = "document"
            doc_types[doc_id] = doc_type
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = [(doc_id, score, doc_types[doc_id]) for doc_id, score in sorted_docs[:top_n]]
    logger.info(f"Hybrid RRF fused {len(rankings)} retrieval streams into {len(results)} top results.")
    return results
