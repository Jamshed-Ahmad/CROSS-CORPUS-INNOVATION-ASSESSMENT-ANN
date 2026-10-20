"""
build_pairs.py
==============
Converts raw API records into clean processed datasets and generates candidate pairs
using hybrid RAG retrieval (BM25 + TF-IDF + FAISS + RRF) rather than an exhaustive Cartesian product.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.api.schemas import ResearchPaper, Patent
from app.ingestion.preprocessing import preprocess_papers, preprocess_patents
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.tfidf import TFIDFRetriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vector_index import VectorIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.logging_config import logger

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_and_normalize_data(raw_papers_path: Path, raw_patents_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans, validates, and deduplicates raw paper and patent records."""
    logger.info("Cleaning and normalizing raw datasets...")
    with open(raw_papers_path, "r", encoding="utf-8") as f:
        raw_papers = [ResearchPaper(**item) for item in json.load(f)]

    with open(raw_patents_path, "r", encoding="utf-8") as f:
        patent_items = json.load(f)
        raw_patents = []
        for item in patent_items:
            if isinstance(item.get("claims"), str):
                item["claims"] = [item["claims"]]
            if isinstance(item.get("classifications"), str):
                item["classifications"] = [c.strip() for c in item["classifications"].split(",") if c.strip()]
            raw_patents.append(Patent(**item))

    clean_papers, paper_profile = preprocess_papers(raw_papers)
    clean_patents, patent_profile = preprocess_patents(raw_patents)

    logger.info(f"Papers cleaned: {len(clean_papers)} | Profile: {paper_profile}")
    logger.info(f"Patents cleaned: {len(clean_patents)} | Profile: {patent_profile}")

    df_papers = pd.DataFrame([p.model_dump() for p in clean_papers])
    df_patents = pd.DataFrame([pt.model_dump() for pt in clean_patents])

    # Save processed datasets
    try:
        df_papers.to_parquet(DATA_PROCESSED / "papers.parquet", index=False)
        df_patents.to_parquet(DATA_PROCESSED / "patents.parquet", index=False)
    except Exception:
        pass
    df_papers.to_csv(DATA_PROCESSED / "papers_clean.csv", index=False)
    df_patents.to_csv(DATA_PROCESSED / "patents_clean.csv", index=False)

    return df_papers, df_patents, clean_papers, clean_patents


def generate_candidate_pairs(
    clean_papers: List[ResearchPaper],
    clean_patents: List[Patent],
    top_k_per_paper: int = 10
) -> pd.DataFrame:
    """
    Generates candidate pairs using hybrid RAG retrieval.
    Avoids O(N x M) explosion by selecting the top-K candidate patents per paper.
    """
    logger.info("Building hybrid retrieval indices for candidate pair generation...")
    encoder = EmbeddingGenerator()
    bm25 = BM25Retriever()
    tfidf = TFIDFRetriever()
    vector_index = VectorIndex(encoder)

    bm25.index_documents(clean_papers, clean_patents)
    tfidf.index_documents(clean_papers, clean_patents)
    vector_index.index_documents(clean_papers, clean_patents)

    patent_dict = {pt.patent_id: pt for pt in clean_patents}
    candidate_rows = []

    logger.info(f"Querying hybrid RAG for each of the {len(clean_papers)} papers...")
    for paper in clean_papers:
        paper_query = f"{paper.title}. {paper.abstract or ''}"

        bm25_hits = [h for h in bm25.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]
        tfidf_hits = [h for h in tfidf.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]
        vec_hits = [h for h in vector_index.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]

        fused = reciprocal_rank_fusion([bm25_hits, tfidf_hits, vec_hits], top_n=top_k_per_paper)

        for pt_id, rrf_score, _ in fused:
            if pt_id in patent_dict:
                pt = patent_dict[pt_id]
                candidate_rows.append({
                    "paper_id": paper.paper_id,
                    "patent_id": pt.patent_id,
                    "paper_title": paper.title,
                    "patent_title": pt.title,
                    "paper_abstract": paper.abstract or "",
                    "patent_abstract": pt.abstract or "",
                    "patent_claims": " ".join(pt.claims or []),
                    "rrf_score": round(float(rrf_score), 5),
                    "paper_year": paper.publication_year or 2023,
                    "patent_year": int(pt.publication_date[:4]) if pt.publication_date else 2023
                })

    df_candidates = pd.DataFrame(candidate_rows).drop_duplicates(subset=["paper_id", "patent_id"])
    logger.info(f"Generated {len(df_candidates)} candidate pairs using hybrid RAG retrieval.")

    try:
        df_candidates.to_parquet(DATA_PROCESSED / "candidates.parquet", index=False)
    except Exception:
        pass
    df_candidates.to_csv(DATA_PROCESSED / "candidates.csv", index=False)
    return df_candidates


if __name__ == "__main__":
    raw_p = DATA_RAW / "papers.json"
    raw_pt = DATA_RAW / "patents.json"
    if not raw_p.exists() or not raw_pt.exists():
        from training.collect_data import run_collection
        run_collection()

    df_papers, df_patents, clean_p, clean_pt = clean_and_normalize_data(raw_p, raw_pt)
    generate_candidate_pairs(clean_p, clean_pt, top_k_per_paper=10)
