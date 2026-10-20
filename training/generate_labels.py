"""
generate_labels.py
==================
Generates non-circular ground-truth labels and extracts 10 RAG-derived features.
Applies leakage-safe grouped train/validation/test splitting by paper entities.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.api.schemas import ResearchPaper, Patent
from app.retrieval.embeddings import EmbeddingGenerator
from app.features.relationship_features import (
    RELATIONSHIP_FEATURE_COLUMNS,
    compute_pairwise_rag_features,
    assign_independent_ground_truth_label
)
from app.logging_config import logger

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_TRAINING = PROJECT_ROOT / "data" / "training"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_TRAINING.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_labeled_feature_dataset() -> pd.DataFrame:
    """Loads candidates, computes 10 features, and assigns independent ground-truth labels."""
    logger.info("Loading clean papers and patents for feature computation...")
    papers_path = DATA_PROCESSED / "papers_clean.csv"
    patents_path = DATA_PROCESSED / "patents_clean.csv"

    df_papers = pd.read_csv(papers_path)
    df_patents = pd.read_csv(patents_path)

    encoder = EmbeddingGenerator()

    paper_texts = [f"{row['title']}. {str(row.get('abstract', ''))}" for _, row in df_papers.iterrows()]
    patent_texts = [f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}" for _, row in df_patents.iterrows()]

    paper_emb = encoder.encode(paper_texts)
    patent_emb = encoder.encode(patent_texts)

    tfidf_vec = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_vec.fit(paper_texts + patent_texts)

    paper_tfidf = tfidf_vec.transform(paper_texts)
    patent_tfidf = tfidf_vec.transform(patent_texts)
    tfidf_sim_matrix = cosine_similarity(paper_tfidf, patent_tfidf)

    paper_lookup = {row["paper_id"]: (idx, row) for idx, (_, row) in enumerate(df_papers.iterrows())}
    patent_lookup = {row["patent_id"]: (idx, row) for idx, (_, row) in enumerate(df_patents.iterrows())}

    # Load candidates
    candidates_path = DATA_PROCESSED / "candidates.csv"
    if candidates_path.exists():
        df_cand = pd.read_csv(candidates_path)
    else:
        # Fallback cartesian
        cand_list = []
        for p_id in df_papers["paper_id"]:
            for pt_id in df_patents["patent_id"]:
                cand_list.append({"paper_id": p_id, "patent_id": pt_id, "rrf_score": 0.05})
        df_cand = pd.DataFrame(cand_list)

    logger.info(f"Computing 10 RAG-derived features for {len(df_cand)} candidate pairs...")
    pair_rows = []

    for _, cand in df_cand.iterrows():
        p_id = cand["paper_id"]
        pt_id = cand["patent_id"]

        if p_id not in paper_lookup or pt_id not in patent_lookup:
            continue

        p_idx, p_row = paper_lookup[p_id]
        pt_idx, pt_row = patent_lookup[pt_id]

        p_obj = ResearchPaper(
            paper_id=p_id,
            title=p_row["title"],
            abstract=str(p_row.get("abstract", "")),
            publication_year=int(p_row.get("publication_year", 2023))
        )
        pt_obj = Patent(
            patent_id=pt_id,
            title=pt_row["title"],
            abstract=str(pt_row.get("abstract", "")),
            claims=[str(pt_row.get("claims", ""))],
            publication_date=str(pt_row.get("publication_date", "2023-01-01"))
        )

        tfidf_sim = float(tfidf_sim_matrix[p_idx, pt_idx])
        bm25_norm = float(cand.get("rrf_score", 0.05)) * 10.0  # proportional

        feats = compute_pairwise_rag_features(
            paper=p_obj,
            patent=pt_obj,
            paper_emb=paper_emb[p_idx],
            patent_emb=patent_emb[pt_idx],
            tfidf_sim=tfidf_sim,
            bm25_score=bm25_norm
        )

        label = assign_independent_ground_truth_label(
            paper=p_obj,
            patent=pt_obj,
            jaccard=feats["jaccard_similarity"],
            title_sim=feats["title_similarity"]
        )

        row_dict = {
            "paper_id": p_id,
            "patent_id": pt_id,
            "label": label,
            **feats
        }
        pair_rows.append(row_dict)

    df_features = pd.DataFrame(pair_rows)
    df_features.to_csv(DATA_TRAINING / "all_pairs_labeled.csv", index=False)
    logger.info(f"Dataset assembled: {len(df_features)} pairs (Pos: {(df_features['label']==1).sum()}, Neg: {(df_features['label']==0).sum()}).")
    return df_features


def perform_leakage_safe_split(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by unique paper entities (Group Split) to guarantee
    zero document leakage between training, validation, and test sets.
    """
    logger.info("Executing leakage-safe grouped split by unique paper IDs...")
    unique_papers = df_features["paper_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_papers)

    n_total = len(unique_papers)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_papers = set(unique_papers[:n_train])
    val_papers = set(unique_papers[n_train : n_train + n_val])
    test_papers = set(unique_papers[n_train + n_val :])

    train_df = df_features[df_features["paper_id"].isin(train_papers)].copy()
    val_df = df_features[df_features["paper_id"].isin(val_papers)].copy()
    test_df = df_features[df_features["paper_id"].isin(test_papers)].copy()

    # Fit scaler strictly on training split only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[RELATIONSHIP_FEATURE_COLUMNS].values)
    X_val_scaled = scaler.transform(val_df[RELATIONSHIP_FEATURE_COLUMNS].values)
    X_test_scaled = scaler.transform(test_df[RELATIONSHIP_FEATURE_COLUMNS].values)

    rel_model_dir = MODELS_DIR / "relationship_model"
    rel_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, rel_model_dir / "feature_scaler.pkl")
    logger.info(f"StandardScaler fitted strictly on training split and saved to {rel_model_dir / 'feature_scaler.pkl'}")

    # Save splits
    train_df.to_csv(DATA_TRAINING / "train.csv", index=False)
    val_df.to_csv(DATA_TRAINING / "validation.csv", index=False)
    test_df.to_csv(DATA_TRAINING / "test.csv", index=False)

    try:
        train_df.to_parquet(DATA_TRAINING / "train.parquet", index=False)
        val_df.to_parquet(DATA_TRAINING / "validation.parquet", index=False)
        test_df.to_parquet(DATA_TRAINING / "test.parquet", index=False)
    except Exception:
        pass

    logger.info(f"Split statistics:")
    logger.info(f"  Train: {len(train_df)} pairs ({len(train_papers)} papers, Pos: {(train_df['label']==1).sum()})")
    logger.info(f"  Val:   {len(val_df)} pairs ({len(val_papers)} papers, Pos: {(val_df['label']==1).sum()})")
    logger.info(f"  Test:  {len(test_df)} pairs ({len(test_papers)} papers, Pos: {(test_df['label']==1).sum()})")

    return train_df, val_df, test_df


if __name__ == "__main__":
    df_feat = build_labeled_feature_dataset()
    perform_leakage_safe_split(df_feat)
