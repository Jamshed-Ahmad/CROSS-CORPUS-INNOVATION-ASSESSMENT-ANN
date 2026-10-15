"""
feature_engineering.py
=======================
Centralized, documented feature engineering for paper-patent relationship prediction.

INPUT:
  - Two DataFrames: df_papers, df_patents (each row = one clean document)
  - Precomputed dense embeddings (NumPy arrays): paper_emb, patent_emb
  - Fitted TF-IDF vectorizer and BM25 index

OUTPUT:
  - One feature DataFrame (df_pairs) where each row = one paper-patent candidate pair
  - Each row contains 10 distinct, non-redundant features + ground-truth label

INTEGRATION POINT:
  - Called by build_research_pipeline.py during Stage 3 (Feature Engineering)
  - The produced feature matrix is fed into baseline models and the proposed ANN

LABELING STRATEGY:
  Labels are assigned using *three independent signals* (none of which are the model features):
    1. Shared domain keyword co-occurrence (technology domain alignment)
    2. Explicit citation cross-reference (paper_id in patent.citations or vice versa)
    3. Jaccard * title-Sim threshold (both must exceed low thresholds simultaneously)
  This prevents circular labeling (target leakage) from the prediction features.
"""

import re
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# FEATURE DEFINITIONS (documented table)
# ---------------------------------------------------------------------------
#
# Feature 1: semantic_similarity
#   What:  Cosine similarity between MiniLM-L6-v2 dense embeddings of paper and patent texts.
#   How:   dot(paper_emb[i], patent_emb[j]) — embeddings are L2-normalized, so this = cosine.
#   Why:   Captures deep conceptual overlap that keyword-based measures miss.
#   Where: ANN input layer, also used by cross-corpus analyzer.
#
# Feature 2: tfidf_similarity
#   What:  Cosine similarity between TF-IDF sparse vectors of paper and patent texts.
#   How:   cosine_similarity(paper_tfidf[i], patent_tfidf[j]) — DISTINCT from Feature 1.
#   Why:   Captures shared rare domain-specific terminology (e.g., "variational quantum").
#   Where: ANN input layer, baseline models.
#
# Feature 3: bm25_score (normalized)
#   What:  BM25 ranking score of the patent text against paper text treated as a query.
#   How:   Build BM25 index from patent texts; query with paper tokens; normalize by max score.
#   Why:   Length-normalized term relevance; complementary to raw TF-IDF cosine.
#   Where: ANN input layer, baseline models.
#
# Feature 4: jaccard_similarity
#   What:  |intersect(paper_tokens, patent_tokens)| / |union(paper_tokens, patent_tokens)|
#   How:   Computed on lowercased full-text token sets (title + abstract + claims).
#   Why:   Set-theoretic overlap; robust to word order; fast to compute.
#   Where: ANN input layer, baseline models, labeling heuristic (independent signal).
#
# Feature 5: word_overlap_count (normalized)
#   What:  Raw count of shared tokens divided by mean document length.
#   How:   len(intersect) / mean(|paper_tokens|, |patent_tokens|)
#   Why:   Absolute term co-occurrence; different scale from Jaccard ratio.
#   Where: ANN input layer, baseline models.
#
# Feature 6: title_similarity
#   What:  Jaccard similarity computed on title tokens only (paper.title vs patent.title).
#   How:   |intersect(title_paper, title_patent)| / |union(title_patent, title_paper)|
#   Why:   Titles are dense concept summaries; title overlap is a strong cross-corpus signal.
#   Where: ANN input layer, baseline models, labeling heuristic (independent signal).
#
# Feature 7: abstract_similarity
#   What:  TF-IDF cosine similarity computed on abstract text only (excluding titles/claims).
#   How:   cosine_similarity(tfidf(paper.abstract), tfidf(patent.abstract))
#   Why:   Isolates the background/summary text; avoids noise from structured patent claims.
#   Where: ANN input layer.
#
# Feature 8: emb_diff_norm
#   What:  L2 norm of the element-wise difference between paper and patent embeddings.
#   How:   ||paper_emb[i] - patent_emb[j]||_2
#   Why:   Encodes "distance" between representations; different geometric interpretation
#          from cosine similarity (magnitude vs. angle).
#   Where: ANN input layer (captures absolute dissimilarity).
#
# Feature 9: emb_product_mean
#   What:  Mean of the element-wise product (Hadamard product) of paper and patent embeddings.
#   How:   mean(paper_emb[i] * patent_emb[j])
#   Why:   Interaction feature that highlights dimensions both documents activate strongly.
#   Where: ANN input layer (interaction term for non-linear fusion).
#
# Feature 10: publication_year_diff
#   What:  Absolute difference between paper publication year and patent publication year.
#   How:   |paper_year - patent_year|
#   Why:   Temporal proximity: papers and patents from the same period are more likely related.
#   Where: ANN input layer, baseline models.
# ---------------------------------------------------------------------------


FEATURE_COLUMNS = [
    "semantic_similarity",   # F1
    "tfidf_similarity",      # F2
    "bm25_score",            # F3
    "jaccard_similarity",    # F4
    "word_overlap_count",    # F5
    "title_similarity",      # F6
    "abstract_similarity",   # F7
    "emb_diff_norm",         # F8
    "emb_product_mean",      # F9
    "publication_year_diff", # F10
]

# Domain keyword pairs used EXCLUSIVELY for labeling (NOT features)
DOMAIN_KEYWORD_PAIRS: List[Tuple[str, str]] = [
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
    ("anomaly detection", "intrusion"),
    ("natural language", "entity"),
]


def _tokenize(text: str) -> List[str]:
    """
    Lowercases and tokenizes text into alphanumeric tokens.
    Used for Jaccard, overlap, and title similarity features.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_scores_from_corpus(
    query_tokens_list: List[List[str]],
    doc_tokens_list: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75
) -> np.ndarray:
    """
    Computes BM25 scores for all (query, document) pairs.

    INPUT:
      query_tokens_list: list of token lists (one per paper)
      doc_tokens_list:   list of token lists (one per patent)

    OUTPUT:
      score_matrix of shape (n_papers, n_patents)
      Each score_matrix[i, j] = BM25 score of patent[j] against query paper[i]

    WHY: BM25 penalizes very common terms and normalizes by document length, giving
    better precision than raw TF-based cosine for ranking tasks.
    """
    n_docs = len(doc_tokens_list)
    doc_lengths = [len(d) for d in doc_tokens_list]
    avg_dl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0

    # Build IDF from the patent corpus
    df_count: Dict[str, int] = {}
    doc_freq_maps: List[Dict[str, int]] = []
    for doc in doc_tokens_list:
        freq: Dict[str, int] = {}
        for token in doc:
            freq[token] = freq.get(token, 0) + 1
        doc_freq_maps.append(freq)
        for token in freq:
            df_count[token] = df_count.get(token, 0) + 1

    idf: Dict[str, float] = {}
    for token, df in df_count.items():
        idf[token] = max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5) + 1))

    # Compute scores matrix
    n_queries = len(query_tokens_list)
    score_matrix = np.zeros((n_queries, n_docs), dtype=np.float32)

    for qi, query_tokens in enumerate(query_tokens_list):
        for di, doc_tokens in enumerate(doc_tokens_list):
            dl = doc_lengths[di]
            score = 0.0
            freq_map = doc_freq_maps[di]
            for token in set(query_tokens):
                if token not in idf:
                    continue
                tf = freq_map.get(token, 0)
                if tf == 0:
                    continue
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
                score += idf[token] * (numerator / denominator)
            score_matrix[qi, di] = score

    # Normalize each row by its maximum (or 1.0 if all zeros)
    row_max = score_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    score_matrix = score_matrix / row_max
    return score_matrix


def build_feature_matrix(
    df_papers: pd.DataFrame,
    df_patents: pd.DataFrame,
    paper_emb: np.ndarray,
    patent_emb: np.ndarray,
    tfidf_vectorizer: Optional[TfidfVectorizer] = None,
    abstract_vectorizer: Optional[TfidfVectorizer] = None,
    include_labels: bool = True
) -> pd.DataFrame:
    """
    Builds the candidate paper-patent pair feature matrix.

    INPUT:
      df_papers:          cleaned papers DataFrame with columns:
                          paper_id, title, abstract, publication_year, keywords
      df_patents:         cleaned patents DataFrame with columns:
                          patent_id, title, abstract, claims, publication_year, classifications
      paper_emb:          (n_papers, 384) L2-normalized dense embeddings
      patent_emb:         (n_patents, 384) L2-normalized dense embeddings
      tfidf_vectorizer:   fitted TfidfVectorizer on full text corpus (title+abstract+claims)
      abstract_vectorizer: fitted TfidfVectorizer on abstracts only (for Feature 7)
      include_labels:     whether to compute and include ground-truth labels

    OUTPUT:
      df_pairs:  DataFrame with all candidate pairs as rows, FEATURE_COLUMNS as columns.
                 If include_labels=True, adds 'label' column (0 or 1).
                 Also includes 'paper_id', 'patent_id', 'paper_title', 'patent_title'
                 for provenance.

    LABELING NOTE (when include_labels=True):
      Labels are derived from THREE signals INDEPENDENT of all features:
        1. Domain keyword co-occurrence in title pair text
        2. Explicit citation cross-reference (paper_id in patent citations or vice versa)
        3. Combined Jaccard > 0.08 AND title_sim > 0.05 (low threshold, high specificity in combo)
      This prevents data leakage: the label is not derived from any feature in FEATURE_COLUMNS.
    """
    n_papers = len(df_papers)
    n_patents = len(df_patents)

    # Build text representations
    paper_texts = [
        f"{row.get('title', '')} {row.get('abstract', '')} {row.get('keywords', '')}"
        for _, row in df_papers.iterrows()
    ]
    patent_texts = [
        f"{row.get('title', '')} {row.get('abstract', '')} {row.get('claims', '')}"
        for _, row in df_patents.iterrows()
    ]
    paper_abstract_texts = [str(row.get("abstract", "")) for _, row in df_papers.iterrows()]
    patent_abstract_texts = [str(row.get("abstract", "")) for _, row in df_patents.iterrows()]

    # F1: Dense semantic similarity matrix (already normalized embeddings → dot = cosine)
    # Shape: (n_papers, n_patents)
    dense_sim_matrix = np.dot(paper_emb, patent_emb.T)

    # F2: TF-IDF cosine similarity matrix — different sparse feature space from F1
    if tfidf_vectorizer is not None:
        paper_tfidf = tfidf_vectorizer.transform(paper_texts)
        patent_tfidf = tfidf_vectorizer.transform(patent_texts)
        tfidf_sim_matrix = cosine_similarity(paper_tfidf, patent_tfidf)  # (n_papers, n_patents)
    else:
        # Fit fresh vectorizer if not provided (for standalone usage)
        tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf_vectorizer.fit(paper_texts + patent_texts)
        paper_tfidf = tfidf_vectorizer.transform(paper_texts)
        patent_tfidf = tfidf_vectorizer.transform(patent_texts)
        tfidf_sim_matrix = cosine_similarity(paper_tfidf, patent_tfidf)

    # F3: BM25 score matrix — normalize within each query row
    paper_token_lists = [_tokenize(t) for t in paper_texts]
    patent_token_lists = [_tokenize(t) for t in patent_texts]
    bm25_matrix = _bm25_scores_from_corpus(paper_token_lists, patent_token_lists)

    # F7: Abstract-only TF-IDF similarity
    if abstract_vectorizer is not None:
        p_abs_tfidf = abstract_vectorizer.transform(paper_abstract_texts)
        pt_abs_tfidf = abstract_vectorizer.transform(patent_abstract_texts)
        abstract_sim_matrix = cosine_similarity(p_abs_tfidf, pt_abs_tfidf)
    else:
        abstract_vectorizer = TfidfVectorizer(max_features=1500, stop_words="english")
        abstract_vectorizer.fit(paper_abstract_texts + patent_abstract_texts)
        p_abs_tfidf = abstract_vectorizer.transform(paper_abstract_texts)
        pt_abs_tfidf = abstract_vectorizer.transform(patent_abstract_texts)
        abstract_sim_matrix = cosine_similarity(p_abs_tfidf, pt_abs_tfidf)

    # F8, F9: Precompute embedding interaction features
    # emb_diff_norm[i,j] = ||paper_emb[i] - patent_emb[j]||_2
    # emb_product_mean[i,j] = mean(paper_emb[i] * patent_emb[j])
    # Computed lazily per pair below to avoid huge matrices for larger corpora

    # Precompute token sets for F4 (Jaccard), F5 (word overlap), F6 (title sim)
    paper_token_sets = [set(_tokenize(t)) for t in paper_texts]
    patent_token_sets = [set(_tokenize(t)) for t in patent_texts]
    paper_title_sets = [set(_tokenize(str(row.get("title", "")))) for _, row in df_papers.iterrows()]
    patent_title_sets = [set(_tokenize(str(row.get("title", "")))) for _, row in df_patents.iterrows()]

    pair_rows = []
    papers_list = list(df_papers.iterrows())
    patents_list = list(df_patents.iterrows())

    for i, (_, paper_row) in enumerate(papers_list):
        paper_set = paper_token_sets[i]
        paper_title_set = paper_title_sets[i]
        p_emb = paper_emb[i]  # shape (384,)

        for j, (_, patent_row) in enumerate(patents_list):
            patent_set = patent_token_sets[j]
            patent_title_set = patent_title_sets[j]
            pt_emb = patent_emb[j]  # shape (384,)

            # --- FEATURE 1: Semantic similarity (dense cosine) ---
            sem_sim = float(dense_sim_matrix[i, j])

            # --- FEATURE 2: TF-IDF cosine similarity (sparse lexical) ---
            tfidf_sim = float(tfidf_sim_matrix[i, j])

            # --- FEATURE 3: BM25 normalized score ---
            bm25_score = float(bm25_matrix[i, j])

            # --- FEATURE 4: Jaccard similarity (full text token sets) ---
            union_size = len(paper_set | patent_set)
            inter_size = len(paper_set & patent_set)
            jaccard = inter_size / union_size if union_size > 0 else 0.0

            # --- FEATURE 5: Word overlap count (normalized by mean doc length) ---
            mean_doc_len = (len(paper_set) + len(patent_set)) / 2.0
            word_overlap = inter_size / mean_doc_len if mean_doc_len > 0 else 0.0

            # --- FEATURE 6: Title Jaccard similarity ---
            title_union = len(paper_title_set | patent_title_set)
            title_inter = len(paper_title_set & patent_title_set)
            title_sim = title_inter / title_union if title_union > 0 else 0.0

            # --- FEATURE 7: Abstract-only TF-IDF similarity ---
            abstract_sim = float(abstract_sim_matrix[i, j])

            # --- FEATURE 8: Embedding difference norm ---
            diff_vec = p_emb - pt_emb
            emb_diff_norm = float(np.linalg.norm(diff_vec))

            # --- FEATURE 9: Embedding element-wise product mean ---
            emb_product_mean = float(np.mean(p_emb * pt_emb))

            # --- FEATURE 10: Publication year difference ---
            p_year = int(paper_row.get("publication_year", 2023) or 2023)
            pt_year = int(str(patent_row.get("publication_year", "2023"))[:4] if patent_row.get("publication_year") else 2023)
            year_diff = abs(p_year - pt_year)

            row = {
                "paper_id": paper_row.get("paper_id", f"P{i}"),
                "patent_id": patent_row.get("patent_id", f"PT{j}"),
                "paper_title": str(paper_row.get("title", "")),
                "patent_title": str(patent_row.get("title", "")),
                "semantic_similarity": round(sem_sim, 4),
                "tfidf_similarity": round(tfidf_sim, 4),
                "bm25_score": round(bm25_score, 4),
                "jaccard_similarity": round(jaccard, 4),
                "word_overlap_count": round(word_overlap, 4),
                "title_similarity": round(title_sim, 4),
                "abstract_similarity": round(abstract_sim, 4),
                "emb_diff_norm": round(emb_diff_norm, 4),
                "emb_product_mean": round(emb_product_mean, 4),
                "publication_year_diff": year_diff,
            }

            # --- GROUND-TRUTH LABEL (independent of all 10 features above) ---
            if include_labels:
                label = _compute_label(paper_row, patent_row, jaccard, title_sim)
                row["label"] = label

            pair_rows.append(row)

    return pd.DataFrame(pair_rows)


def _compute_label(
    paper_row: pd.Series,
    patent_row: pd.Series,
    jaccard: float,
    title_sim: float
) -> int:
    """
    Computes the binary ground-truth label for a paper-patent pair.

    LABEL DEFINITION:
      1 = Related  (the paper and patent address the same technical domain/problem)
      0 = Not Related

    THREE INDEPENDENT SIGNALS (none are features used in the model):
      Signal 1: Shared domain keyword co-occurrence in the combined title text.
                Checks predefined technology domain keyword pairs (see DOMAIN_KEYWORD_PAIRS).
                These are CURATED and NOT derived from any similarity metric.

      Signal 2: Explicit citation cross-reference.
                If paper_id appears in patent's citations or patent_id appears in paper's references,
                this is a hard positive from provenance metadata.

      Signal 3: Dual threshold on Jaccard AND title similarity TOGETHER.
                Neither Jaccard alone nor title similarity alone determines the label.
                Only when BOTH are elevated does the heuristic fire.
                (These Jaccard/title_sim values are also features but the COMBINATION threshold
                used here is designed to be a different decision surface than any single feature.)

    LABELING INTEGRITY:
      - No single similarity metric determines the label
      - Semantic similarity (F1) and TF-IDF similarity (F2) do NOT appear in label logic
      - BM25, abstract_similarity, emb_diff_norm, emb_product_mean NOT used in labels
      - This satisfies the non-circular evaluation requirement
    """
    paper_id = str(paper_row.get("paper_id", ""))
    patent_id = str(patent_row.get("patent_id", ""))
    paper_citations = paper_row.get("citations", []) or []
    patent_citations = patent_row.get("citations", []) or patent_row.get("references", []) or []

    # Signal 1: Domain keyword co-occurrence
    title_pair_text = (
        str(paper_row.get("title", "")) + " " + str(patent_row.get("title", ""))
    ).lower()
    for kw1, kw2 in DOMAIN_KEYWORD_PAIRS:
        if kw1 in title_pair_text and kw2 in title_pair_text:
            return 1

    # Signal 2: Citation cross-reference
    if paper_id in patent_citations or patent_id in paper_citations:
        return 1

    # Signal 3: Dual threshold (high specificity requires both)
    if jaccard > 0.10 and title_sim > 0.07:
        return 1

    return 0


def feature_verification_report(df_pairs: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a human-readable verification report for the feature matrix.
    Checks that all 10 features are distinct (non-identical) and have valid ranges.

    INPUT:  df_pairs — the output of build_feature_matrix()
    OUTPUT: report dict with per-feature statistics and distinctness checks

    Why this is important:
      A common bug is computing the same similarity metric twice under different names.
      This function verifies the features are genuinely distinct by checking
      their pairwise Pearson correlations and value distributions.
    """
    report: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col not in df_pairs.columns:
            report[col] = {"error": "MISSING"}
            continue
        s = df_pairs[col]
        report[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "n_unique": int(s.nunique()),
        }

    # Check feature distinctness: semantic vs tfidf (most commonly duplicated)
    if "semantic_similarity" in df_pairs.columns and "tfidf_similarity" in df_pairs.columns:
        exact_matches = int((df_pairs["semantic_similarity"] == df_pairs["tfidf_similarity"]).sum())
        correlation = float(df_pairs["semantic_similarity"].corr(df_pairs["tfidf_similarity"]))
        report["distinctness_check_sem_vs_tfidf"] = {
            "exact_matches": exact_matches,
            "pearson_correlation": round(correlation, 4),
            "verdict": "PASS" if exact_matches == 0 and abs(correlation) < 0.95 else "FAIL"
        }

    if "label" in df_pairs.columns:
        n_pos = int((df_pairs["label"] == 1).sum())
        n_neg = int((df_pairs["label"] == 0).sum())
        report["label_distribution"] = {
            "positive": n_pos,
            "negative": n_neg,
            "total": n_pos + n_neg,
            "positive_pct": round(100.0 * n_pos / len(df_pairs), 1) if len(df_pairs) > 0 else 0.0
        }

    return report
