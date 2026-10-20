"""
tfidf.py
========
TF-IDF sparse lexical retrieval engine.
Complements BM25 with exact n-gram term frequency-inverse document frequency weighting.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.api.schemas import ResearchPaper, Patent
from app.logging_config import logger


class TFIDFRetriever:
    """TF-IDF sparse vector retrieval model for scientific papers and patent documents."""

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True
        )
        self.doc_ids: List[str] = []
        self.doc_types: List[str] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def index_documents(self, papers: List[ResearchPaper], patents: List[Patent]) -> None:
        """Indexes papers and patents into a unified TF-IDF vector space."""
        self.doc_ids = []
        self.doc_types = []
        corpus_texts = []

        for p in papers:
            self.doc_ids.append(p.paper_id)
            self.doc_types.append("paper")
            text = f"{p.title}. {p.abstract or ''} {' '.join(p.keywords or [])}"
            corpus_texts.append(text)

        for pt in patents:
            self.doc_ids.append(pt.patent_id)
            self.doc_types.append("patent")
            claims_text = " ".join(pt.claims or [])
            text = f"{pt.title}. {pt.abstract or ''} {claims_text}"
            corpus_texts.append(text)

        if corpus_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
            self.is_indexed = True
            logger.info(f"TFIDFRetriever indexed {len(corpus_texts)} documents (vocab size={len(self.vectorizer.vocabulary_)}).")
        else:
            self.is_indexed = False

    def search(self, query: str, top_k: int = 10, doc_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Searches the TF-IDF space and returns list of (doc_id, cosine_similarity_score).
        """
        if not self.is_indexed or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        results = []
        for idx, score in enumerate(sims):
            if doc_type and self.doc_types[idx] != doc_type:
                continue
            if score > 0.0:
                results.append((self.doc_ids[idx], float(score), self.doc_types[idx]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_pair_similarity(self, text_a: str, text_b: str) -> float:
        """Computes TF-IDF cosine similarity between two arbitrary texts."""
        if not self.is_indexed:
            vec = TfidfVectorizer(stop_words="english")
            try:
                mat = vec.fit_transform([text_a, text_b])
                return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
            except ValueError:
                return 0.0

        vecs = self.vectorizer.transform([text_a, text_b])
        return float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
