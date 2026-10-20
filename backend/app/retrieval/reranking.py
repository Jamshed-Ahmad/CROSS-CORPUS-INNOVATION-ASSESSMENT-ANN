"""
reranking.py
============
Query-Aware Second-Stage Neural Reranker and Field-Aware Candidate Filter.
Evaluates fine-grained interaction between structured research queries and candidate
papers/patents, computing Research Relevance Scores (0-100), Research Value Scores (0-100),
relevance tiers, and enforcing hard domain gates to prevent irrelevant recommendations.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from app.api.schemas import ResearchPaper, Patent, ResearchIdea
from app.logging_config import logger

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "you", "your", "system", "method", "apparatus", "device", "based", "using"
}


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ["tions", "tion", "ings", "ing", "ed", "es", "s"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def _tokenize(text: str) -> Set[str]:
    raw_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', text) if w.lower() not in STOPWORDS and len(w) > 2]
    tokens = set()
    for w in raw_words:
        tokens.add(w)
        stem = _stem(w)
        if stem:
            tokens.add(stem)
    return tokens


def _jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


def _overlap_ratio(query_tokens: Set[str], doc_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(doc_tokens)) / len(query_tokens)


class CrossEncoderReranker:
    """
    Second-stage query-aware cross-attention reranker with field-aware weighting,
    hard relevance filtering, and research-value scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold: float = 40.0,
        use_neural_cross_encoder: bool = True
    ):
        self.relevance_threshold = relevance_threshold
        self.use_neural = use_neural_cross_encoder and _CROSS_ENCODER_AVAILABLE
        self.cross_encoder: Optional[Any] = None

        if self.use_neural:
            try:
                logger.info(f"CrossEncoderReranker: Initializing cross-encoder '{model_name}'...")
                self.cross_encoder = CrossEncoder(model_name, max_length=512)
                logger.info("CrossEncoderReranker: Neural cross-encoder loaded successfully.")
            except Exception as e:
                logger.warning(f"CrossEncoderReranker: Could not initialize neural cross-encoder ({e}). Using deterministic field-aware reranker.")
                self.cross_encoder = None

    def rerank_papers(self, idea: ResearchIdea, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """
        Reranks academic papers using query decomposition, field-aware weighting,
        and hard relevance gating.
        """
        if not papers:
            return []

        query_tokens = _tokenize(idea.original_text)
        prob_tokens = _tokenize(idea.parsed_problem)
        tech_tokens = _tokenize(idea.proposed_technology)
        domain_tokens = _tokenize(idea.domain)

        scored_papers: List[Tuple[float, float, str, str, ResearchPaper]] = []

        # Prepare pairs for batch neural cross-encoder if available
        neural_scores = {}
        if self.cross_encoder:
            try:
                pairs = [(idea.original_text, f"{p.title}. {p.abstract or ''}") for p in papers]
                raw_preds = self.cross_encoder.predict(pairs)
                for idx, p in enumerate(papers):
                    # Sigmoid transform to [0, 1]
                    s = 1.0 / (1.0 + math.exp(-float(raw_preds[idx])))
                    neural_scores[p.paper_id] = s
            except Exception as e:
                logger.warning(f"CrossEncoder inference note: {e}")

        for p in papers:
            title_tokens = _tokenize(p.title)
            abstract_tokens = _tokenize(p.abstract or "")
            keyword_tokens = set(k.lower() for k in p.keywords)

            # Field-Aware Alignment
            title_prob_overlap = _overlap_ratio(prob_tokens, title_tokens)
            title_tech_overlap = _overlap_ratio(tech_tokens, title_tokens)
            title_score = 0.6 * title_prob_overlap + 0.4 * title_tech_overlap

            abstract_overlap = _overlap_ratio(query_tokens, abstract_tokens)
            keyword_overlap = _overlap_ratio(query_tokens, keyword_tokens)
            domain_alignment = _overlap_ratio(domain_tokens, title_tokens.union(abstract_tokens))

            # Cross-encoder signal
            neural_sim = neural_scores.get(p.paper_id, (title_score + abstract_overlap) / 2.0)

            # Weighted Field-Aware Relevance Score (0 - 100)
            # Title: 25%, Abstract: 35%, Neural Cross-Encoder: 20%, Keywords: 10%, Domain: 10%
            raw_relevance = (
                0.25 * title_score +
                0.35 * abstract_overlap +
                0.20 * neural_sim +
                0.10 * keyword_overlap +
                0.10 * domain_alignment
            )

            # Hard Gate Penalty: if paper has zero overlap with core problem and technology
            core_overlap = len(prob_tokens.union(tech_tokens).intersection(title_tokens.union(abstract_tokens)))
            if core_overlap == 0:
                raw_relevance *= 0.25

            relevance_score = round(min(100.0, max(0.0, raw_relevance * 100.0)), 1)

            # Research Value Score (0 - 100)
            # Relevance (40%) + Text Specificity (25%) + Core Alignment (20%) + Citation Credibility (15%)
            text_len = len(p.abstract or "")
            specificity = min(1.0, text_len / 1200.0) if text_len > 0 else 0.2
            cite_signal = min(1.0, math.log1p(len(p.references)) / 3.0)
            raw_value = (
                0.40 * (relevance_score / 100.0) +
                0.25 * specificity +
                0.20 * (title_prob_overlap + abstract_overlap) / 2.0 +
                0.15 * cite_signal
            )
            value_score = round(min(100.0, max(0.0, raw_value * 100.0)), 1)

            # Relevance Tier
            if relevance_score >= 85.0:
                tier = "Highly Recommended"
            elif relevance_score >= 70.0:
                tier = "Strongly Relevant"
            elif relevance_score >= 50.0:
                tier = "Relevant"
            elif relevance_score >= 35.0:
                tier = "Weakly Relevant"
            else:
                tier = "Not Recommended"

            # Explainable rationale
            matched_terms = list(query_tokens.intersection(title_tokens.union(abstract_tokens)))[:4]
            why_relevant = (
                f"Addresses '{idea.parsed_problem}' using related technical concepts: {', '.join(matched_terms)}."
                if matched_terms else f"Contextual domain alignment with '{idea.domain}'."
            )

            p.relevance_score = relevance_score
            p.research_value_score = value_score
            p.relevance_tier = tier
            p.why_relevant = why_relevant

            scored_papers.append((relevance_score, value_score, tier, why_relevant, p))

        # Sort primarily by relevance_score, then research_value_score
        scored_papers.sort(key=lambda x: (x[0], x[1]), reverse=True)
        ranked = [item[4] for item in scored_papers]

        logger.info(
            f"CrossEncoderReranker reranked {len(papers)} papers. "
            f"Top Paper: '{ranked[0].title[:45]}...' (Score={ranked[0].relevance_score}, Tier={ranked[0].relevance_tier})"
        )
        return ranked

    def rerank_patents(self, idea: ResearchIdea, patents: List[Patent]) -> List[Patent]:
        """
        Reranks industrial patents with heavy claim weighting, technical mechanism
        matching, and hard domain filtering to reject unrelated prior art.
        """
        if not patents:
            return []

        query_tokens = _tokenize(idea.original_text)
        prob_tokens = _tokenize(idea.parsed_problem)
        tech_tokens = _tokenize(idea.proposed_technology)
        domain_tokens = _tokenize(idea.domain)

        # 1. Neural Cross-Encoder Scoring for Patents
        neural_scores = []
        if self.cross_encoder is not None:
            try:
                pairs = [
                    (
                        f"{idea.original_text}. Problem: {idea.parsed_problem}. Technology: {idea.proposed_technology}",
                        f"{pt.title}. {pt.abstract or ''}. Claims: {' '.join(pt.claims[:2]) if pt.claims else ''}"
                    )
                    for pt in patents
                ]
                raw_scores = self.cross_encoder.predict(pairs)
                min_s = float(np.min(raw_scores))
                max_s = float(np.max(raw_scores))
                spread = max_s - min_s if max_s > min_s else 1.0
                neural_scores = [float((s - min_s) / spread) for s in raw_scores]
            except Exception as e:
                logger.warning(f"CrossEncoder patent inference note: {e}")
                neural_scores = [0.5] * len(patents)
        else:
            neural_scores = [0.5] * len(patents)

        scored_patents: List[Tuple[float, str, str, Patent]] = []

        for i, pt in enumerate(patents):
            title_tokens = _tokenize(pt.title)
            abstract_tokens = _tokenize(pt.abstract or "")
            claims_text = " ".join(pt.claims or [])
            claim_tokens = _tokenize(claims_text)
            cpc_text = " ".join(pt.classifications)
            cpc_tokens = _tokenize(cpc_text)

            # Field-Aware Matching (Claims 30%, Neural 30%, Title 15%, Abstract 15%, Domain 10%)
            claim_tech_overlap = _overlap_ratio(tech_tokens, claim_tokens)
            claim_prob_overlap = _overlap_ratio(prob_tokens, claim_tokens)
            claim_score = 0.5 * claim_tech_overlap + 0.5 * claim_prob_overlap if pt.claims else 0.5 * _overlap_ratio(query_tokens, abstract_tokens)

            title_score = _overlap_ratio(tech_tokens.union(prob_tokens), title_tokens)
            abstract_score = _overlap_ratio(query_tokens, abstract_tokens)
            domain_score = _overlap_ratio(domain_tokens, title_tokens.union(abstract_tokens).union(claim_tokens))
            neural_score = neural_scores[i]

            raw_relevance = (
                0.30 * claim_score +
                0.30 * neural_score +
                0.15 * title_score +
                0.15 * abstract_score +
                0.10 * domain_score
            )

            # HARD RELEVANCE GATE:
            # If patent shares ZERO tokens with the proposed technology or problem,
            # or if it is from an entirely contradictory technical domain,
            # severely penalize to ensure it falls into 'Not Recommended'.
            combined_tokens = title_tokens.union(abstract_tokens).union(claim_tokens)
            core_overlap = len(tech_tokens.union(prob_tokens).intersection(combined_tokens))
            
            if core_overlap == 0:
                raw_relevance *= 0.15
            else:
                # If both tech and problem match claims, apply alignment boost
                if claim_tech_overlap > 0 and claim_prob_overlap > 0:
                    raw_relevance = min(1.0, raw_relevance * 1.35)

            relevance_score = round(min(100.0, max(0.0, raw_relevance * 100.0)), 1)

            # Tier
            if relevance_score >= 80.0:
                tier = "Highly Recommended"
            elif relevance_score >= 65.0:
                tier = "Strongly Relevant"
            elif relevance_score >= 45.0:
                tier = "Relevant"
            elif relevance_score >= 30.0:
                tier = "Weakly Relevant"
            else:
                tier = "Not Recommended"

            # Matched technical components
            matched_tech = ", ".join(list(tech_tokens.intersection(combined_tokens))[:3]) or "General industrial apparatus"
            matched_prob = ", ".join(list(prob_tokens.intersection(combined_tokens))[:3]) or "Domain technical specifications"
            
            why_relevant = (
                f"Patent claims disclose mechanisms related to '{matched_tech}' targeting '{matched_prob}'."
                if core_overlap > 0 else f"General classification overlap with {pt.classifications[:2]}."
            )

            # Find relevant claim sentences
            claim_evidence = []
            if pt.claims:
                for cl in pt.claims:
                    if any(t in cl.lower() for t in tech_tokens.union(prob_tokens)):
                        claim_evidence.append(cl[:200])
                    if len(claim_evidence) >= 2:
                        break
            if not claim_evidence and pt.abstract:
                claim_evidence.append(pt.abstract[:200])

            pt.relevance_score = relevance_score
            pt.relevance_tier = tier
            pt.technical_match = matched_tech
            pt.matched_problem = matched_prob
            pt.matched_technology = idea.proposed_technology
            pt.matched_application = idea.application_area
            pt.why_relevant = why_relevant
            pt.claim_evidence = claim_evidence

            scored_patents.append((relevance_score, tier, why_relevant, pt))

        # Sort patents by relevance score
        scored_patents.sort(key=lambda x: x[0], reverse=True)
        ranked = [item[3] for item in scored_patents]

        logger.info(
            f"CrossEncoderReranker reranked {len(patents)} patents. "
            f"Top Patent: '{ranked[0].title[:45]}...' (Score={ranked[0].relevance_score}, Tier={ranked[0].relevance_tier})"
        )
        return ranked
