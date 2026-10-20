# System Architecture Specification

## 1. High-Level Pipeline Flow Architecture

```text
                               ┌───────────────────────────────────┐
                               │       User Research Idea          │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │           Query Parser            │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                   ┌───────────────────────────────────────────────────────────┐
                   │                     DATA_MODE Router                      │
                   └──────────────┬─────────────────────────────┬──────────────┘
                                  │                             │
                   DATA_MODE=live │                             │ DATA_MODE=mock
                                  ▼                             ▼
             ┌───────────────────────────────┐     ┌───────────────────────────┐
             │ OpenAlex API (Papers)         │     │ Mock Paper Fixture        │
             │ EPO OPS API (Patents)         │     │ Mock Patent Fixture       │
             └──────────────┬────────────────┘     └────────────┬──────────────┘
                            │                                   │
                            └─────────────────┬─────────────────┘
                                              │
                                              ▼
                               ┌───────────────────────────────────┐
                               │  Normalization & Text Cleaning    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │       HYBRID RAG RETRIEVAL        │
                               │                                   │
                               │  BM25 (Sparse Lexical)            │
                               │  TF-IDF (Domain Keyword Weight)   │
                               │  Sentence-Transformer (Dense)     │
                               │  FAISS Vector Index (MIP / L2)    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │   Reciprocal Rank Fusion (RRF)    │
                               │   RRF(d) = Σ 1 / (60 + r_m(d))    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │     Candidate Pair Generator      │
                               │  (Top-K retrieval without O(N*M)) │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │   DUAL-TOWER + CROSS-ATTENTION    │
                               │   RELATIONSHIP MODEL (PyTorch)    │
                               │                                   │
                               │  - Paper Tower Encoder            │
                               │  - Patent Tower Encoder           │
                               │  - Multi-Head Cross-Attention     │
                               │  - 10 RAG Features Fusion         │
                               │  - Relationship Prob ∈ [0, 1]     │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │  RESEARCH GAP DETECTION NETWORK   │
                               │  (Multi-Task Neural Network)      │
                               │                                   │
                               │  - Feature Saliency Attention     │
                               │  - Head 1: Gap Score ∈ [0, 1]     │
                               │  - Head 2: Gap Type (5 Classes)   │
                               │  - Head 3: Confidence Score       │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │    Composite Innovation Scorer    │
                               │  (Novelty, Feasibility, Impact)   │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │     Strict Evidence Validator     │
                               │  (Hallucination & Provenance Gate)│
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ Interactive React Dashboard +     │
                               │ 3D Animated Universe Canvas       │
                               └───────────────────────────────────┘
```

---

## 2. Architectural Separation of Responsibilities

The framework clearly separates five fundamental responsibilities:

1. **RAG Retrieval Responsibility:**
   - **Role:** Finds relevant papers, patents, and evidence passages without predicting relationships or claiming research gaps.
   - **Components:** OpenAlex API, EPO OPS API, BM25, TF-IDF, Sentence Transformer (`all-MiniLM-L6-v2`), FAISS, Reciprocal Rank Fusion (RRF).

2. **Relationship Learning Responsibility:**
   - **Role:** Learns the technological and semantic interaction between candidate paper-patent pairs.
   - **Components:** `DualTowerCrossAttentionRelationshipModel` (`PaperTowerEncoder`, `PatentTowerEncoder`, `BiDirectionalCrossAttention`, RAG feature fusion).
   - **Output:** $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0, 1]$.

3. **Research Gap Detection Responsibility:**
   - **Role:** Discovers and scores potential research gaps from cross-corpus coverage, unresolved technical limitations, combination rarity, and relationship scores.
   - **Components:** `ResearchGapDetectionNetwork` (Multi-Task heads for Gap Probability, Gap Category, and Grounding Confidence).

4. **Opportunity Assessment Responsibility:**
   - **Role:** Quantifies innovation potential, technical feasibility, and commercial prior-art differentiation.
   - **Components:** `InnovationScorer`.

5. **Evidence Grounding & Validation Responsibility:**
   - **Role:** Guarantees strict factual provenance; prevents unsupported or hallucinated claims by verifying that every finding has verbatim supporting text and source URLs.
   - **Components:** `EvidenceValidationService`.

---

## 3. Core Framework Modules

1. **`app.ingestion`**: Adapters for OpenAlex, EPO OPS (OAuth2 token negotiation, caching), and local fixtures.
2. **`app.retrieval`**: BM25 keyword indexer, TF-IDF sparse retriever, FAISS dense vector retriever, and RRF rank fusion.
3. **`app.features`**: Non-leaking relationship feature extraction (10 RAG signals) and gap candidate feature extraction (12 cross-corpus signals).
4. **`app.models`**: PyTorch neural models:
   - `paper_encoder.py` (`PaperTowerEncoder`)
   - `patent_encoder.py` (`PatentTowerEncoder`)
   - `cross_attention.py` (`BiDirectionalCrossAttention`)
   - `relationship_model.py` (`DualTowerCrossAttentionRelationshipModel`)
   - `gap_detection_model.py` (`ResearchGapDetectionNetwork`)
5. **`app.analysis`**: `RelationshipAnalyzer`, `GapDetector`, and `InnovationScorer`.
6. **`app.services`**: `AnalysisPipelineService` orchestrator and `EvidenceValidationService`.
7. **`app.api`**: FastAPI endpoints (`/api/ideas/analyze`, `/api/health`, `/api/search/papers`, `/api/search/patents`).
