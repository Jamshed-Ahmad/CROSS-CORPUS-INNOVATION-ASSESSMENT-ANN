# System Models & Tech Stack Specification

**Project Title:** Evidence-Grounded AI Framework for Cross-Corpus Innovation Assessment  
**Document Purpose:** Detailed technical reference of all models, algorithms, neural architectures, frameworks, and technologies utilized across the system.  
**Architecture Version:** 2.0.0 (Neural Architecture)

---

## 🏛️ 1. Architectural Division of Responsibilities

The system enforces strict separation across 5 core stages:
1. **Hybrid RAG Retrieval Layer:** Finds relevant papers and patents from OpenAlex and EPO OPS using multi-modal search without making relationship assertions.
2. **Siamese / Dual-Tower + Cross-Attention Model:** Learns fine-grained semantic and technical relationships between paper-patent candidate pairs ($P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0, 1]$).
3. **Research Gap Detection Network:** A multi-task PyTorch network that analyzes cross-corpus coverage, unresolved technical limitations, combination rarity, and relationship scores to detect and categorize research gaps.
4. **Innovation Opportunity Scorer:** Quantifies multi-factor novelty, feasibility, and market potential.
5. **Evidence Validation Layer:** Enforces strict provenance; discards any gap without ground-truth cited passages.

---

## 🤖 2. System Models Specification

### A. Hybrid RAG Retrieval Layer (Preserved)

1. **Okapi BM25 Sparse Lexical Model**
   * **Library:** `rank-bm25`
   * **Purpose:** Exact technical-term matching, domain keyword matching, and author/assignee name retrieval.

2. **TF-IDF Sparse Lexical Model**
   * **Module:** `backend/app/retrieval/tfidf.py` (`scikit-learn`)
   * **Purpose:** Rare domain keyword weighting across heterogeneous vocabularies of academic literature and patent claims.

3. **Sentence Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)**
   * **Framework:** PyTorch & Hugging Face `sentence-transformers`
   * **Vector Dimension:** 384 dimensions
   * **Purpose:** Dense contextual representation encoding for papers, patent claims, and user research ideas.

4. **FAISS Dense Vector Index Model**
   * **Library:** `faiss-cpu` (Facebook AI Similarity Search)
   * **Index Type:** `IndexFlatIP` (Maximum Inner Product / Cosine Similarity)
   * **Purpose:** Sub-millisecond dense semantic search across indexed paper and patent embeddings.

5. **Reciprocal Rank Fusion (RRF) Model**
   * **Module:** `backend/app/retrieval/fusion.py`
   * **Formula:** $RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$
   * **Purpose:** Robust multi-retriever rank fusion combining BM25, TF-IDF, and FAISS rankings without score scale bias.

6. **Pretrained Neural Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**
   * **Framework:** PyTorch & Hugging Face `sentence-transformers`
   * **Module:** `backend/app/retrieval/reranking.py` (`CrossEncoderReranker`)
   * **Purpose:** Full query-document cross-attention evaluating technical relationship beyond dual-tower vector dot-products.
   * **Field-Aware Relevance:** Papers weighted by Abstract (35%), Title (25%), Neural (20%), Keywords (10%), Domain (10%). Patents weighted by Claims (30%), Neural (30%), Title (15%), Abstract (15%), Domain (10%).
   * **Hard Relevance Filtering:** Eliminates candidate patents lacking core technical/problem overlap to prevent false prior art. Produces Research Relevance Score (0-100) and Research Value Score (0-100).

---

### B. Neural Relationship Model (NEW: Replaces PaperPatentANN & GPT-4o)

7. **Siamese / Dual-Tower + Cross-Attention Model**
   * **Module:** `backend/app/models/relationship_model.py` (`DualTowerCrossAttentionRelationshipModel`)
   * **Framework:** PyTorch 2.x
   * **Architecture:**
     - **Paper Tower (`PaperTowerEncoder`):** Linear projection + LayerNorm + Dropout mapping 384-d paper embeddings to 256-d latent space.
     - **Patent Tower (`PatentTowerEncoder`):** Linear projection + LayerNorm + Dropout mapping 384-d patent embeddings to 256-d latent space.
     - **Cross-Attention (`BiDirectionalCrossAttention`):** Multi-Head Bidirectional Cross-Attention (4 attention heads, $d_k=64$, projection dimension 256) learning bidirectional concept interactions:
       $$A_{\text{paper} \to \text{patent}} = \operatorname{Softmax}\left(\frac{Q_{\text{paper}} K_{\text{patent}}^T}{\sqrt{d_k}}\right) V_{\text{patent}}$$
     - **RAG Feature Fusion Projection:** 10-dimensional projection layer taking normalized retrieval features ($S_{\text{bm25}}, S_{\text{tfidf}}, S_{\text{semantic}}, D_{\text{faiss}}, S_{\text{rrf}}, Jaccard, Overlap, S_{\text{title}}, \Delta_{\text{year}}, CPC_{\text{overlap}}$).
     - **Classification Head:** Dense $(256 \times 5 + 64) \to 256 \to 128 \to 1$ with LeakyReLU and Sigmoid activation.
   * **Output:** $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0, 1]$.
   * **Checkpoint:** `models/relationship_model/relationship_model.pt`.

---

### C. Research Gap Detection Network (NEW: Replaces GPT-4o Analysis)

7. **Evidence-Grounded Multi-Task Gap Detection Network**
   * **Module:** `backend/app/models/gap_detection_model.py` (`ResearchGapDetectionNetwork`)
   * **Framework:** PyTorch 2.x
   * **Input Vector (12 features):** Relationship score, academic density, patent density, academic-to-patent ratio, recurring limitation frequency, technology combination rarity, semantic novelty, retrieval reciprocal rank, evidence strength, temporal publication velocity, domain breadth, unresolved bottleneck signal.
   * **Architecture:**
     - **Feature Saliency Attention:** Self-attention layer dynamically weighting salient gap indicators.
     - **Shared Backbone:** Dense $128 \to \text{ReLU} \to \text{Dropout}(0.2) \to 64 \to \text{ReLU}$.
     - **Multi-Task Head 1 (Gap Score):** Dense $64 \to 1$ with Sigmoid ($\text{BCE Loss}$, continuous gap probability).
     - **Multi-Task Head 2 (Gap Type):** Dense $64 \to 5$ with LogSoftmax ($\text{Cross-Entropy Loss}$, 5 categories: `TECHNICAL_GAP`, `RESEARCH_TO_APPLICATION_GAP`, `COMBINATION_GAP`, `PATENT_COVERAGE_GAP`, `APPLICATION_GAP`).
     - **Multi-Task Head 3 (Confidence):** Dense $64 \to 1$ with Sigmoid ($\text{MSE Loss}$, evidence grounding confidence).
   * **Checkpoint:** `models/gap_detection_model/gap_detection_model.pt`.

---

### D. Decision & Validation Layers (Preserved & Refactored)

8. **Composite Innovation Scorer**
   * **Module:** `backend/app/analysis/innovation_scoring.py`
   * **Purpose:** Transparent multi-factor opportunity scoring weighting novelty, relationship strength, and gap magnitude.

9. **Strict Evidence Validation**
   * **Module:** `backend/app/services/evidence_validation.py`
   * **Purpose:** Hallucination prevention; requires exact passage text, source ID, and verified URL for every presented finding.

---

## 🛠️ 3. Technology Stack Reference

| Category | Technology / Library | Version | Purpose & Scope |
| :--- | :--- | :--- | :--- |
| **Backend Core** | **Python** | 3.10 - 3.13 | Core runtime and numerical processing |
| **Deep Learning** | **PyTorch** | `>=2.1.0` | Dual-Tower Cross-Attention and Multi-Task Gap Network |
| **Transformers** | **Sentence-Transformers / HF** | `>=2.6.0` | Dense text embeddings (`all-MiniLM-L6-v2`) |
| **API Framework** | **FastAPI** | `>=0.110.0` | High-performance asynchronous REST API |
| **ASGI Server** | **Uvicorn** | `>=0.28.0` | Production ASGI web server |
| **Data Validation** | **Pydantic** | `>=2.6.0` | Strict API schema typing and input validation |
| **Vector Index** | **FAISS (`faiss-cpu`)** | `>=1.8.0` | Dense vector indexing and similarity search |
| **Lexical IR** | **Rank-BM25** | `>=0.2.2` | Sparse keyword matching |
| **Machine Learning** | **scikit-learn** | `>=1.4.0` | TF-IDF retriever, feature scalers, baseline benchmarks |
| **Data Analytics** | **NumPy & Pandas** | `>=1.26.0` / `>=2.2.0` | Array manipulation, parquet storage, evaluation reports |
| **Frontend UI** | **React + Vite** | `^18.2.0` / `^5.1.0` | Interactive dashboard UI with 3D canvas |
| **Styling** | **Vanilla CSS3** | Custom | Glassmorphic dark design system |

---

## 📋 4. Component Evolution (Old vs. New)

| Component | Status | Disposition / Replacement |
| :--- | :--- | :--- |
| **OpenAlex API** | **PRESERVED** | Live scholarly literature ingestion |
| **EPO OPS API** | **PRESERVED** | Live patent claims and classification ingestion |
| **BM25 Retriever** | **PRESERVED** | Sparse technical-term matching |
| **TF-IDF Retriever** | **PRESERVED / ENHANCED** | Rare domain keyword weighting |
| **Sentence-Transformer** | **PRESERVED** | Dense contextual representations (`MiniLM-L6-v2`) |
| **FAISS Vector Index** | **PRESERVED** | Sub-millisecond dense search |
| **Reciprocal Rank Fusion**| **PRESERVED** | Multi-retriever rank fusion ($k=60$) |
| **PaperPatentANN** | **REMOVED** | Replaced by PyTorch Dual-Tower + Cross-Attention |
| **GPT-4o Analysis** | **REMOVED** | Replaced by Multi-Task Research Gap Detection Network |
| **Dual-Tower Network** | **NEW** | Domain-specific paper & patent projections |
| **Cross-Attention** | **NEW** | 4-head bidirectional concept interaction learning |
| **Gap Detection Network**| **NEW** | Multi-Task evidence-grounded gap scoring & typing |
| **Evidence Validation** | **PRESERVED** | Strict citation and passage provenance verification |

