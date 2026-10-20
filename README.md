# Evidence-Grounded AI Framework for Cross-Corpus Innovation Assessment

An end-to-end, evidence-grounded AI decision-support platform that jointly analyzes research papers and patent literature to identify unexploited research gaps, prior-art boundaries, and multidimensional innovation opportunities.

---

## 🌟 Executive Summary

Traditional innovation analysis operates in domain silos: academic literature focuses on fundamental scientific discoveries, while patent databases emphasize commercial claims and prior art. This platform bridges the gap by implementing a **Hybrid RAG + Dual-Tower Cross-Attention + Multi-Task Research Gap Detection Pipeline** that grounds every innovation recommendation in traceable passages from both live OpenAlex paper literature and European Patent Office (EPO OPS) patent records.

---

## 🏛️ Architectural Division of Responsibilities

The system enforces strict separation of mathematical and empirical responsibilities:

1. **RAG Retrieval Responsibility (`app.retrieval`):**
   - Finds relevant papers and patents from OpenAlex and EPO OPS without making premature relationship or gap claims.
   - Combines BM25, TF-IDF, Sentence-Transformers (`all-MiniLM-L6-v2`), and FAISS via Reciprocal Rank Fusion (RRF, $k=60$).
2. **Siamese / Dual-Tower + Cross-Attention Model (`app.models.relationship_model`):**
   - Learns fine-grained semantic and technical relationships between paper-patent candidate pairs.
   - PyTorch architecture combining independent domain projections, bidirectional multi-head cross-attention ($Q, K, V$), and 10 normalized RAG retrieval features.
   - Outputs: $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0, 1]$.
3. **Research Gap Detection Network (`app.models.gap_detection_model`):**
   - Discovers and categorizes research gaps from cross-corpus coverage, unresolved technical limitations, combination rarity, and relationship scores.
   - Multi-task PyTorch network predicting Gap Score (BCE), Gap Category (5 classes, CE), and Confidence Score (MSE).
4. **Innovation Opportunity Scorer (`app.analysis.innovation_scoring`):**
   - Quantifies innovation potential, technical feasibility, and commercial prior-art differentiation.
5. **Evidence Grounding & Validation (`app.services.evidence_validation`):**
   - Enforces strict provenance; discards any gap without ground-truth cited passages and source URLs.

---

## 🏗️ System Architecture Flow

```text
                                  USER QUERY
                                       │
                         Query Analysis & Keyword Extraction
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     OpenAlex API (REST)                                   EPO OPS API (OAuth2)
   [Research Literature]                                    [Patent Claims & CPC]
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                          Data Preprocessing & Cleaning
                   (HTML stripping, token normalization, deduplication)
                                       │
                                       ▼
                            HYBRID RAG RETRIEVAL
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       BM25 Engine               TF-IDF Engine          Sentence-Transformer
   (Exact keyword match)     (Rare term weighting)      (MiniLM-L6-v2 dense)
            │                          │                          │
            │                          │                          ▼
            │                          │                      FAISS Index
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                       Reciprocal Rank Fusion (RRF, k=60)
                                       │
                                       ▼
                           Top-K Retrieved Papers + Patents
                                       │
                                       ▼
                           Candidate Pair Construction
                      (Top-K retrieval without Cartesian explosion)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
       Paper Representation                        Patent Representation
   (MiniLM + Paper Projection)                 (MiniLM + Patent Projection)
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
                     Bidirectional Multi-Head Cross-Attention
                         (Token & concept-level interaction)
                                       │
                                       ▼
                           Learned Interaction Vector
                                       +
                           10 Scaled RAG Features
             (BM25, TF-IDF, Semantic sim, FAISS dist, RRF, Jaccard, Overlap, etc.)
                                       │
                                       ▼
                         Dual-Tower Relationship Head
                                       │
                                       ▼
                      Paper-Patent Relationship Score ∈ [0, 1]
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │    RESEARCH GAP DETECTION NETWORK (PyTorch)│
                 │                                           │
                 │  Input: 12 Corpus & Evidence Features     │
                 │  (Coverage, Rarity, Limitations, Rel-Score)│
                 │                                           │
                 │          Feature Saliency Attention       │
                 │                 Dense Layers              │
                 │                                           │
                 │  [Head 1: Gap Score] [Head 2: Gap Type]  │
                 │        [Head 3: Confidence Score]         │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                              Evidence Validation
               (Strict passage extraction, URL verification, provenance)
                                       │
                                       ▼
                         Final Innovation Assessment
                      (Exported to Dashboard & API)
```

---

## 🔑 Environment Configuration

Copy `backend/.env.example` to `backend/.env`:

```bash
cd backend
cp .env.example .env
```

### Live Data Mode (`DATA_MODE=live`)
```env
DATA_MODE=live
OPENALEX_API_KEY=your_key_here
EPO_CONSUMER_KEY=your_epo_consumer_key
EPO_CONSUMER_SECRET=your_epo_consumer_secret
```

### Mock Data Mode (`DATA_MODE=mock`)
```env
DATA_MODE=mock
```
*Mock data mode uses local deterministic fixtures and does not require external API credentials or network connections.*

---

## 🧪 Model Training & Reproducible Pipeline

The project includes an end-to-end, leak-free training pipeline:

```bash
# 1. Fetch live corpus from OpenAlex and EPO OPS APIs
python training/collect_data.py

# 2. Build candidate pairs using hybrid RAG retrieval
python training/build_pairs.py

# 3. Compute 10 RAG features, independent non-circular labels, and paper-grouped splits
python training/generate_labels.py

# 4. Train Siamese / Dual-Tower + Cross-Attention Relationship Model
python training/train_relationship_model.py

# 5. Train Multi-Task Research Gap Detection Network
python training/train_gap_model.py

# 6. Run comparative baseline evaluation and ablation study
python training/evaluate.py
```

### Benchmark Results (Test Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.6184 | 0.7436 | 0.6042 | 0.6667 | 0.6756 |
| **Support Vector Machine (RBF)** | 0.6053 | 0.8000 | 0.5000 | 0.6154 | 0.7009 |
| **Random Forest (100 trees)** | 0.6711 | 0.8108 | 0.6250 | 0.7059 | 0.7016 |
| **Gradient Boosting Classifier** | 0.5395 | 0.6585 | 0.5625 | 0.6067 | 0.6321 |
| **Proposed Dual-Tower + Cross-Attention** | **0.7632** | **0.8125** | **0.8125** | **0.8125** | **0.7232** |

### Candidate Ranking Metrics
- **Mean Reciprocal Rank (MRR):** `0.8906`
- **Precision@1 / NDCG@1:** `0.8750`
- **Precision@3 / NDCG@3:** `0.7917 / 0.8750`
- **Precision@5 / NDCG@5:** `0.7250 / 0.8750`
- **Recall@10 / NDCG@10:** `1.0000 / 0.9209`

---

## 📓 Executable Research Notebook

A complete, self-contained research workflow is provided at:
`notebooks/cross_corpus_innovation_pipeline.ipynb`

It demonstrates:
1. Executive summary & technical contributions
2. Configurable API collection
3. Automatic corpus statistics computation
4. Hybrid RAG retrieval demonstration (BM25 + TF-IDF + FAISS + RRF)
5. Non-circular ground-truth labeling and leak-free splitting
6. Dual-Tower + Cross-Attention inference
7. Multi-Task Gap Detection Network inference
8. Baseline comparisons, ranking metrics, and ablation study
9. Complete end-to-end inference demonstration

---

## 🚀 Getting Started & Local Installation

### Prerequisites
- **Python 3.10 - 3.13**
- **Node.js 18+**

### 1. Backend Setup & Startup
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*API documentation is available at `http://127.0.0.1:8000/docs`.*

### 2. Frontend Setup & Startup
```bash
cd frontend
npm install
npm run dev
```
*The React + Vite Dashboard will be accessible at `http://localhost:5173`.*

### 3. Automated Unit Testing
```bash
python -m pytest backend/tests/unit/test_pipeline_unit.py -v
```

---

## ⚠️ Operational & Legal Disclaimer

> [!CAUTION]
> **Prototype Disclaimer**: The system is a research and decision-support prototype. It MUST NOT claim to establish legal patentability, legal freedom to operate, automated patent filing, or guaranteed commercial success. It clearly distinguishes preliminary computational assessments from decisions requiring legal, technical, intellectual-property, or commercial expertise.
