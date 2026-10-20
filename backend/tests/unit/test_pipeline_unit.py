import sys
import os
import pytest

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.api.schemas import AnalysisRequest, ResearchPaper, Patent
from app.ingestion.preprocessing import preprocess_papers, preprocess_patents
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vector_index import VectorIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.analysis.cross_corpus import CrossCorpusAnalyzer
from app.analysis.rationale_extraction import RationaleExtractor
from app.analysis.gap_detection import GapDetector
from app.analysis.innovation_scoring import MultidimensionalScorer
from app.services.analysis_pipeline import AnalysisPipelineService

def test_preprocessing():
    raw_papers = [
        ResearchPaper(paper_id="P1", title="Title One ", abstract="Abstract text."),
        ResearchPaper(paper_id="P1", title="Title One Duplicate", abstract="Abstract text.")
    ]
    clean_p, profile = preprocess_papers(raw_papers)
    assert len(clean_p) == 1
    assert profile["duplicates_removed"] == 1

def test_retrieval_and_rrf():
    papers = [ResearchPaper(paper_id="P1", title="Graph Neural Networks in Drug Affinity", keywords=["graph"])]
    patents = [Patent(patent_id="T1", title="Graph Convolutions for Affinity Scoring", claims=["spatial graph"])]
    
    bm25 = BM25Retriever()
    bm25.index_documents(papers, patents)
    res_bm25 = bm25.search("graph neural networks")
    assert len(res_bm25) > 0

    encoder = EmbeddingGenerator()
    vindex = VectorIndex(encoder)
    vindex.index_documents(papers, patents)
    res_vec = vindex.search("graph neural networks")
    assert len(res_vec) > 0

    fused = reciprocal_rank_fusion([res_bm25, res_vec])
    assert len(fused) > 0

def test_cross_corpus_and_rationale():
    papers = [ResearchPaper(paper_id="P1", title="GNN for Drug Target", abstract="GNN achieves 14% improvement but fails on 3D dynamic shift.")]
    patents = [Patent(patent_id="T1", title="Substructure Scoring System", claims=["claim 1: spatial graph convolution"], citations=["P1"], description="Embodiments fail when ligand flexibility exceeds 12 bonds.")]
    
    analyzer = CrossCorpusAnalyzer()
    rels, ev1 = analyzer.analyze_relationships(papers, patents)
    assert len(rels) > 0

    extractor = RationaleExtractor()
    paper_rat, ev2 = extractor.extract_from_papers(papers)
    patent_rat, ev3 = extractor.extract_from_patents(patents)
    assert "P1" in paper_rat
    assert "T1" in patent_rat

from app.ingestion.papers import reconstruct_openalex_abstract, OpenAlexPaperAdapter
from app.ingestion.epo_auth import EPOAuthManager

def test_openalex_abstract_reconstruction():
    inv_index = {
        "Graph": [0],
        "neural": [1],
        "networks": [2],
        "improve": [3],
        "affinity": [4]
    }
    reconstructed = reconstruct_openalex_abstract(inv_index)
    assert reconstructed == "Graph neural networks improve affinity"
    assert reconstruct_openalex_abstract(None) is None
    assert reconstruct_openalex_abstract({}) is None

def test_epo_auth_manager():
    auth = EPOAuthManager(consumer_key="", consumer_secret="")
    token = auth.get_valid_token()
    assert token is None

def test_end_to_end_pipeline():
    service = AnalysisPipelineService(data_mode="mock")
    req = AnalysisRequest(
        idea_text="Graph neural networks and spatial attention for molecular drug target affinity prediction",
        data_mode="mock",
        top_k=3
    )
    res = service.run_analysis(req)
    assert res.status in ["success", "partial_success"]
    assert len(res.papers) > 0
    assert len(res.patents) > 0
    assert len(res.relationships) > 0
    assert len(res.gaps) > 0
    assert len(res.recommendations) > 0
    assert len(res.evidence) > 0

