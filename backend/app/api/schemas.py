from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Section 7 Schema Definitions
# ---------------------------------------------------------------------------

class ResearchPaper(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    source_provider: str = "mock"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_version: str = "1.0.0"
    # Research relevance and quality metrics (0 - 100)
    relevance_score: Optional[float] = None
    research_value_score: Optional[float] = None
    relevance_tier: Optional[str] = None  # Highly Recommended, Strongly Relevant, Relevant, Weakly Relevant
    why_relevant: Optional[str] = None

class Patent(BaseModel):
    patent_id: str
    application_id: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    claims: Optional[List[str]] = None
    description: Optional[str] = None
    publication_date: Optional[str] = None
    classifications: List[str] = Field(default_factory=list)
    assignee: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source_provider: str = "mock"
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_version: str = "1.0.0"
    # Patent relevance and prior-art alignment metrics (0 - 100)
    relevance_score: Optional[float] = None
    relevance_tier: Optional[str] = None
    technical_match: Optional[str] = None
    matched_problem: Optional[str] = None
    matched_technology: Optional[str] = None
    matched_application: Optional[str] = None
    why_relevant: Optional[str] = None
    claim_evidence: List[str] = Field(default_factory=list)

class ResearchIdea(BaseModel):
    idea_id: str
    original_text: str
    parsed_problem: str
    domain: str
    proposed_technology: str
    objective: str
    application_area: str
    constraints: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Structured decomposition components
    research_problem: Optional[str] = None
    method_technique: Optional[str] = None
    target_domain: Optional[str] = None
    target_object: Optional[str] = None
    desired_outcome: Optional[str] = None
    salient_keywords: List[str] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)

class EvidenceRecord(BaseModel):
    evidence_id: str
    document_id: str
    document_type: str  # "paper" or "patent"
    passage_text: str
    source_url: Optional[str] = None
    passage_location: Optional[str] = None
    evidence_type: str  # "limitation", "solution", "method", "advantage"
    extraction_method: str = "rule_based_keyword_span"
    confidence: Optional[float] = 1.0

class CrossCorpusRelationship(BaseModel):
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # "paper-to-paper", "patent-to-patent", "paper-to-patent"
    score: float
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    extraction_method: str = "semantic_and_citation"
    confidence: Optional[float] = None
    # Neural relationship probability P(Related | Paper, Patent) in [0, 1]
    # Populated by DualTowerCrossAttentionRelationshipModel.
    ann_relevance_score: Optional[float] = None
    relationship_probability: Optional[float] = None

class CandidateGap(BaseModel):
    gap_id: str
    idea_id: str
    gap_description: str
    gap_type: str  # "TECHNICAL_GAP", "RESEARCH_TO_APPLICATION_GAP", "COMBINATION_GAP", "PATENT_COVERAGE_GAP", "APPLICATION_GAP"
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)
    reasoning: str
    evidence_strength: float = 0.0
    validation_status: str = "candidate"
    # Evidence-grounded structured gap fields
    gap_statement: Optional[str] = None
    why_gap: Optional[str] = None
    existing_research: Optional[str] = None
    existing_patent_landscape: Optional[str] = None
    missing_capability: Optional[str] = None
    research_opportunity: Optional[str] = None
    gap_score: Optional[float] = None  # 0 - 100
    confidence_score: Optional[float] = None  # 0 - 100
    evidence_passages: List[Dict[str, Any]] = Field(default_factory=list)

class RecommendationRecord(BaseModel):
    recommendation_id: str
    idea_id: str
    opportunity_description: str
    problem_addressed: str
    existing_solutions: List[str] = Field(default_factory=list)
    identified_gap: str
    assessment_factors: Dict[str, float] = Field(default_factory=dict)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    potential_advantages: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    actionability: str
    limitations: str
    generation_model_version: str = "ann-1.0"
    prompt_version: str = "v1"

# API Request and Response Schemas
class AnalysisRequest(BaseModel):
    idea_text: str
    domain: Optional[str] = None
    objective: Optional[str] = None
    application_area: Optional[str] = None
    constraints: Optional[List[str]] = None
    top_k: int = 5

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str  # "success", "partial_failure", "failed"
    parsed_idea: ResearchIdea
    papers: List[ResearchPaper]
    patents: List[Patent]
    relationships: List[CrossCorpusRelationship]
    gaps: List[CandidateGap]
    recommendations: List[RecommendationRecord]
    evidence: List[EvidenceRecord]
    scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
