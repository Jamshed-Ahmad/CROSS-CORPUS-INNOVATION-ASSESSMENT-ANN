"""
rationale_extraction.py
========================
Fine-grained sentence-level evidence extraction engine.
Identifies explicit research limitations, performance bottlenecks, technical challenges,
and patent claim restrictions with exact document provenance.
"""

import re
import uuid
from typing import List, Dict, Any, Tuple
from app.api.schemas import ResearchPaper, Patent, EvidenceRecord
from app.logging_config import logger

class RationaleExtractor:
    """Extracts structured technical fields and passage-level evidence from papers and patents."""

    PAPER_LIMITATION_KEYWORDS = [
        "limitation", "fail", "struggle", "unable", "lack", "drawback", "noise",
        "sensitivity", "degrades", "missing", "sparse", "exclude", "variance",
        "bottleneck", "unresolved", "challenge", "performance drop", "requires large",
        "generalization", "computational overhead", "latency", "robustness",
        "out-of-distribution", "environmental constraint", "illumination", "benchmark",
        "difficult", "vulnerable", "inability", "restricted to", "unexplored"
    ]

    PATENT_LIMITATION_KEYWORDS = [
        "limited to", "restricted to", "requires", "dependent on", "predetermined threshold",
        "operating constraint", "pre-calculated", "static", "fails when", "rigid-body",
        "offline", "susceptibility", "susceptible", "constraint", "excluding", "omits", "boundary"
    ]

    ADVANTAGE_KEYWORDS = [
        "improve", "speedup", "achieve", "enhance", "outperform", "efficient",
        "accuracy", "reducing", "novel", "superior", "state-of-the-art", "scalable"
    ]

    METHOD_KEYWORDS = [
        "propose", "present", "develop", "using", "architecture", "algorithm",
        "method", "apparatus", "system", "framework", "convolution", "transformer", "network"
    ]

    def extract_from_papers(self, papers: List[ResearchPaper]) -> Tuple[Dict[str, Dict[str, Any]], List[EvidenceRecord]]:
        rationales = {}
        evidence_list = []

        for paper in papers:
            text = f"{paper.title}. {paper.abstract or ''}"
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]

            paper_limitations = []
            paper_advantages = []
            paper_methods = []

            for idx, s in enumerate(sentences):
                s_lower = s.lower()

                # Extract Explicit Limitations
                if any(kw in s_lower for kw in self.PAPER_LIMITATION_KEYWORDS):
                    ev_id = f"EV-PLIM-{uuid.uuid4().hex[:6]}"
                    paper_limitations.append({
                        "text": s,
                        "evidence_id": ev_id,
                        "source": paper.paper_id
                    })
                    evidence_list.append(EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=paper.paper_id,
                        document_type="paper",
                        passage_text=s,
                        source_url=paper.source_url,
                        passage_location=f"sentence_{idx+1}",
                        evidence_type="limitation",
                        extraction_method="sentence_level_limitation_span",
                        confidence=0.92
                    ))

                # Extract Advantages
                if any(kw in s_lower for kw in self.ADVANTAGE_KEYWORDS):
                    ev_id = f"EV-PADV-{uuid.uuid4().hex[:6]}"
                    paper_advantages.append({
                        "text": s,
                        "evidence_id": ev_id,
                        "source": paper.paper_id
                    })
                    evidence_list.append(EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=paper.paper_id,
                        document_type="paper",
                        passage_text=s,
                        source_url=paper.source_url,
                        passage_location=f"sentence_{idx+1}",
                        evidence_type="advantage",
                        extraction_method="sentence_level_advantage_span",
                        confidence=0.88
                    ))

                # Extract Proposed Methods
                if any(kw in s_lower for kw in self.METHOD_KEYWORDS):
                    ev_id = f"EV-PMETH-{uuid.uuid4().hex[:6]}"
                    paper_methods.append({
                        "text": s,
                        "evidence_id": ev_id,
                        "source": paper.paper_id
                    })
                    evidence_list.append(EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=paper.paper_id,
                        document_type="paper",
                        passage_text=s,
                        source_url=paper.source_url,
                        passage_location=f"sentence_{idx+1}",
                        evidence_type="method",
                        extraction_method="sentence_level_method_span",
                        confidence=0.85
                    ))

            rationales[paper.paper_id] = {
                "problem_addressed": paper.title,
                "objective": paper.abstract[:180] if paper.abstract else paper.title,
                "methods": paper_methods,
                "advantages": paper_advantages,
                "limitations": paper_limitations,
                "keywords": paper.keywords
            }

        logger.info(f"RationaleExtractor extracted {len(evidence_list)} evidence records across {len(papers)} papers.")
        return rationales, evidence_list

    def extract_from_patents(self, patents: List[Patent]) -> Tuple[Dict[str, Dict[str, Any]], List[EvidenceRecord]]:
        rationales = {}
        evidence_list = []

        for patent in patents:
            text_corpus = []
            if patent.title:
                text_corpus.append(patent.title)
            if patent.abstract:
                text_corpus.append(patent.abstract)
            if patent.description:
                text_corpus.append(patent.description)
            if patent.claims:
                text_corpus.extend(patent.claims)

            full_text = " ".join(text_corpus)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 15]

            patent_limitations = []
            patent_claims_evidence = []
            patent_solutions = []

            for idx, s in enumerate(sentences):
                s_lower = s.lower()

                # Extract Patent Claim Limitations / Operating Constraints
                if any(kw in s_lower for kw in self.PATENT_LIMITATION_KEYWORDS):
                    ev_id = f"EV-PTLIM-{uuid.uuid4().hex[:6]}"
                    patent_limitations.append({
                        "text": s,
                        "evidence_id": ev_id,
                        "source": patent.patent_id
                    })
                    evidence_list.append(EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=patent.patent_id,
                        document_type="patent",
                        passage_text=s,
                        source_url=patent.source_url,
                        passage_location=f"patent_clause_{idx+1}",
                        evidence_type="limitation",
                        extraction_method="claim_restriction_span",
                        confidence=0.94
                    ))

                # Extract Claim Specifications / Solutions
                if any(kw in s_lower for kw in ["comprising", "apparatus configured to", "method comprising", "configured to execute"]):
                    ev_id = f"EV-PTSOL-{uuid.uuid4().hex[:6]}"
                    patent_solutions.append({
                        "text": s,
                        "evidence_id": ev_id,
                        "source": patent.patent_id
                    })
                    evidence_list.append(EvidenceRecord(
                        evidence_id=ev_id,
                        document_id=patent.patent_id,
                        document_type="patent",
                        passage_text=s,
                        source_url=patent.source_url,
                        passage_location=f"patent_claim_{idx+1}",
                        evidence_type="solution",
                        extraction_method="claimed_mechanism_span",
                        confidence=0.90
                    ))

            rationales[patent.patent_id] = {
                "technical_field": patent.classifications,
                "inventive_solution": patent.title,
                "claim_limitations": patent_limitations,
                "claimed_mechanisms": patent_solutions,
                "assignee": patent.assignee
            }

        logger.info(f"RationaleExtractor extracted {len(evidence_list)} evidence records across {len(patents)} patents.")
        return rationales, evidence_list
