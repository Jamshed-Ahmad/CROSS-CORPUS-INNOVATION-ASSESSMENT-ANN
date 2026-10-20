"""
query_parser.py
===============
Advanced Query Decomposition and Structured Research Representation Engine.
Decomposes user research queries into fine-grained conceptual, technical,
and application components to guide field-aware hybrid retrieval and reranking.
"""

import uuid
import re
from typing import List, Dict, Set, Tuple, Optional
from app.api.schemas import AnalysisRequest, ResearchIdea
from app.logging_config import logger

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "you", "your", "yours", "yourself", "yourselves"
}

DOMAIN_PATTERNS: List[Tuple[List[str], str]] = [
    (["drug", "binding", "affinity", "ligand", "protein", "receptor", "docking", "molecular", "pharmaceutical"], "Biomedicine & Molecular Discovery"),
    (["crop", "plant", "agriculture", "grape", "leaf", "disease", "foliage", "farm", "soil", "harvest", "pest"], "Agriculture & Plant Pathology"),
    (["heart", "cardiac", "cardiovascular", "ecg", "arrhythmia", "infarction", "coronary"], "Cardiology & Healthcare"),
    (["house", "housing", "real estate", "property", "rent", "valuation", "pricing", "mortgage"], "Real Estate & Econometrics"),
    (["battery", "electrolyte", "solid-state", "lithium", "anode", "cathode", "energy storage"], "Materials Science & Energy Storage"),
    (["cancer", "tumor", "oncology", "biopsy", "histopathology", "mri", "ct scan"], "Oncology & Medical Imaging"),
    (["fpga", "hardware", "edge device", "microcontroller", "embedded", "low-power", "systolic array", "npu", "edge computing"], "Computer Science & Edge Computing"),
    (["quantum", "qubit", "eigensolver", "superconducting", "quantum circuit"], "Quantum Computing & Physics"),
    (["climate", "carbon", "emission", "greenhouse", "solar", "renewable", "wind turbine"], "Environmental Science & Clean Energy"),
    (["finance", "fraud", "credit", "stock", "trading", "cryptocurrency", "blockchain"], "Finance & FinTech"),
    (["robot", "autonomous", "vehicle", "lidar", "slam", "drone", "uav", "manipulator"], "Robotics & Autonomous Systems")
]

TECH_PATTERNS: List[Tuple[List[str], str]] = [
    (["hyperspectral", "multispectral", "spectral imaging"], "Hyperspectral / Multispectral Imaging"),
    (["transformer", "self-attention", "attention mechanism"], "Transformer Self-Attention Networks"),
    (["graph neural network", "gnn", "graph convolutional", "gcn"], "Graph Neural Networks"),
    (["convolutional neural network", "cnn", "resnet", "yolo"], "Deep Convolutional Neural Networks"),
    (["federated learning", "privacy-preserving", "differential privacy"], "Federated & Privacy-Preserving Learning"),
    (["reinforcement learning", "rl", "policy gradient", "q-learning"], "Deep Reinforcement Learning"),
    (["solid-state electrolyte", "composite electrolyte", "garnet"], "Solid-State Electrolyte Interface"),
    (["quantization", "pruning", "knowledge distillation"], "Model Compression & Quantization"),
    (["multi-modal", "multimodal", "cross-modal"], "Multi-Modal Feature Fusion")
]


class QueryParser:
    """
    Decomposes user research proposals and questions into structured,
    searchable, and field-aware research components.
    """

    def parse_idea(self, request: AnalysisRequest) -> ResearchIdea:
        text = request.idea_text.strip().replace("\r", " ").replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)

        # 1. Extract clean tokens and technical n-grams
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9\-_]+', text)]
        content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        
        # 2. Extract technical terms (unigrams and bigrams)
        technical_terms = []
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if words[i] not in STOPWORDS or words[i+1] not in STOPWORDS:
                technical_terms.append(pair)

        # 3. Dynamic Domain Extraction
        inferred_domain = request.domain
        if not inferred_domain or inferred_domain.lower() in ["string", "none", "unknown", ""]:
            for keywords, d_name in DOMAIN_PATTERNS:
                if any(kw in text.lower() for kw in keywords):
                    inferred_domain = d_name
                    break
            if not inferred_domain:
                inferred_domain = "Applied Artificial Intelligence & Engineering"

        # 4. Proposed Technology Extraction
        inferred_tech = getattr(request, 'proposed_technology', None)
        if not inferred_tech or inferred_tech.lower() in ["string", "none", "unknown", ""]:
            for keywords, t_name in TECH_PATTERNS:
                if any(kw in text.lower() for kw in keywords):
                    inferred_tech = t_name
                    break
            if not inferred_tech:
                # Fallback: extract salient method noun-phrases
                matched_phrases = re.findall(r'(?:using|via|based on|with|through)\s+([a-zA-Z0-9\s\-]+)', text, flags=re.IGNORECASE)
                if matched_phrases:
                    inferred_tech = matched_phrases[0].strip()[:60]
                else:
                    salient = content_words[:3]
                    inferred_tech = " ".join(salient).title() if salient else "Advanced Computational Method"

        # 5. Method / Technique Extraction
        method_candidates = []
        for m_kw in ["deep learning", "machine learning", "neural network", "attention", "quantization", "pruning", 
                     "clustering", "regression", "svm", "random forest", "spectroscopy", "imaging", "graph network"]:
            if m_kw in text.lower():
                method_candidates.append(m_kw.title())
        method_technique = ", ".join(method_candidates[:2]) if method_candidates else inferred_tech

        # 6. Research Problem Extraction
        # Look for explicit problem clauses or targets
        prob_matches = re.findall(r'(?:for|aimed at|targeting|predicting|detecting|optimizing|solving)\s+([a-zA-Z0-9\s\-]+)', text, flags=re.IGNORECASE)
        if prob_matches:
            parsed_problem = prob_matches[0].strip()[:100]
        else:
            parsed_problem = text[:100]

        # 7. Application Area & Target Object
        app_area = request.application_area
        if not app_area or app_area.lower() in ["string", "none", "unknown", ""]:
            if "disease" in text.lower() or "health" in text.lower() or "diagnosis" in text.lower():
                app_area = "Automated Diagnostic & Health Monitoring"
                target_obj = "Biological / Clinical Specimens"
            elif "edge" in text.lower() or "embedded" in text.lower() or "iot" in text.lower():
                app_area = "Resource-Constrained Edge Platforms"
                target_obj = "Low-Power Processing Units"
            elif "price" in text.lower() or "market" in text.lower() or "valuation" in text.lower():
                app_area = "Quantitative Market Valuation & Asset Pricing"
                target_obj = "Market Property & Asset Records"
            elif "crop" in text.lower() or "grape" in text.lower() or "agriculture" in text.lower():
                app_area = "Precision Agriculture & Crop Protection"
                target_obj = "Foliage & Crop Canopy"
            elif "battery" in text.lower() or "lithium" in text.lower() or "energy" in text.lower():
                app_area = "High-Energy Density Battery Systems"
                target_obj = "Electrode-Electrolyte Interfaces"
            else:
                app_area = f"{inferred_domain} Practical Deployment"
                target_obj = "Target System & Domain Data"
        else:
            target_obj = "Target Domain Artifacts"

        # 8. Constraints & Desired Outcome
        constraints = request.constraints or []
        if not constraints or all(c.lower() in ["string", "none"] for c in constraints):
            inferred_constraints = []
            if any(k in text.lower() for k in ["edge", "low-power", "energy", "embedded", "mobile", "battery"]):
                inferred_constraints.append("Resource & energy-constrained compute budget")
            if any(k in text.lower() for k in ["early", "real-time", "latency", "sub-second"]):
                inferred_constraints.append("Strict low-latency execution")
            if any(k in text.lower() for k in ["heterogeneous", "illumination", "noise", "robust", "variation"]):
                inferred_constraints.append("Robustness to environmental / acquisition noise")
            if any(k in text.lower() for k in ["privacy", "hospital", "federated", "clinical"]):
                inferred_constraints.append("Strict decentralized data privacy compliance")
            if not inferred_constraints:
                inferred_constraints = ["High precision and reproducibility under varying empirical conditions"]
            constraints = inferred_constraints

        objective = request.objective
        if not objective or objective.lower() in ["string", "none", "unknown", ""]:
            objective = f"Develop {inferred_tech} to resolve {parsed_problem} within {app_area}."

        desired_outcome = f"Accurate, robust, and scalable resolution of {parsed_problem}"

        idea_id = f"IDEA-{uuid.uuid4().hex[:6]}"

        logger.info(
            f"QueryParser decomposed query into: Domain='{inferred_domain}', "
            f"Tech='{inferred_tech}', Problem='{parsed_problem}', AppArea='{app_area}'"
        )

        return ResearchIdea(
            idea_id=idea_id,
            original_text=text,
            parsed_problem=parsed_problem,
            domain=inferred_domain,
            proposed_technology=inferred_tech,
            objective=objective,
            application_area=app_area,
            constraints=constraints,
            research_problem=parsed_problem,
            method_technique=method_technique,
            target_domain=inferred_domain,
            target_object=target_obj,
            desired_outcome=desired_outcome,
            salient_keywords=content_words[:8],
            technical_terms=technical_terms[:6]
        )
