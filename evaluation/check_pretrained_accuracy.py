import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Ensure backend directory is in python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.retrieval.embeddings import EmbeddingGenerator
from app.ingestion.papers import OpenAlexPaperAdapter, MockPaperAdapter
from app.ingestion.patents import EPOPatentAdapter, MockPatentAdapter
from app.ingestion.preprocessing import clean_text

def get_relatability_tier(score: float) -> Tuple[str, str]:
    """Returns a readable label and visual status for a cosine similarity score."""
    if score >= 0.70:
        return "HIGHLY RELATABLE", "[HIGH]"
    elif score >= 0.45:
        return "MODERATELY RELATABLE", "[MEDIUM]"
    elif score >= 0.20:
        return "WEAKLY RELATABLE", "[WEAK]"
    else:
        return "UNRELATED / NOISE", "[FILTERED]"

def evaluate_pretrained_accuracy(
    query: str,
    top_k: int = 5,
    force_live: bool = False,
    verbose: bool = False
):
    print("=" * 80)
    print("        PRETRAINED MODEL ACCURACY & RELATABILITY EVALUATION TOOL        ")
    print("=" * 80)
    print(f"Target Query : \"{query}\"")
    print(f"Top-K Depth  : {top_k}")
    
    # 1. Initialize Pretrained Model
    print("\n[1/4] Loading Pretrained Sentence Transformer...")
    embedder = EmbeddingGenerator()
    print(f"  -> Model Name       : {embedder.model_name}")
    print(f"  -> Vector Dimension : {embedder.dimension} dense dimensions")
    print(f"  -> L2 Normalization : Enabled (Euclidean dot-product = Cosine Similarity)")

    # 2. Ingest API Records
    print("\n[2/4] Fetching Candidate API Data (OpenAlex & EPO OPS)...")
    fixtures_dir = BACKEND_DIR / "data" / "fixtures"
    papers_fixture = fixtures_dir / "papers.json"
    patents_fixture = fixtures_dir / "patents.json"

    # Select adapter based on CLI flag or settings
    use_live = force_live or (settings.DATA_MODE == "live")
    if use_live:
        print("  -> Mode: LIVE APIs (OpenAlex Works API & EPO OPS API)")
        paper_adapter = OpenAlexPaperAdapter(max_results=top_k * 2)
        patent_adapter = EPOPatentAdapter(max_results=top_k * 2)
    else:
        print("  -> Mode: Deterministic Local Fixtures (Offline Mode)")
        paper_adapter = MockPaperAdapter(str(papers_fixture))
        patent_adapter = MockPatentAdapter(str(patents_fixture))

    raw_papers = paper_adapter.fetch_papers(query, limit=top_k * 2)
    raw_patents = patent_adapter.fetch_patents(query, limit=top_k * 2)

    # Fallback to fixtures if live returned empty
    if not raw_papers and papers_fixture.exists():
        print("  -> Fallback: Using local paper fixture records.")
        raw_papers = MockPaperAdapter(str(papers_fixture)).fetch_papers(query, limit=top_k * 2)
    if not raw_patents and patents_fixture.exists():
        print("  -> Fallback: Using local patent fixture records.")
        raw_patents = MockPatentAdapter(str(patents_fixture)).fetch_patents(query, limit=top_k * 2)

    print(f"  -> Retrieved {len(raw_papers)} papers and {len(raw_patents)} patents.")

    # 3. Compute Embeddings & Similarities
    print("\n[3/4] Generating Embeddings & Computing Cosine Relatability...")
    query_emb = embedder.encode([query])[0]

    # Process Papers
    paper_scores = []
    for p in raw_papers:
        text = f"{p.title}. {p.abstract or ''}"
        emb = embedder.encode([text])[0]
        sim = float(np.dot(query_emb, emb))
        paper_scores.append((sim, p))
    paper_scores.sort(key=lambda x: x[0], reverse=True)

    # Process Patents
    patent_scores = []
    for pt in raw_patents:
        claims_str = " ".join(pt.claims or [])
        text = f"{pt.title}. {pt.abstract or ''} {claims_str}"
        emb = embedder.encode([text])[0]
        sim = float(np.dot(query_emb, emb))
        patent_scores.append((sim, pt))
    patent_scores.sort(key=lambda x: x[0], reverse=True)

    # Unrelated Control Baseline (Semantic Discrimination Check)
    unrelated_controls = [
        "Archaeological excavation methods for bronze age ceramic pottery in Mesopotamia.",
        "European municipal solid waste management and recycling quota regulations.",
        "Statistical review of residential property mortgage interest rates in suburban Chicago."
    ]
    control_scores = []
    for ctrl in unrelated_controls:
        emb = embedder.encode([ctrl])[0]
        sim = float(np.dot(query_emb, emb))
        control_scores.append(sim)

    # 4. Display Relatability Reports
    print("\n" + "=" * 80)
    print(f"RESEARCH PAPERS RELATABILITY (Top-{min(top_k, len(paper_scores))})")
    print("=" * 80)
    top_papers = paper_scores[:top_k]
    for idx, (score, p) in enumerate(top_papers, 1):
        tier, badge = get_relatability_tier(score)
        print(f"\n#{idx} {badge} Score: {score:.4f} ({score * 100:.1f}%) | {tier}")
        print(f"    Title : {p.title}")
        print(f"    Year  : {p.publication_year or 'N/A'} | ID: {p.paper_id} | Provider: {p.source_provider}")
        if verbose and p.abstract:
            snippet = clean_text(p.abstract)[:160] + "..."
            print(f"    Abstract: {snippet}")

    print("\n" + "=" * 80)
    print(f"PATENTS PRIOR-ART RELATABILITY (Top-{min(top_k, len(patent_scores))})")
    print("=" * 80)
    top_patents = patent_scores[:top_k]
    for idx, (score, pt) in enumerate(top_patents, 1):
        tier, badge = get_relatability_tier(score)
        print(f"\n#{idx} {badge} Score: {score:.4f} ({score * 100:.1f}%) | {tier}")
        print(f"    Title : {pt.title}")
        print(f"    ID    : {pt.patent_id} | Assignee: {pt.assignee or 'Unknown'}")
        if verbose and pt.abstract:
            snippet = clean_text(pt.abstract)[:160] + "..."
            print(f"    Abstract: {snippet}")

    # 5. Summary Statistics & Accuracy Diagnostics
    print("\n" + "=" * 80)
    print("                       ACCURACY & DISCRIMINATION DIAGNOSTICS                    ")
    print("=" * 80)

    top_paper_sims = [s for s, _ in top_papers]
    top_patent_sims = [s for s, _ in top_patents]
    all_top_sims = top_paper_sims + top_patent_sims

    mean_relatability = np.mean(all_top_sims) if all_top_sims else 0.0
    top1_paper_sim = top_paper_sims[0] if top_paper_sims else 0.0
    top1_patent_sim = top_patent_sims[0] if top_patent_sims else 0.0
    mean_control_sim = np.mean(control_scores)
    separation_margin = mean_relatability - mean_control_sim

    print(f"  * Average Relatability (Top-{top_k} Candidates) : {mean_relatability * 100:.2f}% (Score: {mean_relatability:.4f})")
    print(f"  * Top-1 Paper Match Relatability            : {top1_paper_sim * 100:.2f}% (Score: {top1_paper_sim:.4f})")
    print(f"  * Top-1 Patent Match Relatability           : {top1_patent_sim * 100:.2f}% (Score: {top1_patent_sim:.4f})")
    print(f"  * Unrelated Control Average (Baseline Noise): {mean_control_sim * 100:.2f}% (Score: {mean_control_sim:.4f})")
    print(f"  * Semantic Discrimination Margin (+/-)      : +{separation_margin * 100:.2f}% difference")
    
    threshold = settings.RELEVANCE_THRESHOLD
    passing_count = sum(1 for s in all_top_sims if s >= threshold)
    total_count = len(all_top_sims)
    pass_pct = (passing_count / total_count * 100) if total_count > 0 else 0.0
    print(f"  * System Threshold Pass Rate (>= {threshold})    : {passing_count}/{total_count} ({pass_pct:.1f}%)")

    print("\n--- MODEL ACCURACY CONCLUSION ---")
    if separation_margin >= 0.35 and mean_relatability >= 0.50:
        print("  [SUCCESS] HIGH DISCRIMINATIVE ACCURACY:")
        print(f"  The pretrained model '{embedder.model_name}' successfully isolates relevant")
        print(f"  scientific and patent records from the API, maintaining a strong +{separation_margin*100:.1f}%")
        print("  margin of separation above unrelated baseline noise.")
    elif mean_relatability >= 0.30:
        print("  [ACCEPTABLE] MODERATE RELATABILITY:")
        print("  The retrieved records are contextually aligned with moderate semantic density.")
    else:
        print("  [LOW] QUERY / CORPUS MISMATCH:")
        print("  The retrieved documents have weak semantic alignment with the given query.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Pretrained SentenceTransformer Accuracy & API Relatability")
    parser.add_argument(
        "--query",
        type=str,
        default="Using graph neural networks and dynamic spatial attention to predict drug-target affinity",
        help="Research idea query text to evaluate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to report"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force querying live OpenAlex and EPO OPS APIs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display text snippets for each paper and patent"
    )
    args = parser.parse_args()

    evaluate_pretrained_accuracy(
        query=args.query,
        top_k=args.top_k,
        force_live=args.live,
        verbose=args.verbose
    )
