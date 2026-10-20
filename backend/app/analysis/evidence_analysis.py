"""
evidence_analysis.py
====================
Evidence analysis and passage-level extraction engine.
Wraps RationaleExtractor for limitations, advantages, methods, and patent claims.
"""

from app.analysis.rationale_extraction import RationaleExtractor

__all__ = ["RationaleExtractor"]
