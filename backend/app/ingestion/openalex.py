"""
openalex.py
===========
Clean ingestion module for the OpenAlex scholarly research literature API.
"""

from app.ingestion.papers import (
    OpenAlexPaperAdapter,
    MockPaperAdapter,
    reconstruct_openalex_abstract
)

__all__ = ["OpenAlexPaperAdapter", "MockPaperAdapter", "reconstruct_openalex_abstract"]
