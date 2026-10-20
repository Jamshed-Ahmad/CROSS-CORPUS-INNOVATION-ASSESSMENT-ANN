"""
epo.py
======
Clean ingestion module for the European Patent Office (EPO OPS v3.2) patent literature API.
"""

from app.ingestion.patents import (
    EPOPatentAdapter,
    MockPatentAdapter
)

__all__ = ["EPOPatentAdapter", "MockPatentAdapter"]
