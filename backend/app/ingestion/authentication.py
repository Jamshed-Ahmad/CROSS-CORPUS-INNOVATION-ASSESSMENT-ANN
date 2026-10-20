"""
authentication.py
=================
OAuth 2.0 authentication and in-memory token manager for EPO OPS API.
"""

from app.ingestion.epo_auth import EPOAuthManager

__all__ = ["EPOAuthManager"]
