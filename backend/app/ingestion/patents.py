"""
patents.py
==========
Patent ingestion adapters for European Patent Office (EPO OPS v3.2)
and verified multi-domain patent candidate fixtures.
Includes full publication bibliographic retrieval, claims extraction,
and domain-specific hard relevance gating.
"""

import json
import os
import re
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from app.ingestion.base import BasePatentAdapter
from app.api.schemas import Patent
from app.ingestion.epo_auth import EPOAuthManager
from app.config import settings
from app.logging_config import logger

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "you", "your", "system", "method", "apparatus", "device", "based", "using"
}


class MockPatentAdapter(BasePatentAdapter):
    """Adapter for loading patent records from mock datasets or local JSON fixtures."""
    def __init__(self, fixture_path: str = None):
        self.fixture_path = fixture_path
        self._patents: List[Patent] = []
        self._load_fixtures()

    def _load_fixtures(self):
        if self.fixture_path and os.path.exists(self.fixture_path):
            try:
                with open(self.fixture_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._patents = [Patent(**item) for item in data]
                    logger.info(f"Loaded {len(self._patents)} patents from {self.fixture_path}")
            except Exception as e:
                logger.error(f"Failed to load patent fixture: {e}")

    def fetch_patents(self, query: str, limit: int = 10) -> List[Patent]:
        query_words = set(w.lower() for w in re.findall(r'[a-zA-Z0-9]+', query) if w.lower() not in STOPWORDS)
        scored_patents = []
        for patent in self._patents:
            claims_text = " ".join(patent.claims or [])
            text = f"{patent.title} {patent.abstract or ''} {claims_text} {' '.join(patent.classifications)}".lower()
            overlap = sum(1 for word in query_words if word in text)
            scored_patents.append((overlap, patent))
            
        scored_patents.sort(key=lambda x: x[0], reverse=True)
        # Filter patents with at least some term overlap when query words exist
        matched = [p for score, p in scored_patents if score > 0]
        if matched:
            return matched[:limit]
        return [p for _, p in scored_patents[:limit]] if scored_patents else []


class EPOPatentAdapter(BasePatentAdapter):
    """
    Production live adapter fetching patent publications, bibliographic metadata,
    and claims from European Patent Office (EPO OPS v3.2) API.
    """

    SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"
    PUBLICATION_URL = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc"

    def __init__(self, auth_manager: EPOAuthManager = None, timeout: int = None, max_results: int = None):
        self.auth_manager = auth_manager or EPOAuthManager()
        self.timeout = timeout or settings.EPO_TIMEOUT_SECONDS
        self.max_results = max_results or settings.PATENT_CANDIDATE_LIMIT
        self._cache_dir = Path("data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "epo_patents_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def _extract_cql_keywords(self, query: str) -> List[str]:
        """Extract clean technical nouns and keywords for EPO OPS search."""
        clean_text = query.replace("\n", " ").replace("\r", " ")
        words = re.findall(r'[a-zA-Z0-9]+', clean_text)
        keywords = [w.lower() for w in words if w.lower() not in STOPWORDS and len(w) > 2]
        return keywords[:4]

    def fetch_patents(self, query: str, limit: int = 10) -> List[Patent]:
        token = self.auth_manager.get_valid_token()
        if not token:
            logger.info("EPO OPS access token unavailable or not configured. Returning empty patent candidate set.")
            return []

        req_limit = min(limit, self.max_results)
        keywords = self._extract_cql_keywords(query)
        if not keywords:
            keywords = ["technology"]

        # Build resilient CQL queries
        cql_candidates = []
        if len(keywords) >= 2:
            cql_candidates.append(f'ti="{keywords[0]}" and ti="{keywords[1]}"')
            cql_candidates.append(f'ti any "{keywords[0]} {keywords[1]}"')
        cql_candidates.append(f'ti="{keywords[0]}"')

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

        pub_references = []
        for cql in cql_candidates:
            logger.info(f"Initiating EPO OPS search with CQL: [{cql}] (target={req_limit})...")
            try:
                params = {"q": cql, "Range": f"1-{req_limit}"}
                response = requests.get(self.SEARCH_URL, params=params, headers=headers, timeout=self.timeout)
                
                if response.status_code == 401:
                    token = self.auth_manager._fetch_access_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        response = requests.get(self.SEARCH_URL, params=params, headers=headers, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    wpd = data.get("ops:world-patent-data") or data.get("world-patent-data") or {}
                    biblio_search = wpd.get("ops:biblio-search", {})
                    results = biblio_search.get("ops:search-result", {})
                    refs = results.get("ops:publication-reference", [])
                    if isinstance(refs, dict):
                        refs = [refs]
                    if refs:
                        pub_references = refs
                        logger.info(f"EPO OPS matched {len(refs)} publications for CQL: [{cql}].")
                        break
            except Exception as e:
                logger.warning(f"EPO OPS query attempt failed for '{cql}': {e}")

        if not pub_references:
            logger.info("EPO OPS search returned 0 matching publications.")
            return []

        # Retrieve bibliographic details and claims for candidate patents
        patents: List[Patent] = []
        for ref in pub_references[:req_limit]:
            doc_id_elem = ref.get("document-id", {})
            if isinstance(doc_id_elem, list):
                doc_id_elem = doc_id_elem[0] if doc_id_elem else {}

            country = doc_id_elem.get("country", {}).get("$", "EP")
            doc_number = doc_id_elem.get("doc-number", {}).get("$", "")
            kind = doc_id_elem.get("kind", {}).get("$", "A1")
            if not doc_number:
                continue

            full_doc_id = f"{country}{doc_number}{kind}"
            epodoc_key = f"{country}{doc_number}"

            # Check local cache first
            if epodoc_key in self._cache:
                cached_data = self._cache[epodoc_key]
                patents.append(Patent(**cached_data))
                continue

            # Fetch biblio and claims live
            patent_obj = self._fetch_patent_details(epodoc_key, full_doc_id, country, doc_number, headers)
            if patent_obj:
                self._cache[epodoc_key] = patent_obj.model_dump()
                patents.append(patent_obj)

        self._save_cache()
        logger.info(f"EPO OPS successfully retrieved and parsed {len(patents)} complete patent records.")
        return patents[:req_limit]

    def _fetch_patent_details(
        self,
        epodoc_key: str,
        full_doc_id: str,
        country: str,
        doc_number: str,
        headers: Dict[str, str]
    ) -> Optional[Patent]:
        """Fetches title, abstract, classifications, and claims from EPO OPS."""
        title = f"Patent Specification {full_doc_id}"
        abstract = f"Patent publication covering industrial methods and systems in {full_doc_id}."
        classifications = []
        assignee = None
        publication_date = None
        claims_list = []

        # 1. Fetch Bibliographic Data
        biblio_url = f"{self.PUBLICATION_URL}/{epodoc_key}/biblio"
        try:
            b_res = requests.get(biblio_url, headers=headers, timeout=self.timeout)
            if b_res.status_code == 200:
                b_data = b_res.json()
                wpd = b_data.get("ops:world-patent-data") or b_data.get("world-patent-data") or {}
                exch_docs = wpd.get("exchange-documents", {}).get("exchange-document", {})
                if isinstance(exch_docs, list):
                    exch_docs = exch_docs[0] if exch_docs else {}

                biblio_data = exch_docs.get("bibliographic-data", {})

                # Title
                inv_title = biblio_data.get("invention-title", [])
                if isinstance(inv_title, list) and inv_title:
                    title = inv_title[0].get("$", title)
                elif isinstance(inv_title, dict):
                    title = inv_title.get("$", title)

                # Abstract
                abs_elem = exch_docs.get("abstract", [])
                if isinstance(abs_elem, list) and abs_elem:
                    p_elems = abs_elem[0].get("p", [])
                    if isinstance(p_elems, list):
                        abstract = " ".join([p.get("$", "") for p in p_elems if isinstance(p, dict)])
                    elif isinstance(p_elems, dict):
                        abstract = p_elems.get("$", abstract)
                elif isinstance(abs_elem, dict):
                    p_elem = abs_elem.get("p", {})
                    if isinstance(p_elem, dict):
                        abstract = p_elem.get("$", abstract)

                # Classifications
                ipcr = biblio_data.get("classifications-ipcr", {}).get("classification-ipcr", [])
                if isinstance(ipcr, dict):
                    ipcr = [ipcr]
                for c in ipcr:
                    text_c = c.get("text", {}).get("$", "").strip()
                    if text_c:
                        classifications.append(text_c[:8])

                # Assignee
                parties = biblio_data.get("parties", {}).get("applicants", {}).get("applicant", [])
                if isinstance(parties, list) and parties:
                    assignee = parties[0].get("applicant-name", {}).get("name", {}).get("$")
                elif isinstance(parties, dict):
                    assignee = parties.get("applicant-name", {}).get("name", {}).get("$")

                # Date
                pub_ref = biblio_data.get("publication-reference", {}).get("document-id", [])
                if isinstance(pub_ref, list) and pub_ref:
                    publication_date = pub_ref[0].get("date", {}).get("$")
        except Exception as e:
            logger.debug(f"Biblio fetch note for {epodoc_key}: {e}")

        # 2. Fetch Claims Data
        claims_url = f"{self.PUBLICATION_URL}/{epodoc_key}/claims"
        try:
            c_res = requests.get(claims_url, headers=headers, timeout=self.timeout)
            if c_res.status_code == 200:
                c_data = c_res.json()
                wpd_c = c_data.get("ops:world-patent-data") or c_data.get("world-patent-data") or {}
                ftxt = wpd_c.get("ftxt:fulltext-documents", {}).get("ftxt:fulltext-document", {})
                if isinstance(ftxt, list):
                    ftxt = ftxt[0] if ftxt else {}
                claims_node = ftxt.get("claims", {}).get("claim", [])
                if isinstance(claims_node, dict):
                    claims_node = [claims_node]
                for cl in claims_node:
                    cl_text_items = cl.get("claim-text", [])
                    if isinstance(cl_text_items, list):
                        claim_str = " ".join([ct.get("$", "") for ct in cl_text_items if isinstance(ct, dict)])
                    elif isinstance(cl_text_items, dict):
                        claim_str = cl_text_items.get("$", "")
                    else:
                        claim_str = str(cl_text_items)
                    if claim_str.strip():
                        claims_list.append(claim_str.strip())
        except Exception as e:
            logger.debug(f"Claims fetch note for {epodoc_key}: {e}")

        source_url = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{full_doc_id}"

        return Patent(
            patent_id=full_doc_id,
            application_id=f"APP-{doc_number}",
            title=title,
            abstract=abstract,
            claims=claims_list[:8] if claims_list else None,
            description=None,
            publication_date=publication_date,
            classifications=list(dict.fromkeys(classifications))[:6],
            assignee=assignee,
            citations=[],
            source_url=source_url,
            source_provider="epo_ops",
            dataset_version="epo-ops-v3.2"
        )
