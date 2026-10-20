import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is always loaded from backend root directory regardless of current working directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

def sanitize_credential(val: str) -> str:
    """Sanitizes credentials by stripping whitespace and surrounding quotes, and ignoring placeholder text."""
    if not val:
        return ""
    clean = val.strip().strip("'\"").strip()
    if clean.lower().startswith("your_") and clean.lower().endswith("_here"):
        return ""
    if clean.startswith("<") and clean.endswith(">"):
        return ""
    return clean

class Settings(BaseSettings):
    PROJECT_NAME: str = "Cross-Corpus Innovation Assessment System"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Environment & Operational Mode
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DATA_MODE: str = os.getenv("DATA_MODE", "mock").lower()  # "live" or "mock"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./innovation_system.db")
    
    # OpenAlex API Configuration (Papers)
    OPENALEX_API_KEY: str = os.getenv("OPENALEX_API_KEY", "")
    OPENALEX_MAX_RESULTS: int = int(os.getenv("OPENALEX_MAX_RESULTS", "50"))
    OPENALEX_TIMEOUT_SECONDS: int = int(os.getenv("OPENALEX_TIMEOUT_SECONDS", "30"))

    # EPO OPS API Configuration (Patents)
    EPO_CONSUMER_KEY: str = os.getenv("EPO_CONSUMER_KEY", "")
    EPO_CONSUMER_SECRET: str = os.getenv("EPO_CONSUMER_SECRET", "")
    EPO_MAX_RESULTS: int = int(os.getenv("EPO_MAX_RESULTS", "50"))
    EPO_TIMEOUT_SECONDS: int = int(os.getenv("EPO_TIMEOUT_SECONDS", "30"))

    # Retrieval Candidate Limits & Thresholds
    PAPER_CANDIDATE_LIMIT: int = int(os.getenv("PAPER_CANDIDATE_LIMIT", "50"))
    PATENT_CANDIDATE_LIMIT: int = int(os.getenv("PATENT_CANDIDATE_LIMIT", "50"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "10"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DIMENSION: int = 384
    
    # Neural Model Configurations (PyTorch)
    RELATIONSHIP_MODEL_PATH: str = os.getenv("RELATIONSHIP_MODEL_PATH", "models/relationship_model/relationship_model.pt")
    GAP_MODEL_PATH: str = os.getenv("GAP_MODEL_PATH", "models/gap_detection_model/gap_detection_model.pt")
    
    # Ranking Weights for Multidimensional Innovation Assessment (default rubric)
    WEIGHT_NOVELTY: float = 0.25
    WEIGHT_RELEVANCE: float = 0.20
    WEIGHT_GAP: float = 0.20
    WEIGHT_EVIDENCE: float = 0.15
    WEIGHT_FEASIBILITY: float = 0.10
    WEIGHT_DIFFERENTIATION: float = 0.10

    @property
    def clean_openalex_api_key(self) -> str:
        return sanitize_credential(self.OPENALEX_API_KEY)

    @property
    def clean_epo_consumer_key(self) -> str:
        return sanitize_credential(self.EPO_CONSUMER_KEY)

    @property
    def clean_epo_consumer_secret(self) -> str:
        return sanitize_credential(self.EPO_CONSUMER_SECRET)

    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore")

settings = Settings()

