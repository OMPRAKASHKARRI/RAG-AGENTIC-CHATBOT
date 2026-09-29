"""Central configuration: environment variables, model names and RAG tuning knobs.

Secrets are read lazily through :func:`require_env` so that importing this module
never fails (``GET /health`` keeps working) but any code path that actually needs a
credential fails with a clear, secret-free message.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def require_env(name: str) -> str:
    """Return a required environment variable or raise :class:`ConfigError`."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def _env_number(name: str, default: float, cast: type = float) -> float | int:
    """Read a numeric environment override, falling back to ``default``."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number.") from exc


# --- Credentials (read on demand) -------------------------------------------
def groq_api_key() -> str:
    """Groq key - used ONLY for answer generation."""
    return require_env("GROQ_API_KEY")


def pinecone_api_key() -> str:
    """Pinecone key - used ONLY for the vector database."""
    return require_env("PINECONE_API_KEY")


# --- Models ------------------------------------------------------------------
# Pinecone Inference hosted embedding model (uses PINECONE_API_KEY; no local ML runtime).
# Pinecone documents llama-text-embed-v2 as: default dimension 1024, max batch size 96,
# max sequence length 2048 tokens.
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024  # the model's documented default output size
EMBEDDING_BATCH_SIZE = 96  # the model's documented maximum batch size
EMBEDDING_MAX_RETRIES = 3  # retries on HTTP 429 with exponential backoff
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 60

# --- Pinecone ----------------------------------------------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index-api")
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
UPSERT_BATCH_SIZE = 100

# --- Ingestion ---------------------------------------------------------------
PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"
CHUNK_SIZE = int(_env_number("CHUNK_SIZE", 1000, int))
CHUNK_OVERLAP = int(_env_number("CHUNK_OVERLAP", 200, int))

# --- Retrieval and confidence ------------------------------------------------
TOP_K = int(_env_number("TOP_K", 3, int))
# Chunks whose cosine score is below RELEVANCE_THRESHOLD are treated as irrelevant.
# NOTE: these defaults were NOT calibrated for llama-text-embed-v2 on the real eBook;
# validate them with the benchmark queries and tune via .env.
# If NO retrieved chunk reaches it, the question is refused without calling Groq.
RELEVANCE_THRESHOLD = float(_env_number("RELEVANCE_THRESHOLD", 0.30))
# Cosine scores are linearly mapped from [SCORE_FLOOR, SCORE_CEILING] onto [0, 1].
SCORE_FLOOR = float(_env_number("SCORE_FLOOR", 0.20))
SCORE_CEILING = float(_env_number("SCORE_CEILING", 0.75))
# If the LLM itself declares the context insufficient, confidence is capped here.
REFUSAL_CONFIDENCE_CAP = 0.2

# --- API ---------------------------------------------------------------------
MAX_QUERY_CHARS = 2000

REFUSAL_MESSAGE = "I cannot answer this based on the provided document."

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
if SCORE_CEILING <= SCORE_FLOOR:
    raise ConfigError("SCORE_CEILING must be greater than SCORE_FLOOR.")
