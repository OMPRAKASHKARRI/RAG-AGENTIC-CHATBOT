"""Client factories and translation of provider errors into safe API errors."""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import groq
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone
from pinecone.exceptions import (
    ForbiddenException,
    NotFoundException,
    PineconeApiException,
    UnauthorizedException,
)

from src import config

logger = logging.getLogger(__name__)


class RagError(Exception):
    """An error whose message is safe to show to API users (never contains secrets)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_AUTH_ERRORS = (
    groq.AuthenticationError,
    groq.PermissionDeniedError,
    UnauthorizedException,
    ForbiddenException,
)
_RATE_LIMIT_ERRORS = (groq.RateLimitError,)
_CONNECTION_ERRORS = (groq.APIConnectionError,)
_NOT_FOUND_ERRORS = (groq.NotFoundError, NotFoundException)


def to_rag_error(exc: Exception, service: str) -> RagError:
    """Map any exception to a :class:`RagError` with a safe message.

    Provider exception messages can echo request details, so only the exception
    *class name* is ever logged and the message is never forwarded to the user.
    """
    if isinstance(exc, RagError):
        return exc
    if isinstance(exc, config.ConfigError):
        return RagError(str(exc), 503)

    logger.error("%s call failed: %s", service, type(exc).__name__)

    if isinstance(exc, _AUTH_ERRORS):
        return RagError(
            f"{service} rejected the configured credentials. Check the API key in .env.", 502
        )
    if isinstance(exc, _RATE_LIMIT_ERRORS) or (
        isinstance(exc, PineconeApiException) and getattr(exc, "status", None) == 429
    ):
        return RagError(f"{service} rate limit reached. Please retry shortly.", 429)
    if isinstance(exc, _CONNECTION_ERRORS):
        return RagError(f"Could not reach {service}. Check network connectivity and retry.", 503)
    if isinstance(exc, _NOT_FOUND_ERRORS):
        return RagError(f"{service}: the requested model or resource was not found.", 502)
    return RagError(f"{service} request failed ({type(exc).__name__}).", 502)


class PineconeEmbeddings(Embeddings):
    """LangChain-style embeddings backed by Pinecone Inference (no PyTorch, no local model).

    Documents are embedded with ``input_type="passage"`` and queries with
    ``input_type="query"``, as the model requires. Requests are batched to the model's
    maximum batch size and every returned vector is checked against the configured dimension.
    """

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + config.EMBEDDING_BATCH_SIZE]
            result = self._embed_batch(batch, input_type)
            vectors.extend([float(x) for x in item["values"]] for item in result)
        if len(vectors) != len(texts) or any(
            len(vector) != config.EMBEDDING_DIMENSION for vector in vectors
        ):
            raise RagError(
                f"Embedding API returned unexpected vectors (expected {len(texts)} vectors of "
                f"dimension {config.EMBEDDING_DIMENSION}).",
                502,
            )
        return vectors

    def _embed_batch(self, batch: list[str], input_type: str):
        """Call Pinecone Inference, retrying rate-limit (HTTP 429) errors with backoff."""
        for attempt in range(config.EMBEDDING_MAX_RETRIES + 1):
            try:
                return get_pinecone().inference.embed(
                    model=config.EMBEDDING_MODEL,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
            except PineconeApiException as exc:
                if getattr(exc, "status", None) != 429 or attempt == config.EMBEDDING_MAX_RETRIES:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")  # pragma: no cover

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question."""
        return self._embed([text], "query")[0]


@lru_cache(maxsize=1)
def get_embeddings() -> PineconeEmbeddings:
    """Embedding client (Pinecone Inference, authenticated with PINECONE_API_KEY)."""
    return PineconeEmbeddings()


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Groq chat model used for grounded answer generation."""
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.groq_api_key(),
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    """Pinecone control-plane client."""
    return Pinecone(api_key=config.pinecone_api_key())
