"""FastAPI application exposing the RAG chatbot."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src import config
from src.clients import RagError, to_rag_error
from src.graph import ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="Answers questions strictly from the Agentic AI eBook "
    "(Pinecone-hosted embeddings + Pinecone + LangGraph + Groq).",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """Body of ``POST /chat``."""

    query: str = Field(..., min_length=1, max_length=config.MAX_QUERY_CHARS)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject whitespace-only queries and trim the rest."""
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace only")
        return value


class RetrievedChunk(BaseModel):
    """One retrieved chunk with its Pinecone cosine similarity."""

    content: str
    page: int | None = None
    score: float
    source: str | None = None
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    """Body returned by ``POST /chat``."""

    final_answer: str
    retrieved_context: list[RetrievedChunk]
    confidence_score: float = Field(..., ge=0.0, le=1.0)


@app.exception_handler(RagError)
async def rag_error_handler(_: Request, exc: RagError) -> JSONResponse:
    """Return safe provider/config errors without internals."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Never leak stack traces for unexpected failures."""
    logger.error("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
def root() -> dict[str, str]:
    """Basic API information."""
    return {
        "name": app.title,
        "version": app.version,
        "chat": "POST /chat",
        "health": "GET /health",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe (does not call external services)."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using only the ingested eBook."""
    try:
        state = ask(request.query)
    except Exception as exc:
        raise to_rag_error(exc, "RAG pipeline") from exc
    return ChatResponse(
        final_answer=state["answer"],
        retrieved_context=[RetrievedChunk(**chunk) for chunk in state["context"]],
        confidence_score=state["score"],
    )
