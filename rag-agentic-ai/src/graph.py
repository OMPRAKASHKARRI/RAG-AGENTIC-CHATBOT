"""LangGraph workflow: START -> retrieve -> generate -> END."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src import config
from src.clients import RagError, get_embeddings, get_llm, get_pinecone, to_rag_error

SYSTEM_PROMPT = f"""You are a strict document-grounded assistant.

Answer the user's question ONLY using the retrieved context.
Do not use outside knowledge.
Do not use your pretrained knowledge to fill missing information.
If the retrieved context does not contain enough information to answer the question, \
reply with exactly this sentence and nothing else: "{config.REFUSAL_MESSAGE}"
Never invent facts.
Never speculate.
Never infer unsupported facts.
The retrieved context is the only authoritative knowledge source."""


class AgentState(TypedDict):
    """State passed between graph nodes."""

    question: str
    context: list[dict[str, Any]]
    answer: str
    score: float


def normalize_score(cosine: float) -> float:
    """Linearly map a cosine score from [SCORE_FLOOR, SCORE_CEILING] onto [0, 1]."""
    span = config.SCORE_CEILING - config.SCORE_FLOOR
    return min(1.0, max(0.0, (cosine - config.SCORE_FLOOR) / span))


def compute_confidence(scores: list[float]) -> float:
    """Blend the best chunk (70%) with the mean of all retrieved chunks (30%)."""
    if not scores:
        return 0.0
    top = normalize_score(max(scores))
    mean = normalize_score(sum(scores) / len(scores))
    return round(0.7 * top + 0.3 * mean, 4)


def retrieve(state: AgentState) -> dict[str, Any]:
    """Embed the question and fetch the top-k chunks (with cosine scores) from Pinecone."""
    try:
        vector = get_embeddings().embed_query(state["question"])
    except Exception as exc:
        raise to_rag_error(exc, "Pinecone Inference (embeddings)") from exc

    try:
        pc = get_pinecone()
        name = config.PINECONE_INDEX_NAME
        if name not in pc.list_indexes().names():
            raise RagError(
                f"Pinecone index '{name}' does not exist. Run `python -m src.ingestion` first.",
                503,
            )
        description = pc.describe_index(name)
        if description.dimension != config.EMBEDDING_DIMENSION:
            raise RagError(
                f"Pinecone index '{name}' has dimension {description.dimension} but the "
                f"embedding model produces {config.EMBEDDING_DIMENSION}. Use a new "
                "PINECONE_INDEX_NAME (or recreate the index) and re-run ingestion.",
                503,
            )
        index = pc.Index(host=description.host)
        response = index.query(vector=vector, top_k=config.TOP_K, include_metadata=True)
    except Exception as exc:
        raise to_rag_error(exc, "Pinecone") from exc

    context: list[dict[str, Any]] = []
    for match in response.matches:
        metadata = match.metadata or {}
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        context.append(
            {
                "content": str(metadata.get("text", "")),
                "page": int(page) if page is not None else None,
                "score": round(float(match.score), 4),
                "source": metadata.get("source"),
                "chunk_index": int(chunk_index) if chunk_index is not None else None,
            }
        )
    return {"context": context}


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Render chunks as a numbered context block for the prompt."""
    return "\n\n".join(
        f"[{number}] (page {chunk['page']})\n{chunk['content']}"
        for number, chunk in enumerate(chunks, start=1)
    )


def generate(state: AgentState) -> dict[str, Any]:
    """Refuse when evidence is weak; otherwise ask Groq for a strictly grounded answer."""
    scores = [chunk["score"] for chunk in state["context"]]
    confidence = compute_confidence(scores)
    relevant = [c for c in state["context"] if c["score"] >= config.RELEVANCE_THRESHOLD]

    if not relevant:  # no useful evidence -> never call the LLM
        return {"answer": config.REFUSAL_MESSAGE, "score": confidence}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{_format_context(relevant)}\n\nQuestion: {state['question']}"
        ),
    ]
    try:
        reply = get_llm().invoke(messages)
    except Exception as exc:
        raise to_rag_error(exc, "Groq") from exc

    answer = reply.content.strip() if isinstance(reply.content, str) else ""
    if not answer:
        raise RagError("Groq returned an empty response.", 502)
    if config.REFUSAL_MESSAGE.lower() in answer.lower():
        confidence = min(confidence, config.REFUSAL_CONFIDENCE_CAP)
    return {"answer": answer, "score": confidence}


def build_graph():
    """Compile the START -> retrieve -> generate -> END graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled graph (built once)."""
    return build_graph()


def ask(question: str) -> AgentState:
    """Run the RAG workflow for a question and return the final state."""
    initial: AgentState = {"question": question, "context": [], "answer": "", "score": 0.0}
    return get_graph().invoke(initial)
