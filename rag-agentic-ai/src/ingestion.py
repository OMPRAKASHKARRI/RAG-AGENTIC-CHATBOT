"""PDF ingestion: extract -> chunk -> embed (Pinecone Inference) -> upsert (Pinecone).

Run with:  python -m src.ingestion [--pdf PATH] [--reset]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from pinecone.db_data import Index
from pinecone.exceptions import NotFoundException
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from src import config
from src.clients import RagError, get_embeddings, get_pinecone, to_rag_error

logger = logging.getLogger(__name__)


def load_pages(pdf_path: Path) -> list[Document]:
    """Extract text from every non-empty page of the PDF (1-based page numbers)."""
    if not pdf_path.is_file():
        raise RagError(
            f"PDF not found at '{pdf_path}'. Place the Agentic AI eBook at "
            "data/Ebook-Agentic-AI.pdf and run ingestion again.",
            503,
        )
    try:
        reader = PdfReader(str(pdf_path))
        pages = [
            Document(
                page_content=(page.extract_text() or "").strip(),
                metadata={"source": pdf_path.name, "page": number},
            )
            for number, page in enumerate(reader.pages, start=1)
        ]
    except PyPdfError as exc:
        raise RagError(f"Could not read the PDF ({type(exc).__name__}).", 422) from exc
    non_empty = [page for page in pages if page.page_content]
    if not non_empty:
        raise RagError("No extractable text found in the PDF (is it a scanned document?).", 422)
    return non_empty


def split_pages(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks and attach chunk metadata + stable IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    per_page_counter: dict[int, int] = {}
    for chunk in chunks:
        page = chunk.metadata["page"]
        index = per_page_counter.get(page, 0)
        per_page_counter[page] = index + 1
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", Path(chunk.metadata["source"]).stem).strip("-")
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{slug}-p{page}-c{index}-{digest}"
    return chunks


def ensure_index() -> Index:
    """Create the Pinecone index if missing (1024-dim, cosine) and return a handle."""
    try:
        pc = get_pinecone()
        name = config.PINECONE_INDEX_NAME
        if name not in pc.list_indexes().names():
            logger.info("Creating Pinecone index '%s'", name)
            pc.create_index(
                name=name,
                dimension=config.EMBEDDING_DIMENSION,
                metric=config.PINECONE_METRIC,
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
        description = pc.describe_index(name)
        if description.dimension != config.EMBEDDING_DIMENSION or description.metric != "cosine":
            raise RagError(
                f"Existing index '{name}' has dimension {description.dimension} / metric "
                f"{description.metric}, but {config.EMBEDDING_DIMENSION} / cosine is required. "
                "Delete and recreate that index, or set a new PINECONE_INDEX_NAME in .env.",
                503,
            )
        return pc.Index(host=description.host)
    except Exception as exc:
        raise to_rag_error(exc, "Pinecone") from exc


def _missing_ids(index: Index, ids: list[str]) -> set[str]:
    """Return the IDs that are not yet stored in the index (avoids duplicate ingestion)."""
    existing: set[str] = set()
    for start in range(0, len(ids), config.UPSERT_BATCH_SIZE):
        batch = ids[start : start + config.UPSERT_BATCH_SIZE]
        existing.update(index.fetch(ids=batch).vectors.keys())
    return set(ids) - existing


def ingest(pdf_path: Path = config.PDF_PATH, reset: bool = False) -> dict[str, int]:
    """Run the full ingestion pipeline and return simple statistics."""
    pages = load_pages(pdf_path)
    chunks = split_pages(pages)
    index = ensure_index()

    try:
        if reset:
            try:
                index.delete(delete_all=True)
            except NotFoundException:
                pass  # empty index/namespace: nothing to delete
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        missing = _missing_ids(index, ids)
        todo = [c for c in chunks if c.metadata["chunk_id"] in missing]
        embeddings = get_embeddings()

        for start in range(0, len(todo), config.UPSERT_BATCH_SIZE):
            batch = todo[start : start + config.UPSERT_BATCH_SIZE]
            try:
                vectors = embeddings.embed_documents([c.page_content for c in batch])
            except Exception as exc:
                raise to_rag_error(exc, "Pinecone Inference (embeddings)") from exc
            if any(len(vector) != config.EMBEDDING_DIMENSION for vector in vectors):
                raise RagError("Embedding dimension does not match the configured index.", 500)
            index.upsert(
                vectors=[
                    {
                        "id": chunk.metadata["chunk_id"],
                        "values": vector,
                        "metadata": {**chunk.metadata, "text": chunk.page_content},
                    }
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
            )
    except RagError:
        raise
    except Exception as exc:
        raise to_rag_error(exc, "Pinecone") from exc

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "upserted": len(todo),
        "skipped_existing": len(chunks) - len(todo),
    }


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", type=Path, default=config.PDF_PATH, help="Path to the PDF.")
    parser.add_argument(
        "--reset", action="store_true", help="Delete all vectors in the index before ingesting."
    )
    args = parser.parse_args()
    try:
        stats = ingest(args.pdf, reset=args.reset)
    except (RagError, config.ConfigError) as exc:
        print(f"Ingestion failed: {getattr(exc, 'message', str(exc))}", file=sys.stderr)
        return 1
    print(
        f"Ingestion complete: {stats['pages']} pages, {stats['chunks']} chunks, "
        f"{stats['upserted']} upserted, {stats['skipped_existing']} already present."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
