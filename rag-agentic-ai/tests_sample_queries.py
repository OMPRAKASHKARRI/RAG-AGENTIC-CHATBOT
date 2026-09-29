"""Benchmark script: sends the sample queries to a running FastAPI server.

Usage:
    uvicorn app:app            # in one terminal (after running ingestion)
    python tests_sample_queries.py [--url http://127.0.0.1:8000]

Exit code is non-zero if any check fails. Grounded questions PASS when the API returns a
non-refusal answer with retrieved context; the negative question PASSES only on a refusal.
"""

from __future__ import annotations

import argparse
import sys

import requests

from src.config import REFUSAL_MESSAGE

# (question, must_be_refused)
QUERIES: list[tuple[str, bool]] = [
    ("What is Agentic AI according to the eBook?", False),
    ("How do AI agents differ from traditional automation systems?", False),
    ("What are the core components of an Agentic Architecture?", False),
    ("What role does memory play in Agentic AI workflows?", False),
    ("Who won the 2022 FIFA World Cup?", True),  # negative grounding test
    ("What are common use cases of Agentic AI described in the eBook?", False),
]


def run(base_url: str) -> int:
    """Run all queries and return the number of failed checks."""
    failures = 0
    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"Server not reachable at {base_url} ({type(exc).__name__}).")
        return 1
    print(f"GET /health -> {health.json()}\n")

    for number, (question, must_refuse) in enumerate(QUERIES, start=1):
        print("=" * 78)
        print(f"[{number}] Question: {question}")
        try:
            response = requests.post(f"{base_url}/chat", json={"query": question}, timeout=120)
        except requests.RequestException as exc:
            print(f"    Request failed ({type(exc).__name__}). FAIL")
            failures += 1
            continue
        if response.status_code != 200:
            print(f"    HTTP {response.status_code}: {response.text}\n    FAIL")
            failures += 1
            continue

        body = response.json()
        refused = REFUSAL_MESSAGE.lower() in body["final_answer"].lower()
        print(f"Answer: {body['final_answer']}")
        print(f"Confidence score: {body['confidence_score']}")
        print("Retrieved context:")
        for chunk in body["retrieved_context"]:
            preview = chunk["content"][:200].replace("\n", " ")
            print(f"  - page {chunk['page']} | score {chunk['score']} | {preview}...")

        if must_refuse:
            ok = refused
        else:
            ok = not refused and bool(body["retrieved_context"])
        print("Result:", "PASS" if ok else "FAIL")
        failures += 0 if ok else 1

    print("=" * 78)
    print(f"{len(QUERIES) - failures}/{len(QUERIES)} checks passed")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark queries.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    sys.exit(1 if run(parser.parse_args().url.rstrip("/")) else 0)
