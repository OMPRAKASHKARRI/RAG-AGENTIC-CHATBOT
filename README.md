# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangGraph, Pinecone, Groq, and FastAPI. The chatbot answers questions using the **Agentic AI for Executives** eBook as its knowledge source and refuses unsupported questions when the retrieved context does not contain sufficient information.

## Features

- PDF ingestion and text chunking
- Semantic vector retrieval with Pinecone
- Pinecone Inference embeddings (`llama-text-embed-v2`)
- LangGraph retrieval → generation workflow
- Groq LLM for grounded response generation
- Strict document grounding
- Unsupported-question refusal
- FastAPI REST API
- Swagger/OpenAPI documentation
- Retrieved context and relevance/confidence score in API responses
- Automated six-query benchmark

## Architecture

```text
Agentic AI eBook
      |
      v
PDF extraction
      |
      v
Text chunking
      |
      v
Pinecone Inference embeddings
llama-text-embed-v2 (1024 dimensions)
      |
      v
Pinecone vector index
      ^
      |
User query
      |
      v
LangGraph
  retrieve
      |
      v
Relevant context
      |
      v
  generate
      |
      v
Groq LLM
      |
      v
Answer + context + confidence
      |
      v
FastAPI /chat
```

LangGraph flow:

```text
START -> retrieve -> generate -> END
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| RAG orchestration | LangGraph |
| Document processing | PyPDF / LangChain |
| Vector database | Pinecone |
| Embeddings | Pinecone Inference `llama-text-embed-v2` |
| LLM | Groq |
| API | FastAPI |
| Server | Uvicorn |
| Configuration | python-dotenv |
| Benchmark client | requests |

## Project Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── clients.py
│   ├── config.py
│   ├── graph.py
│   └── ingestion.py
├── app.py
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10+
- Pinecone API key
- Groq API key
- Internet connection

The current implementation does **not** require an OpenAI API key.

## Installation

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Environment Variables

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index-api
```

Never commit `.env`. Use `.env.example` for placeholders.

## Document Ingestion

Place the source PDF at:

```text
data/Ebook-Agentic-AI.pdf
```

Run:

```bash
python -m src.ingestion --reset
```

The validated ingestion run produced:

```text
59 pages
119 chunks
119 vectors upserted
0 already present
```

Current Pinecone configuration:

```text
Index: agentic-ai-index-api
Dimension: 1024
Metric: cosine
Embedding: llama-text-embed-v2
```

## Run the API

```bash
uvicorn app:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Health check:

```text
GET /health
```

Expected:

```json
{"status":"ok"}
```

## Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

### POST /chat

Request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Response structure:

```json
{
  "final_answer": "According to the eBook, ...",
  "retrieved_context": [
    {
      "content": "...",
      "page": 8,
      "score": 0.4891,
      "source": "Ebook-Agentic-AI.pdf",
      "chunk_index": 0
    }
  ],
  "confidence_score": 0.5715
}
```

The API returns:

1. The final grounded answer
2. Retrieved source chunks
3. A confidence/relevance score

## Grounding Behavior

The generation step is instructed to answer only from retrieved eBook context.

For unsupported questions, the system returns:

```text
I cannot answer this based on the provided document.
```

For example:

```text
Who won the 2022 FIFA World Cup?
```

is intentionally refused because the answer is outside the supplied document.

## Benchmark Testing

Start the API:

```bash
uvicorn app:app --reload
```

In another terminal:

```bash
python tests_sample_queries.py
```

The six benchmark questions are:

1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. Who won the 2022 FIFA World Cup?
6. What are common use cases of Agentic AI described in the eBook?

### Current benchmark result

```text
6/6 checks passed
```

| # | Test | Confidence | Result |
|---|---|---:|---|
| 1 | Agentic AI definition | 0.5715 | PASS |
| 2 | Agents vs traditional automation | 0.4363 | PASS |
| 3 | Agentic Architecture components | 0.5572 | PASS |
| 4 | Role of memory | 0.4776 | PASS |
| 5 | FIFA World Cup negative test | 0.0000 | PASS — refused |
| 6 | Agentic AI use cases | 0.5599 | PASS |

The negative test is expected to pass by refusing to answer.

## Example Retrieved Knowledge

The benchmark confirms retrieval of document sections covering:

- Agentic AI definition and characteristics
- Differences between agents and rule-based automation
- Perception, reasoning, planning, learning, and execution
- Memory and continuous learning
- Business value and capabilities
- Industry use cases

## Embedding Implementation Note

The assignment reference specifies OpenAI `text-embedding-3-small` with a 1536-dimensional Pinecone index.

This implementation uses Pinecone Inference `llama-text-embed-v2` with a 1024-dimensional Pinecone index instead. This removes the OpenAI embedding dependency while preserving the required RAG pipeline:

```text
PDF
-> chunking
-> embeddings
-> Pinecone
-> LangGraph retrieval
-> Groq generation
-> FastAPI
```

This README intentionally documents the actual implementation rather than claiming that OpenAI embeddings are being used.

## Security

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
.pytest_cache/
```

Verify that `.env` is not tracked:

```bash
git ls-files .env
```

The command should return no output.

## Troubleshooting

### `requests` import warning

```bash
python -m pip install requests
```

Verify:

```bash
python -c "import requests; print(requests.__version__)"
```

### API is not reachable

Start the server:

```bash
uvicorn app:app --reload
```

Then check:

```text
http://127.0.0.1:8000/health
```

### Pinecone dimension mismatch

The current embedding model produces 1024-dimensional vectors. Use:

```text
agentic-ai-index-api
```

and rerun:

```bash
python -m src.ingestion --reset
```


## License

Created for assignment evaluation and educational purposes.
