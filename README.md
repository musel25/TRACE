## TRACE

Tools for building a YANG-based catalog, ingesting embeddings into Qdrant, and
running retrieval or RAG over the resulting data.

### Quick start

1) Create a venv and install deps
```bash
uv sync
```

2) Set your OpenAI key
```bash
export OPENAI_API_KEY=...
```

3) Start Qdrant (local)
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest
```

4) Ingest raw YANG chunks (default mode)
```bash
uv run python scripts/ingest_qdrant.py \
  --yang-root data/yang/vendor/cisco/xr/701 \
  --chunk-chars 1000
```

5) Run retrieval or RAG
```bash
uv run python scripts/retrieve_chunks.py "IOS XR telemetry for BGP"
uv run python scripts/naive_rag.py "Generate telemetry configuration for BGP."
```

### Scripts

Each script can be run with `uv run python scripts/<name>.py --help`.

- `scripts/import_yang_modules.py`: sparse checkout IOS XR YANG modules into `data/yang`.
- `scripts/build_sensor_catalog.py`: build `data/sensor_catalog.jsonl` from YANG.
- `scripts/build_sensor_catalog_deprecated.py`: legacy catalog builder (kept for comparison).
- `scripts/ingest_qdrant.py`: embed + ingest into Qdrant; defaults to raw fixed-window chunks.
- `scripts/retrieve_chunks.py`: retrieve top-k chunks without calling an LLM.
- `scripts/naive_rag.py`: full RAG loop (retrieve, build context, chat).

For extended flags and examples, see `scripts/README.md`.

### RAG (how to wire your own)

The RAG layer is in `src/tracerag/rag/naive.py` and expects:
- Input query: string.
- Retriever: callable `retriever(query, top_k=None, query_filter=None, score_threshold=None)`.
- Chat function: callable that accepts messages and returns a string.

Use the docs here for parameter details and examples:
- `documentation/rag.md`
- `documentation/retrieval.md`

### Evaluation

Evaluation lives in notebooks (MLflow-based):
- `notebooks/evaluation.ipynb`: run RAG evaluations and log results to MLflow.
- `notebooks/rag.ipynb`: ad-hoc RAG experiments.
- `notebooks/visualization.ipynb`: visualize catalog or evaluation artifacts.

To view results:
```bash
mlflow ui
```

### Qdrant collections

Common defaults:
- `catalog_embeddings`: embeddings built from `sensor_catalog.jsonl`.
- `fixed_window_embeddings`: embeddings built from raw YANG chunks.

### Documentation (modules)

Module-level docs live in `documentation/`:
- `documentation/catalog.md`
- `documentation/ingestion.md`
- `documentation/retrieval.md`
- `documentation/rag.md`
- `documentation/yang.md`
