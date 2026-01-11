## Embedding + Qdrant ingestion (src/tracerag/embeddings/qdrant_ingest.py)

Purpose: create embeddings and upsert points into Qdrant.

### Core functions

`build_qdrant_client(host, port, url=None, api_key=None) -> QdrantClient`
- Input: local host/port or a cloud URL + API key.
- Output: initialized Qdrant client.

`ensure_collection(client, collection, vector_size, distance) -> None`
- Creates collection if missing and validates vector size if it exists.

`iter_yang_chunks(yang_root, chunk_chars, max_chunks=None) -> Iterator[Tuple[str, dict]]`
- Input: folder with `.yang` files.
- Output: `(text, payload)` pairs where payload includes:
  - `source_path`, `chunk_index`, `chunk_chars`, `text`

`iter_catalog_entries(catalog_path, limit=None) -> Iterator[Tuple[str, dict]]`
- Input: path to `sensor_catalog.jsonl`.
- Output: `(search_text, payload)` pairs; payload copies the JSONL row.

`ingest_stream(text_payloads, qdrant, openai_client, collection, embedding_model, embed_batch_size, upsert_batch_size, resume=True) -> int`
- Input:
  - `text_payloads`: iterable of `(text, payload)` tuples.
  - `openai_client`: OpenAI client for embeddings.
  - `collection`: Qdrant collection name.
  - `embedding_model`: OpenAI embedding model name.
  - `embed_batch_size`: batch size per embedding request.
  - `upsert_batch_size`: batch size per Qdrant upsert.
  - `resume`: skip already-ingested points when `True`.
- Output: number of points upserted.

`preview_points(client, collection, limit=5) -> None`
- Prints a small preview of stored points and payload keys.

### Typical flow (raw YANG chunks)

```python
from pathlib import Path
from openai import OpenAI

from tracerag.embeddings.qdrant_ingest import (
    build_qdrant_client,
    ensure_collection,
    distance_from_str,
    ingest_stream,
    iter_yang_chunks,
)

qdrant = build_qdrant_client(host="localhost", port=6333, url=None, api_key=None)
ensure_collection(
    qdrant,
    collection="fixed_window_embeddings",
    vector_size=1536,
    distance=distance_from_str("cosine"),
)

text_payloads = iter_yang_chunks(
    yang_root=Path("data/yang/vendor/cisco/xr/701"),
    chunk_chars=1000,
    max_chunks=None,
)

total = ingest_stream(
    text_payloads=text_payloads,
    qdrant=qdrant,
    openai_client=OpenAI(),
    collection="fixed_window_embeddings",
    embedding_model="text-embedding-3-small",
    embed_batch_size=64,
    upsert_batch_size=128,
    resume=True,
)
print(total)
```

### Typical flow (catalog JSONL)

Use `iter_catalog_entries` instead of `iter_yang_chunks` and point the
collection at `catalog_embeddings` or a new name.
