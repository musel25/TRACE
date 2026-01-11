## Retrieval (src/tracerag/retrieval/qdrant.py)

Purpose: retrieve the most relevant chunks from Qdrant given a query.

### Types

`Chunk`
- `id` (Any): Qdrant point id.
- `file_path` (str): best-effort source path.
- `chunk_index` (int): chunk index in source.
- `text` (str): chunk text (from payload).
- `score` (float): similarity score.
- `payload` (Dict[str, Any]): full payload from Qdrant.

`QdrantRetrievalConfig`
- `collection_name` (str)
- `top_k` (int, default 8)
- `query_filter` (Filter | None)
- `score_threshold` (float | None)
- `text_keys` (Sequence[str]): payload keys to search for text.
- `path_keys` (Sequence[str]): payload keys to search for source path.

### Functions

`build_openai_embedding_fn(client, model) -> Callable[[str], List[float]]`
- Input: OpenAI client + embedding model name.
- Output: function that turns text into embeddings.

`retrieve_qdrant_chunks(query, qdrant, embedding_fn, config, top_k=None, query_filter=None, score_threshold=None) -> List[Chunk]`
- Input:
  - `query` (str)
  - `qdrant` (QdrantClient)
  - `embedding_fn` (Callable[[str], List[float]])
  - `config` (QdrantRetrievalConfig)
  - `top_k`, `query_filter`, `score_threshold` (optional overrides)
- Output: list of `Chunk` sorted by score (highest first).

`build_qdrant_retriever(qdrant, embedding_fn, config) -> Retriever`
- Output: function `retriever(query, top_k=None, query_filter=None, score_threshold=None)`.

### Payload expectations

The retriever tries to extract text and source fields from payload keys:
- Text: `("text", "text_preview", "search_text", "description")`
- Path: `("file_path", "source_path", "module", "yang_id")`

If your payload uses different keys, override `text_keys` and `path_keys` in
`QdrantRetrievalConfig`.

### Minimal usage example

```python
from openai import OpenAI
from qdrant_client import QdrantClient

from tracerag.retrieval import (
    QdrantRetrievalConfig,
    build_openai_embedding_fn,
    retrieve_qdrant_chunks,
)

client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embed_fn = build_openai_embedding_fn(client, model="text-embedding-3-small")

config = QdrantRetrievalConfig(collection_name="catalog_embeddings")
chunks = retrieve_qdrant_chunks(
    "IOS XR telemetry for BGP",
    qdrant=qdrant,
    embedding_fn=embed_fn,
    config=config,
    top_k=8,
)
```
