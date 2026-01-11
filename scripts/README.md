## Scripts

### Import YANG modules

`scripts/import_yang_modules.py` pulls a specific IOS-XR vendor folder via git sparse checkout.

Key options:
- `--repo-url`: source repo URL (default: YangModels/yang)
- `--xr-version`: XR version folder (default: 701)
- `--output-root`: root folder to write under (default: `data/yang`)
- `--tmp-dir`: temporary sparse checkout folder (default: `.tmp_yang_sparse`)

Example:
```bash
uv run python scripts/import_yang_modules.py --xr-version 701
```

### Build sensor catalog

`scripts/build_sensor_catalog.py` parses the YANG tree into a JSONL catalog.

Key options:
- `--base-dir`: YANG root folder (default: `data/yang/vendor/cisco/xr/701`)
- `--out-json`: output JSONL path (default: `data/sensor_catalog.jsonl`)
- `--min-leaves`: minimum leaf/leaf-list count per subtree (default: 2)
- `--max-depth`: recursion depth cap (default: no limit)
- `--oper-only` / `--no-oper`: include only operational modules (default: `--oper-only`)
- `--max-leaves-in-text`: number of leaf names in `search_text` (default: 30)
- `--max-text-chars`: hard cap for `search_text` length (default: 2000)
- `--stats`: print `search_text` length stats
- `-v` / `-vv`: increase logging verbosity

Example:
```bash
uv run python scripts/build_sensor_catalog.py --min-leaves 3 --stats
```

### Ingest embeddings into Qdrant

First activate qdrant: 

docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest

`scripts/ingest_qdrant.py` embeds and upserts either:
- fixed-window chunks from raw YANG files (`raw`, default-style naive chunking)
- or `sensor_catalog.jsonl` entries (`catalog`)

Global options:
- `--qdrant-host`: Qdrant host (local mode)
- `--qdrant-port`: Qdrant port (local mode)
- `--qdrant-url`: Qdrant cloud URL (overrides host/port)
- `--qdrant-api-key`: Qdrant cloud API key
- `--embedding-model`: OpenAI embedding model
- `--embedding-dim`: embedding dimension (must match model)
- `--embed-batch-size`: batch size for embedding requests
- `--distance`: vector distance (`cosine`, `dot`, `euclid`)
- `--upsert-batch-size`: batch size for Qdrant upserts
- `--no-resume`: re-embed from scratch instead of skipping already ingested points
- `--preview`: scroll and preview stored points
- `-v` / `-vv`: increase verbosity

Mode: `catalog`
- `--catalog-path`: path to `sensor_catalog.jsonl`
- `--collection`: Qdrant collection name (default: `catalog_embeddings`)
- `--limit`: optional limit for quick tests

Mode: `raw`
- `--yang-root`: root folder containing YANG files
- `--collection`: Qdrant collection name (default: `fixed_window_embeddings`)
- `--chunk-chars`: max chars per chunk (fixed window naive chunking)
- `--limit`: stop after N chunks (for quick tests). Default: embed all.

Examples:
```bash
uv run python scripts/ingest_qdrant.py catalog --catalog-path data/sensor_catalog.jsonl
uv run python scripts/ingest_qdrant.py raw --yang-root data/yang/vendor/cisco/xr/701 --chunk-chars 1000 --limit 10
```

### Naive RAG CLI

`scripts/naive_rag.py` runs a retrieval-augmented generation query using a Qdrant
collection and OpenAI models. It is intentionally configurable so you can vary
the prompt, models, and retrieval settings between experiments.

Key options:
- `--collection`: Qdrant collection (e.g. `catalog_embeddings` or `fixed_window_embeddings`)
- `--top-k`: number of chunks to retrieve
- `--score-threshold`: drop low-scoring hits
- `--filter-field` / `--filter-value`: exact-match payload filter
- `--embedding-model`: embedding model (e.g. `text-embedding-3-small`)
- `--chat-model`: chat model (e.g. `gpt-4.1-mini`)
- `--temperature`: chat temperature
- `--system-prompt`: inline system prompt string
- `--system-prompt-file`: file path containing the system prompt (overrides `--system-prompt`)
- `--answer-instruction`: final instruction appended to the user message
- `--show-context`: print the assembled context before the answer
- `--no-sources`: omit source headers in the context
- `--json`: emit a JSON payload with answer + retrieved chunks

Examples:
```bash
uv run python scripts/naive_rag.py \
  --collection catalog_embeddings \
  --top-k 10 \
  --system-prompt-file data/iosxr_prompt.txt \
  "Generate telemetry configuration for Cisco IOS XR about BGP."

uv run python scripts/naive_rag.py \
  --collection fixed_window_embeddings \
  --top-k 12 \
  --embedding-model text-embedding-3-small \
  --chat-model gpt-4.1-mini \
  --temperature 0.1 \
  --filter-field protocol_tag \
  --filter-value bgp \
  --answer-instruction "Return only IOS XR telemetry configuration." \
  "Generate telemetry configuration for IOS XR about BGP."
```
