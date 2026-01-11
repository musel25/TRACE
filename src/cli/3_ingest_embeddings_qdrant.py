#!/usr/bin/env python3
"""
ingest_embeddings_qdrant.py

Ingest embeddings into Qdrant for:
  (A) Your YANG *catalog* entries (sensor_catalog.jsonl)
  (B) Raw fixed-window chunks of YANG module text

This script:
- Connects to Qdrant (local or cloud)
- Loads catalog JSONL and/or raw YANG files
- Embeds text with OpenAI embeddings
- Creates collections if needed
- Upserts points with payload metadata

Requires
--------
pip install qdrant-client openai

Env
---
OPENAI_API_KEY must be set for OpenAI embeddings.

Local Qdrant quickstart
-----------------------
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest

Examples
--------
# Ingest first 10 catalog entries
python src/embeddings/3_ingest_embeddings_qdrant.py catalog \
  --catalog-path data/sensor_catalog.jsonl \
  --collection catalog_embeddings \
  --limit 10

# Ingest up to 10 raw chunks from YANG folder (1000 chars per chunk)
python src/embeddings/3_ingest_embeddings_qdrant.py raw \
  --yang-root data/yang/vendor/cisco/xr/701 \
  --collection fixed_window_embeddings \
  --max-chunks 10 \
  --chunk-chars 1000

# Use cloud Qdrant (example)
python src/embeddings/3_ingest_embeddings_qdrant.py catalog \
  --qdrant-url https://YOUR-CLUSTER.qdrant.tech \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --catalog-path data/sensor_catalog.jsonl \
  --collection catalog_embeddings
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ----------------------------- Logging ----------------------------------------


def setup_logging(verbosity: int) -> None:
    """
    verbosity: 0 -> WARNING, 1 -> INFO, 2+ -> DEBUG
    """
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


logger = logging.getLogger(__name__)

# -------------------------- OpenAI Embeddings ---------------------------------


class OpenAIEmbedder:
    """
    Thin wrapper around OpenAI embeddings.

    Notes:
    - Uses OPENAI_API_KEY from environment via OpenAI() client.
    - Batches inputs to reduce latency and API overhead.
    """

    def __init__(self, model: str, expected_dim: int, batch_size: int = 64):
        self.client = OpenAI()
        self.model = model
        self.expected_dim = expected_dim
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Returns list of vectors aligned to input order.
        """
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            batch_vecs = [d.embedding for d in resp.data]

            # Basic sanity check (helps catch model/dim mismatch early)
            for v in batch_vecs:
                if len(v) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: got {len(v)} expected {self.expected_dim} "
                        f"(model={self.model})"
                    )

            vectors.extend(batch_vecs)

        return vectors


# ----------------------------- Qdrant Helpers ---------------------------------


def make_qdrant_client(
    *,
    host: Optional[str],
    port: Optional[int],
    url: Optional[str],
    api_key: Optional[str],
) -> QdrantClient:
    """
    Create Qdrant client for either:
      - local: host + port
      - cloud: url (+ api_key)
    """
    if url:
        logger.info("Connecting to Qdrant via url=%s", url)
        return QdrantClient(url=url, api_key=api_key)
    # default local
    h = host or "localhost"
    p = port or 6333
    logger.info("Connecting to Qdrant via host=%s port=%s", h, p)
    return QdrantClient(host=h, port=p)


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    """
    Create collection if missing.
    """
    if client.collection_exists(collection_name):
        logger.info("Collection exists: %s", collection_name)
        return

    logger.info("Creating collection: %s (size=%d, distance=%s)", collection_name, vector_size, distance)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )


def upsert_points_batched(
    client: QdrantClient,
    *,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 128,
) -> None:
    """
    Upsert in batches to avoid huge payloads.
    """
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info("Upserted %d/%d points into %s", min(i + batch_size, total), total, collection_name)


def debug_scroll_preview(
    client: QdrantClient,
    *,
    collection_name: str,
    limit: int = 10,
    show: int = 3,
) -> None:
    """
    Scroll and print a small preview of stored points.
    """
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )

    for p in points[:show]:
        v = p.vector
        v5 = v[:5] if isinstance(v, list) else None
        print(f"Point ID: {p.id}")
        if v5 is not None:
            print(f"Vector (first 5 dims): {v5}...")
        print(f"Payload keys: {sorted(list((p.payload or {}).keys()))}")
        print("---")


# ----------------------------- Catalog Ingest ---------------------------------


def load_catalog_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load catalog from JSONL (one JSON per line).
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_catalog_points(
    rows: List[Dict[str, Any]],
    vectors: List[List[float]],
    *,
    id_offset: int = 0,
) -> List[PointStruct]:
    """
    Build Qdrant points for catalog rows.

    Uses integer ids (required by Qdrant) and stores the original 'yang_id' too.
    """
    if len(rows) != len(vectors):
        raise ValueError("rows and vectors must have the same length")

    points: List[PointStruct] = []
    for idx, (row, vec) in enumerate(zip(rows, vectors)):
        point_id = id_offset + idx  # Qdrant point ID (uint-like)

        payload = {
            "yang_id": row.get("id"),
            "module": row.get("module"),
            "path": row.get("path"),
            "protocol_tag": row.get("protocol_tag"),
            "category": row.get("category"),
            "kind": row.get("kind"),
            "leaf_count": row.get("leaf_count"),
            "description": row.get("description"),
            "leaf_names": row.get("leaf_names"),
            # Keeping search_text is optional; it can bloat payload.
            # Uncomment if you want it stored:
            # "search_text": row.get("search_text"),
        }

        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    return points


def ingest_catalog(
    *,
    qdrant: QdrantClient,
    embedder: OpenAIEmbedder,
    collection: str,
    catalog_path: Path,
    limit: Optional[int],
    distance: Distance,
    upsert_batch_size: int,
    preview: bool,
) -> None:
    rows = load_catalog_jsonl(catalog_path)
    logger.info("Loaded %d catalog rows from %s", len(rows), catalog_path)

    if limit is not None:
        rows = rows[:limit]
        logger.info("Using first %d rows (limit)", len(rows))

    texts = [r.get("search_text", "") or "" for r in rows]
    vectors = embedder.embed_texts(texts)

    ensure_collection(qdrant, collection_name=collection, vector_size=embedder.expected_dim, distance=distance)

    points = build_catalog_points(rows, vectors)
    upsert_points_batched(qdrant, collection_name=collection, points=points, batch_size=upsert_batch_size)

    print(f"Uploaded {len(points)} catalog points to {collection}")

    if preview:
        debug_scroll_preview(qdrant, collection_name=collection)


# ----------------------------- Raw YANG Ingest --------------------------------


@dataclass
class Chunk:
    id: int
    file_path: str
    chunk_index: int
    text: str


def load_yang_files(root: Path) -> List[Path]:
    pattern = str(root / "**" / "*.yang")
    files = [Path(p) for p in glob.glob(pattern, recursive=True)]
    return sorted(files)


def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Naive fixed-size char chunking.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def build_chunks(
    root: Path,
    *,
    chunk_chars: int,
    max_chunks: Optional[int] = None,
) -> List[Chunk]:
    """
    Load *.yang files and create fixed window chunks with a small header.

    max_chunks: if provided, stops early for fast tests.
    """
    files = load_yang_files(root)
    logger.info("Found %d YANG files under %s", len(files), root)

    chunks: List[Chunk] = []
    cid = 0

    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Skipping %s: %s", f, e)
            continue

        pieces = chunk_text(content, max_chars=chunk_chars)
        for i, piece in enumerate(pieces):
            if max_chunks is not None and len(chunks) >= max_chunks:
                logger.info("Stopping early: reached %d chunks", max_chunks)
                return chunks

            # small header so embeddings have source context
            text = f"FILE: {f.name}\nCHUNK: {i}\n{piece}"
            chunks.append(Chunk(id=cid, file_path=str(f), chunk_index=i, text=text))
            cid += 1

    logger.info("Created %d chunks total", len(chunks))
    return chunks


def build_raw_points(
    chunks: List[Chunk],
    vectors: List[List[float]],
) -> List[PointStruct]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    points: List[PointStruct] = []
    for ch, vec in zip(chunks, vectors):
        payload = {
            "module": os.path.basename(ch.file_path),
            "file_path": ch.file_path,
            "chunk_index": ch.chunk_index,
            "text_preview": (ch.text[:500] + "...") if len(ch.text) > 500 else ch.text,
            "source": "raw_yang_module",
        }
        points.append(PointStruct(id=ch.id, vector=vec, payload=payload))
    return points


def ingest_raw_yang(
    *,
    qdrant: QdrantClient,
    embedder: OpenAIEmbedder,
    collection: str,
    yang_root: Path,
    chunk_chars: int,
    max_chunks: Optional[int],
    distance: Distance,
    upsert_batch_size: int,
    preview: bool,
) -> None:
    chunks = build_chunks(yang_root, chunk_chars=chunk_chars, max_chunks=max_chunks)
    logger.info("Built %d chunks", len(chunks))

    texts = [c.text for c in chunks]
    vectors = embedder.embed_texts(texts)

    ensure_collection(qdrant, collection_name=collection, vector_size=embedder.expected_dim, distance=distance)

    points = build_raw_points(chunks, vectors)
    upsert_points_batched(qdrant, collection_name=collection, points=points, batch_size=upsert_batch_size)

    print(f"Uploaded {len(points)} raw chunks to {collection}")

    if preview:
        debug_scroll_preview(qdrant, collection_name=collection)


# --------------------------------- CLI ---------------------------------------


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Embed and ingest YANG catalog / raw chunks into Qdrant.")

#     # Qdrant connection
#     p.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host (local mode).")
#     p.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (local mode).")
#     p.add_argument("--qdrant-url", type=str, default=None, help="Qdrant cloud URL (overrides host/port).")
#     p.add_argument("--qdrant-api-key", type=str, default=None, help="Qdrant cloud API key (if needed).")

#     # Embeddings
#     p.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="OpenAI embedding model.")
#     p.add_argument("--embedding-dim", type=int, default=1536, help="Expected embedding dimension.")
#     p.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding API.")
#     p.add_argument("--distance", type=str, default="cosine", choices=["cosine", "dot", "euclid"], help="Vector distance.")
#     p.add_argument("--upsert-batch-size", type=int, default=128, help="Batch size for Qdrant upserts.")
#     p.add_argument("--preview", action="store_true", help="Scroll and preview stored points.")
#     p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

#     sub = p.add_subparsers(dest="mode", required=True)

#     # Catalog mode
#     pc = sub.add_parser("catalog", help="Ingest sensor_catalog.jsonl entries")
#     pc.add_argument("--catalog-path", type=Path, required=True, help="Path to sensor_catalog.jsonl")
#     pc.add_argument("--collection", type=str, default="catalog_embeddings", help="Qdrant collection name")
#     pc.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")

#     # Raw YANG mode
#     pr = sub.add_parser("raw", help="Ingest raw YANG fixed-window chunks")
#     pr.add_argument("--yang-root", type=Path, required=True, help="Root folder containing YANG files")
#     pr.add_argument("--collection", type=str, default="fixed_window_embeddings", help="Qdrant collection name")
#     pr.add_argument("--chunk-chars", type=int, default=1000, help="Max chars per chunk")
#     pr.add_argument("--max-chunks", type=int, default=None, help="Stop after N chunks (for quick tests)")

#     return p.parse_args()

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed and ingest YANG catalog / raw chunks into Qdrant."
    )

    # Qdrant connection
    p.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host (local mode).")
    p.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (local mode).")
    p.add_argument("--qdrant-url", type=str, default=None, help="Qdrant cloud URL (overrides host/port).")
    p.add_argument("--qdrant-api-key", type=str, default=None, help="Qdrant cloud API key (if needed).")

    # Embeddings
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="OpenAI embedding model.")
    p.add_argument("--embedding-dim", type=int, default=1536, help="Expected embedding dimension.")
    p.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding API.")
    p.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "dot", "euclid"],
        help="Vector distance.",
    )
    p.add_argument("--upsert-batch-size", type=int, default=128, help="Batch size for Qdrant upserts.")
    p.add_argument("--preview", action="store_true", help="Scroll and preview stored points.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

    sub = p.add_subparsers(dest="mode", required=True)

    # Catalog mode
    pc = sub.add_parser("catalog", help="Ingest sensor_catalog.jsonl entries")
    pc.add_argument("--catalog-path", type=Path, required=True, help="Path to sensor_catalog.jsonl")
    pc.add_argument("--collection", type=str, default="catalog_embeddings", help="Qdrant collection name")
    pc.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")

    # Raw YANG mode (defaults changed here)
    pr = sub.add_parser("raw", help="Ingest raw YANG fixed-window chunks")
    pr.add_argument(
        "--yang-root",
        type=Path,
        default=Path("data/yang/vendor/cisco/xr/701"),
        help="Root folder containing YANG files",
    )
    pr.add_argument(
        "--collection",
        type=str,
        default="fixed_window_embeddings",
        help="Qdrant collection name",
    )
    pr.add_argument(
        "--chunk-chars",
        type=int,
        default=1000,
        help="Max chars per chunk",
    )
    pr.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Stop after N chunks (for quick tests)",
    )

    return p.parse_args()



def parse_distance(name: str) -> Distance:
    if name == "cosine":
        return Distance.COSINE
    if name == "dot":
        return Distance.DOT
    if name == "euclid":
        return Distance.EUCLID
    raise ValueError(f"Unknown distance: {name}")


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # Connection
    qdrant = make_qdrant_client(
        host=args.qdrant_host,
        port=args.qdrant_port,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )

    # Embedder
    embedder = OpenAIEmbedder(
        model=args.embedding_model,
        expected_dim=args.embedding_dim,
        batch_size=args.embed_batch_size,
    )

    distance = parse_distance(args.distance)

    # Dispatch
    if args.mode == "catalog":
        ingest_catalog(
            qdrant=qdrant,
            embedder=embedder,
            collection=args.collection,
            catalog_path=args.catalog_path.resolve(),
            limit=args.limit,
            distance=distance,
            upsert_batch_size=args.upsert_batch_size,
            preview=args.preview,
        )
        return 0

    if args.mode == "raw":
        ingest_raw_yang(
            qdrant=qdrant,
            embedder=embedder,
            collection=args.collection,
            yang_root=args.yang_root.resolve(),
            chunk_chars=args.chunk_chars,
            max_chunks=args.max_chunks,
            distance=distance,
            upsert_batch_size=args.upsert_batch_size,
            preview=args.preview,
        )
        return 0

    logger.error("Unknown mode: %s", args.mode)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
