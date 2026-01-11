#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

# ----------------------------- Logging ----------------------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

# -------------------------- OpenAI Embeddings ---------------------------------

class OpenAIEmbedder:
    def __init__(self, model: str, expected_dim: int, batch_size: int = 64):
        self.client = OpenAI()
        self.model = model
        self.expected_dim = expected_dim
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            batch_vecs = [d.embedding for d in resp.data]

            for v in batch_vecs:
                if len(v) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: got {len(v)} expected {self.expected_dim} (model={self.model})"
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
    if url:
        logger.info("Connecting to Qdrant via url=%s", url)
        return QdrantClient(url=url, api_key=api_key)
    h = host or "localhost"
    p = port or 6333
    logger.info("Connecting to Qdrant via host=%s port=%s", h, p)
    return QdrantClient(host=h, port=p)

def parse_distance(name: str) -> Distance:
    if name == "cosine":
        return Distance.COSINE
    if name == "dot":
        return Distance.DOT
    if name == "euclid":
        return Distance.EUCLID
    raise ValueError(f"Unknown distance: {name}")

def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    distance: Distance,
) -> None:
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
    if len(rows) != len(vectors):
        raise ValueError("rows and vectors must have the same length")

    points: List[PointStruct] = []
    for idx, (row, vec) in enumerate(zip(rows, vectors)):
        point_id = id_offset + idx
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
) -> int:
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

    return len(points)

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
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

def build_chunks(
    root: Path,
    *,
    chunk_chars: int,
    max_chunks: Optional[int] = None,
) -> List[Chunk]:
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
) -> int:
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

    return len(points)
