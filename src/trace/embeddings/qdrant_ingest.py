from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_qdrant_client(
    *,
    host: str,
    port: int,
    url: str | None,
    api_key: str | None,
) -> QdrantClient:
    if url:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(host=host, port=port)


def distance_from_str(name: str) -> models.Distance:
    if name == "cosine":
        return models.Distance.COSINE
    if name == "dot":
        return models.Distance.DOT
    if name == "euclid":
        return models.Distance.EUCLID
    raise ValueError(f"Unknown distance: {name}")


def ensure_collection(
    client: QdrantClient,
    *,
    collection: str,
    vector_size: int,
    distance: models.Distance,
) -> None:
    try:
        info = client.get_collection(collection)
    except Exception:
        info = None

    if info is None:
        logger.info("Creating collection %s", collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        return

    existing_size = info.config.params.vectors.size
    if existing_size != vector_size:
        raise ValueError(
            f"Collection '{collection}' vector size mismatch: {existing_size} != {vector_size}"
        )


def chunk_text(text: str, *, max_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i : i + max_chars])
    return chunks


def iter_yang_chunks(
    yang_root: Path,
    *,
    chunk_chars: int,
    max_chunks: Optional[int],
) -> Iterator[Tuple[str, dict]]:
    yang_files = sorted(yang_root.rglob("*.yang"))
    logger.info("Found %d .yang files under %s", len(yang_files), yang_root)
    count = 0
    for path in yang_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, max_chars=chunk_chars)
        for idx, chunk in enumerate(chunks):
            payload = {
                "source_path": str(path),
                "chunk_index": idx,
                "text": chunk,
            }
            yield chunk, payload
            count += 1
            if max_chunks is not None and count >= max_chunks:
                return


def iter_catalog_entries(
    catalog_path: Path,
    *,
    limit: Optional[int],
) -> Iterator[Tuple[str, dict]]:
    with catalog_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            row = json.loads(line)
            payload = dict(row)
            text = row.get("search_text") or ""
            yield text, payload


def embed_texts(
    client: OpenAI,
    *,
    texts: Sequence[str],
    model: str,
) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in resp.data]


T = TypeVar("T")


def batch_iterable(items: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_stream(
    *,
    text_payloads: Iterable[Tuple[str, dict]],
    qdrant: QdrantClient,
    openai_client: OpenAI,
    collection: str,
    embedding_model: str,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> int:
    total = 0
    for batch in batch_iterable(text_payloads, embed_batch_size):
        texts = [item[0] for item in batch]
        payloads = [item[1] for item in batch]
        embeddings = embed_texts(openai_client, texts=texts, model=embedding_model)

        points: List[models.PointStruct] = []
        for i, (payload, vector) in enumerate(zip(payloads, embeddings)):
            point_id = payload.get("id")
            if point_id is None:
                point_id = total + i
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        for chunk in batch_iterable(points, upsert_batch_size):
            qdrant.upsert(collection_name=collection, points=chunk, wait=True)
            total += len(chunk)
            logger.info("Upserted %d points", total)
    return total


def preview_points(client: QdrantClient, collection: str, limit: int = 5) -> None:
    points, _ = client.scroll(collection_name=collection, limit=limit)
    print(f"Previewing {len(points)} points from '{collection}':")
    for p in points:
        print(f"- id={p.id} keys={list((p.payload or {}).keys())}")

