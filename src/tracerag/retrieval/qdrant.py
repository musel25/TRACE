from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter


@dataclass
class Chunk:
    id: Any
    file_path: str
    chunk_index: int
    text: str
    score: float
    payload: Dict[str, Any]


@dataclass(frozen=True)
class QdrantRetrievalConfig:
    collection_name: str
    top_k: int = 8
    query_filter: Optional[Filter] = None
    score_threshold: Optional[float] = None
    text_keys: Sequence[str] = ("text", "text_preview", "search_text", "description")
    path_keys: Sequence[str] = ("file_path", "source_path", "module", "yang_id")


Retriever = Callable[[str, Optional[int], Optional[Filter], Optional[float]], List[Chunk]]


def build_openai_embedding_fn(client: OpenAI, *, model: str) -> Callable[[str], List[float]]:
    def _embed(text: str) -> List[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


def _extract_payload_text(payload: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    description = payload.get("description")
    path = payload.get("path")
    if isinstance(description, str) or isinstance(path, str):
        return f"{description or ''}\n{path or ''}".strip()
    return ""


def _extract_payload_path(payload: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def retrieve_qdrant_chunks(
    query: str,
    *,
    qdrant: QdrantClient,
    embedding_fn: Callable[[str], List[float]],
    config: QdrantRetrievalConfig,
    top_k: Optional[int] = None,
    query_filter: Optional[Filter] = None,
    score_threshold: Optional[float] = None,
) -> List[Chunk]:
    query_vector = embedding_fn(query)
    effective_top_k = top_k if top_k is not None else config.top_k
    effective_filter = query_filter if query_filter is not None else config.query_filter
    effective_threshold = (
        score_threshold if score_threshold is not None else config.score_threshold
    )

    hits = qdrant.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        query_filter=effective_filter,
        limit=effective_top_k,
        with_payload=True,
    ).points

    results: List[Chunk] = []
    for hit in hits:
        if effective_threshold is not None and hit.score < effective_threshold:
            continue

        payload = hit.payload or {}
        text = _extract_payload_text(payload, config.text_keys)
        file_path = _extract_payload_path(payload, config.path_keys)
        chunk_index = payload.get("chunk_index", 0)
        if not isinstance(chunk_index, int):
            try:
                chunk_index = int(chunk_index)
            except (TypeError, ValueError):
                chunk_index = 0

        results.append(
            Chunk(
                id=hit.id,
                file_path=file_path,
                chunk_index=chunk_index,
                text=text,
                score=float(hit.score or 0.0),
                payload=payload,
            )
        )

    return sorted(results, key=lambda c: c.score, reverse=True)[:effective_top_k]


def build_qdrant_retriever(
    *,
    qdrant: QdrantClient,
    embedding_fn: Callable[[str], List[float]],
    config: QdrantRetrievalConfig,
) -> Retriever:
    def _retrieve(
        query: str,
        top_k: Optional[int] = None,
        query_filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Chunk]:
        return retrieve_qdrant_chunks(
            query,
            qdrant=qdrant,
            embedding_fn=embedding_fn,
            config=config,
            top_k=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

    return _retrieve
