from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

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


@dataclass
class RagResponse:
    answer: str
    chunks: List[Chunk]
    context: str
    messages: List[Dict[str, str]]


Retriever = Callable[[str, Optional[int], Optional[Filter], Optional[float]], List[Chunk]]
ChatFn = Callable[[List[Dict[str, str]]], str]


def build_openai_embedding_fn(client: OpenAI, *, model: str) -> Callable[[str], List[float]]:
    def _embed(text: str) -> List[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


def build_openai_chat_fn(
    client: OpenAI,
    *,
    model: str,
    temperature: float = 0.2,
) -> ChatFn:
    def _chat(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    return _chat


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


def build_context_text(
    chunks: Iterable[Chunk],
    *,
    include_sources: bool = True,
    separator: str = "---",
) -> str:
    parts: List[str] = []
    for chunk in chunks:
        if include_sources:
            parts.append(
                f"{separator}\nSource file: {os.path.basename(chunk.file_path)} "
                f"| chunk {chunk.chunk_index}\n{chunk.text}"
            )
        else:
            parts.append(chunk.text)
    return "\n".join(parts)


def build_rag_messages(
    *,
    system_prompt: str,
    user_query: str,
    context: str,
    answer_instruction: str = "Return only the answer.",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"USER REQUEST:\n{user_query}\n\n"
                f"{answer_instruction}"
            ),
        },
    ]


def naive_rag(
    user_query: str,
    *,
    retriever: Callable[[str, Optional[int]], List[Chunk]],
    chat_fn: ChatFn,
    system_prompt: str,
    top_k: int = 8,
    context_builder: Callable[[Iterable[Chunk]], str] = build_context_text,
    answer_instruction: str = "Return only the answer.",
) -> RagResponse:
    chunks = retriever(user_query, top_k)
    context = context_builder(chunks)
    messages = build_rag_messages(
        system_prompt=system_prompt,
        user_query=user_query,
        context=context,
        answer_instruction=answer_instruction,
    )
    answer = chat_fn(messages)
    return RagResponse(answer=answer, chunks=chunks, context=context, messages=messages)
