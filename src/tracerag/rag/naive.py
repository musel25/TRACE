from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from openai import OpenAI

from tracerag.retrieval import Chunk


@dataclass
class RagResponse:
    answer: str
    chunks: List[Chunk]
    context: str
    messages: List[Dict[str, str]]


ChatFn = Callable[[List[Dict[str, str]]], str]


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
