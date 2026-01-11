#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from tracerag.rag import build_context_text, build_openai_chat_fn, naive_rag
from tracerag.retrieval import (
    QdrantRetrievalConfig,
    build_openai_embedding_fn,
    build_qdrant_retriever,
)


DEFAULT_SYSTEM_PROMPT = """You are a Cisco IOS XR network engineer.
Using ONLY the information in the CONTEXT, generate a telemetry configuration.
Output valid IOS XR CLI configuration blocks, nothing else.
If you are unsure, make the best reasonable guess but stay consistent with IOS XR syntax."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a naive RAG query against Qdrant.")

    # Qdrant connection
    p.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host (local mode).")
    p.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (local mode).")
    p.add_argument("--qdrant-url", type=str, default=None, help="Qdrant cloud URL (overrides host/port).")
    p.add_argument("--qdrant-api-key", type=str, default=None, help="Qdrant cloud API key (if needed).")

    # Retrieval
    p.add_argument("--collection", type=str, default="catalog_embeddings", help="Qdrant collection.")
    p.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve.")
    p.add_argument("--score-threshold", type=float, default=None, help="Minimum score to keep.")
    p.add_argument("--filter-field", type=str, default=None, help="Payload field to filter on.")
    p.add_argument("--filter-value", type=str, default=None, help="Exact match value for filter-field.")

    # Embeddings and LLM
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="Embedding model.")
    p.add_argument("--chat-model", type=str, default="gpt-4.1-mini", help="Chat model.")
    p.add_argument("--temperature", type=float, default=0.2, help="Chat temperature.")

    # Prompting
    p.add_argument("--system-prompt", type=str, default=None, help="Inline system prompt.")
    p.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Path to a system prompt file (overrides --system-prompt).",
    )
    p.add_argument(
        "--answer-instruction",
        type=str,
        default="Return only the answer.",
        help="Final instruction appended to the user message.",
    )

    # Output
    p.add_argument("--show-context", action="store_true", help="Print full context before answer.")
    p.add_argument("--no-sources", action="store_true", help="Omit source headers in context.")
    p.add_argument("--json", action="store_true", help="Print JSON output with chunks.")

    p.add_argument("query", type=str, help="User query.")
    return p.parse_args()


def build_filter(filter_field: Optional[str], filter_value: Optional[str]) -> Optional[Filter]:
    if not filter_field:
        return None
    if filter_value is None:
        raise SystemExit("--filter-value is required when --filter-field is set.")
    return Filter(must=[FieldCondition(key=filter_field, match=MatchValue(value=filter_value))])


def main() -> int:
    args = parse_args()

    qdrant = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        host=args.qdrant_host,
        port=args.qdrant_port,
    )

    client = OpenAI()
    embed_fn = build_openai_embedding_fn(client, model=args.embedding_model)
    chat_fn = build_openai_chat_fn(client, model=args.chat_model, temperature=args.temperature)

    retrieval_config = QdrantRetrievalConfig(
        collection_name=args.collection,
    )
    retriever = build_qdrant_retriever(
        qdrant=qdrant,
        embedding_fn=embed_fn,
        config=retrieval_config,
    )

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    elif args.system_prompt:
        system_prompt = args.system_prompt

    query_filter = build_filter(args.filter_field, args.filter_value)
    if args.no_sources:
        context_builder = lambda chunks: build_context_text(chunks, include_sources=False)
    else:
        context_builder = build_context_text

    response = naive_rag(
        args.query,
        retriever=lambda q, k: retriever(
            q, top_k=k, query_filter=query_filter, score_threshold=args.score_threshold
        ),
        chat_fn=chat_fn,
        system_prompt=system_prompt,
        top_k=args.top_k,
        context_builder=context_builder,
        answer_instruction=args.answer_instruction,
    )

    if args.json:
        payload = {
            "answer": response.answer,
            "chunks": [
                {
                    "id": c.id,
                    "file_path": c.file_path,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "text": c.text,
                    "payload": c.payload,
                }
                for c in response.chunks
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.show_context:
        print(response.context)
        print("\n---\n")
    print(response.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
