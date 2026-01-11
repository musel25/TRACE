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

from tracerag.retrieval import (
    QdrantRetrievalConfig,
    build_openai_embedding_fn,
    build_qdrant_retriever,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve relevant chunks from Qdrant.")

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

    # Embeddings
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="Embedding model.")

    # Output
    p.add_argument("--json", action="store_true", help="Print JSON output with chunks.")
    p.add_argument("--no-text", action="store_true", help="Print metadata only.")
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=None,
        help="Truncate chunk text after N characters.",
    )

    p.add_argument("query", type=str, help="User query.")
    return p.parse_args()


def build_filter(filter_field: Optional[str], filter_value: Optional[str]) -> Optional[Filter]:
    if not filter_field:
        return None
    if filter_value is None:
        raise SystemExit("--filter-value is required when --filter-field is set.")
    return Filter(must=[FieldCondition(key=filter_field, match=MatchValue(value=filter_value))])


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


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

    retrieval_config = QdrantRetrievalConfig(collection_name=args.collection)
    retriever = build_qdrant_retriever(
        qdrant=qdrant,
        embedding_fn=embed_fn,
        config=retrieval_config,
    )

    query_filter = build_filter(args.filter_field, args.filter_value)
    chunks = retriever(
        args.query,
        top_k=args.top_k,
        query_filter=query_filter,
        score_threshold=args.score_threshold,
    )

    if args.json:
        payload = {
            "query": args.query,
            "chunks": [
                {
                    "id": c.id,
                    "file_path": c.file_path,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "text": c.text,
                    "payload": c.payload,
                }
                for c in chunks
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for idx, chunk in enumerate(chunks, start=1):
        print(
            f"---\n#{idx} score={chunk.score:.4f} file={chunk.file_path} "
            f"chunk={chunk.chunk_index}"
        )
        if not args.no_text:
            text = truncate_text(chunk.text, args.max_text_chars)
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
