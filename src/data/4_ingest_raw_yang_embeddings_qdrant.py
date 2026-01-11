#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_ingest_lib import (
    OpenAIEmbedder,
    make_qdrant_client,
    parse_distance,
    setup_logging,
    ingest_raw_yang,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed and ingest raw YANG fixed-window chunks into Qdrant.")

    # Qdrant
    p.add_argument("--qdrant-host", type=str, default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--qdrant-url", type=str, default=None)
    p.add_argument("--qdrant-api-key", type=str, default=None)

    # Embeddings
    p.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    p.add_argument("--embedding-dim", type=int, default=1536)
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--distance", type=str, default="cosine", choices=["cosine", "dot", "euclid"])
    p.add_argument("--upsert-batch-size", type=int, default=128)

    # Job (your defaults)
    p.add_argument("--yang-root", type=Path, default=Path("data/yang/vendor/cisco/xr/701"))
    p.add_argument("--collection", type=str, default="fixed_window_embeddings")
    p.add_argument("--chunk-chars", type=int, default=1000)
    p.add_argument("--max-chunks", type=int, default=10)

    p.add_argument("--preview", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)

    return p.parse_args()

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    qdrant = make_qdrant_client(
        host=args.qdrant_host,
        port=args.qdrant_port,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )

    embedder = OpenAIEmbedder(
        model=args.embedding_model,
        expected_dim=args.embedding_dim,
        batch_size=args.embed_batch_size,
    )

    distance = parse_distance(args.distance)

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

if __name__ == "__main__":
    raise SystemExit(main())
