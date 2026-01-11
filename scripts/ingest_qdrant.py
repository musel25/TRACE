#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI

from tracerag.embeddings.qdrant_ingest import (
    build_qdrant_client,
    distance_from_str,
    ensure_collection,
    ingest_stream,
    iter_catalog_entries,
    iter_yang_chunks,
    preview_points,
    setup_logging,
)


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
    p.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-embed from scratch instead of skipping already ingested points.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    p.set_defaults(resume=True)

    sub = p.add_subparsers(dest="mode", required=False)

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
        "--limit",
        type=int,
        default=None,
        help="Stop after N chunks (for quick tests). Default: embed all.",
    )

    argv = sys.argv[1:]
    if not any(arg in {"raw", "catalog"} for arg in argv):
        if not any(arg in {"-h", "--help"} for arg in argv):
            argv = ["raw", *argv]
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    qdrant = build_qdrant_client(
        host=args.qdrant_host,
        port=args.qdrant_port,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )
    openai_client = OpenAI()
    distance = distance_from_str(args.distance)

    collection = args.collection
    ensure_collection(
        qdrant,
        collection=collection,
        vector_size=args.embedding_dim,
        distance=distance,
    )

    if args.mode == "catalog":
        if not args.catalog_path.exists():
            raise SystemExit(f"Catalog not found: {args.catalog_path}")
        if args.catalog_path.stat().st_size == 0:
            raise SystemExit(
                f"Catalog is empty: {args.catalog_path}. Run scripts/build_sensor_catalog.py first."
            )
        text_payloads = iter_catalog_entries(args.catalog_path, limit=args.limit)
    elif args.mode == "raw":
        if not args.yang_root.exists():
            raise SystemExit(f"YANG root not found: {args.yang_root}")
        text_payloads = iter_yang_chunks(
            args.yang_root,
            chunk_chars=args.chunk_chars,
            max_chunks=args.limit,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    total = ingest_stream(
        text_payloads=text_payloads,
        qdrant=qdrant,
        openai_client=openai_client,
        collection=collection,
        embedding_model=args.embedding_model,
        embed_batch_size=args.embed_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        resume=args.resume,
    )

    print(f"Upserted {total} points into '{collection}'.")

    if args.preview:
        preview_points(qdrant, collection)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
