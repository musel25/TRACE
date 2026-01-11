#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag.catalog.builder import build_catalog, setup_logging


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build XR telemetry sensor catalog JSONL from YANG modules.")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/yang/vendor/cisco/xr/701"),
        help="Folder containing YANG files (recursively). Default: data/yang/vendor/cisco/xr/701",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/sensor_catalog.jsonl"),
        help="Output JSONL path. Default: data/sensor_catalog.jsonl",
    )
    p.add_argument(
        "--min-leaves",
        type=int,
        default=2,
        help="Keep subtrees with at least this many leaf names. Default: 2",
    )
    p.add_argument("--max-depth", type=int, default=None, help="Optional recursion depth limit.")

    # mutually exclusive flags to allow default = oper-only True, with an explicit --no-oper to disable
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--oper-only",
        dest="oper_only",
        action="store_true",
        help="Only include operational modules (heuristic). (default)",
    )
    group.add_argument(
        "--no-oper",
        dest="oper_only",
        action="store_false",
        help="Do not restrict to operational modules.",
    )
    p.set_defaults(oper_only=True)

    p.add_argument("--max-leaves-in-text", type=int, default=30, help="Leaf names included in search_text.")
    p.add_argument("--max-text-chars", type=int, default=2000, help="Hard cap for search_text length.")

    # NEW (supported by the improved builder)
    p.add_argument(
        "--max-leaf-names-store",
        type=int,
        default=200,
        help="Cap number of leaf_names stored per entry (leaf_count is always full). Default: 200",
    )

    # Keep the old flag for compatibility; only pass it if your build_catalog still accepts it.
    # If you removed stats from build_catalog, you can delete this arg + the print section below.
    p.add_argument("--stats", action="store_true", help="Print search_text length stats.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

    args = p.parse_args()
    setup_logging(args.verbose)

    # Call build_catalog with the new parameter; keep everything else identical to your previous CLI.
    raise SystemExit(
        build_catalog(
            base_dir=args.base_dir.resolve(),
            out_json=args.out_json.resolve(),
            min_leaves=args.min_leaves,
            max_depth=args.max_depth,
            oper_only=args.oper_only,
            max_leaves_in_text=args.max_leaves_in_text,
            max_text_chars=args.max_text_chars,
            max_leaf_names_store=args.max_leaf_names_store
            # stats=args.stats,  # keep if your build_catalog supports it
        )
    )
