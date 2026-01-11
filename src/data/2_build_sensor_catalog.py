#!/usr/bin/env python3
"""
build_sensor_catalog.py

Build a JSONL "sensor catalog" from Cisco IOS XR YANG modules (pyang).

What it does
------------
1) Recursively loads all *.yang files under a base directory with pyang.
2) (Optionally) filters to operational models (e.g., *-oper.yang or /oper/ in path).
3) Traverses each module to collect container/list subtrees that have >= MIN_LEAVES
   distinct leaf/leaf-list fields.
4) Adds heuristic semantic tags:
   - protocol_tag (bgp/ospf/isis/mpls/ldp/...)
   - category tags (neighbors, state, stats, ...)
5) Builds a compact "search_text" field for embedding/retrieval.
6) Writes entries as JSON Lines (one JSON object per line).

Output schema
-------------
Each line is a dict like:
{
  "id": int,
  "module": str,
  "path": str,
  "kind": "container" | "list",
  "protocol_tag": str | null,
  "category": list[str],
  "leaf_names": list[str],
  "leaf_count": int,
  "description": str,
  "search_text": str
}

Usage
-----
python src/data/2_build_sensor_catalog.py \
  --base-dir data/yang/vendor/cisco/xr/701 \
  --out-json data/sensor_catalog.jsonl \
  --min-leaves 2 \
  --oper-only

Notes
-----
- Requires: pyang
- This script is intentionally "pure extraction + JSONL write". Any plotting/stats
  are behind an optional flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyang import context, repository, statements

# ----------------------------- Logging ----------------------------------------


def setup_logging(verbosity: int) -> None:
    """
    verbosity: 0 -> WARNING, 1 -> INFO, 2+ -> DEBUG
    """
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


logger = logging.getLogger(__name__)

# ----------------------------- YANG Loading -----------------------------------


def load_modules(base_dir: Path) -> Tuple[context.Context, List[statements.Statement]]:
    """
    Parse all .yang files under base_dir with pyang.

    Returns:
      (ctx, modules)
      - ctx: pyang Context after validation
      - modules: list of parsed module statements
    """
    repo = repository.FileRepository(str(base_dir))
    ctx = context.Context(repo)

    modules: List[statements.Statement] = []
    yang_files = sorted(base_dir.rglob("*.yang"))
    logger.info("Found %d .yang files under %s", len(yang_files), base_dir)
    # Inform the user we're starting a possibly long parse step
    print(f"Parsing {len(yang_files)} .yang files under {base_dir} ...")

    for i, p in enumerate(yang_files, 1):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            mod = ctx.add_module(p.name, text)
            if mod is not None:
                modules.append(mod)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", p, e)

        # Periodic progress update to stdout so the user knows we're working
        if i % 50 == 0 or i == len(yang_files):
            print(f"  parsed {i}/{len(yang_files)} files", flush=True)
    # newline after progress
    print("", flush=True)

    # pyang ctx.validate() can be expensive for large repositories. Print a
    # visible message and time the operation so the user knows the script is
    # still working during this pause.
    try:
        print("Validating parsed modules with pyang (this can take a while)...", flush=True)
    except Exception as e:
        # Validation can be noisy but still useful; we allow continuing if parse succeeded.
        logger.warning("pyang ctx.validate() raised: %s", e)
        print(f"pyang validation raised an exception: {e}", flush=True)

    return ctx, modules


def module_source_ref(mod: statements.Statement) -> str:
    """
    Best-effort source reference string for a module, used for filtering.
    """
    try:
        if mod.pos is not None and mod.pos.ref:
            return str(mod.pos.ref)
    except Exception:
        pass
    return str(mod.arg or "")


def is_operational_module(mod: statements.Statement) -> bool:
    """
    Heuristic: focus on operational models for telemetry.
    - filenames ending in "-oper.yang"
    - or paths containing "/oper/"
    """
    ref = module_source_ref(mod)
    r = ref.replace("\\", "/")
    return ("-oper.yang" in r) or ("/oper/" in r)


# -------------------------- Safe Tree Walk Helpers ----------------------------


def iter_children(stmt: statements.Statement) -> Sequence[statements.Statement]:
    """
    Safe iterator over data children (i_children may not exist on leaves).
    """
    ch = getattr(stmt, "i_children", None)
    return ch if ch is not None else ()


def is_container_or_list(stmt: statements.Statement) -> bool:
    return stmt.keyword in ("container", "list")


def is_leaf(stmt: statements.Statement) -> bool:
    return stmt.keyword in ("leaf", "leaf-list")


def collect_leaf_names(stmt: statements.Statement) -> List[str]:
    """
    Collect distinct leaf/leaf-list names under a subtree.
    """
    names: set[str] = set()

    def _walk(s: statements.Statement) -> None:
        if is_leaf(s):
            if s.arg:
                names.add(str(s.arg))
            return
        for ch in iter_children(s):
            _walk(ch)

    _walk(stmt)
    return sorted(names)


# --------------------------- Semantic Tagging ---------------------------------


def guess_protocol_tag(module_name: str, path: str) -> Optional[str]:
    """
    Heuristic mapping from module/path to protocol tag.
    """
    s = (module_name + ":" + path).lower()

    mapping = [
        ("bgp", "bgp"),
        ("ospf", "ospf"),
        ("isis", "isis"),
        ("mpls", "mpls"),
        ("ldp", "ldp"),
        ("pim", "multicast"),
        ("igmp", "multicast"),
        ("rib", "routing"),
        ("route", "routing"),
        ("ifmgr", "interfaces"),
        ("ipv4-if", "interfaces"),
        ("ipv6-if", "interfaces"),
        ("interface", "interfaces"),
        ("qos", "qos"),
        ("acl", "acl"),
        ("infra", "platform"),
        ("platform", "platform"),
        ("bfd", "bfd"),
        ("tunnel", "tunnel"),
        ("ip-ma", "tunnel"),
        ("gre", "tunnel"),
        ("l2tun", "tunnel"),
        ("vxlan", "tunnel"),
        ("ethernet", "l2"),
        ("l2vpn", "l2"),
        ("bridge-domain", "l2"),
        ("mac", "l2"),
        ("arp", "neighbor"),
        ("neighbor", "neighbor"),
        ("ntp", "timing"),
        ("ptp", "timing"),
        ("clock", "timing"),
        ("aaa", "security"),
        ("crypto", "security"),
        ("ssh", "security"),
        ("telemetry", "telemetry"),
        ("mdt", "telemetry"),
        ("process", "system"),
        ("memory", "system"),
        ("cpu", "system"),
    ]

    for key, tag in mapping:
        if key in s:
            return tag
    return None


def guess_categories(path: str, leaf_names: List[str]) -> List[str]:
    """
    Heuristic 'what is this subtree about?' tags.
    """
    tags: set[str] = set()
    p = path.lower()
    leaves_str = " ".join(leaf_names).lower()

    def add_if(substr: str, tag: str, haystack: str) -> None:
        if substr in haystack:
            tags.add(tag)

    # Path-based hints
    add_if("neighbor", "neighbors", p)
    add_if("peer", "neighbors", p)
    add_if("process", "process", p)
    add_if("session", "sessions", p)
    add_if("interface", "interfaces", p)
    add_if("intf", "interfaces", p)
    add_if("rib", "routes", p)
    add_if("route", "routes", p)
    add_if("prefix", "prefixes", p)
    add_if("traffic", "traffic", p)
    add_if("counter", "stats", p)
    add_if("alarm", "alarms", p)
    add_if("event", "events", p)
    add_if("tunnel", "tunnels", p)
    add_if("bfd", "bfd", p)

    # Leaf-name-based hints
    if any(x in leaves_str for x in ["state", "status", "up", "down", "admin-state", "oper-state"]):
        tags.add("state")
    if any(x in leaves_str for x in ["packets", "octets", "bytes", "drops", "errors", "counter"]):
        tags.add("stats")
    if any(x in leaves_str for x in ["utilization", "usage", "percent"]):
        tags.add("utilization")

    # Fallback
    if not tags:
        tags.add("state")

    return sorted(tags)


# -------------------- Traversal & Catalog Building ----------------------------


def build_path(module_name: str, ancestors: List[str], current: str) -> str:
    """
    Canonical XR sensor path: Module:elem1/elem2/...
    """
    elems = ancestors + [current]
    return f"{module_name}:{'/'.join(elems)}"


def traverse_module(
    mod: statements.Statement,
    *,
    min_leaves: int = 1,
    max_depth: Optional[int] = None,
) -> List[Dict]:
    """
    Extract candidate sensor paths for a single module.

    Criteria:
      - Consider all container/list subtrees
      - Keep those with >= min_leaves distinct leaf names
      - Depth can be limited with max_depth
    """
    module_name = str(mod.arg)
    results: List[Dict] = []

    def _walk(stmt: statements.Statement, ancestors: List[str], depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return

        if is_container_or_list(stmt):
            leaf_names = collect_leaf_names(stmt)
            leaf_count = len(leaf_names)
            if leaf_count >= min_leaves:
                desc_stmt = stmt.search_one("description")
                desc = (desc_stmt.arg.strip() if desc_stmt and desc_stmt.arg else "")
                path = build_path(module_name, ancestors, str(stmt.arg))
                protocol_tag = guess_protocol_tag(module_name, path)
                categories = guess_categories(path, leaf_names)

                results.append(
                    {
                        "module": module_name,
                        "path": path,
                        "kind": stmt.keyword,  # "container" or "list"
                        "protocol_tag": protocol_tag,
                        "category": categories,
                        "leaf_names": leaf_names,
                        "leaf_count": leaf_count,
                        "description": desc,
                    }
                )

        # Recurse into child containers/lists
        if is_container_or_list(stmt) or stmt.keyword == "module":
            next_anc = ancestors + ([str(stmt.arg)] if stmt.keyword in ("container", "list") else [])
            for ch in iter_children(stmt):
                if is_container_or_list(ch):
                    _walk(ch, next_anc, depth + 1)

    # top-level data nodes
    for ch in iter_children(mod):
        if is_container_or_list(ch):
            _walk(ch, [], 1)

    return results


# ----------------------- Search Text Enrichment -------------------------------


def build_search_text(
    entry: Dict,
    *,
    max_leaves_in_text: int = 30,
    max_text_chars: int = 2000,
) -> str:
    """
    Build an embedding-friendly summary string for an entry.
    """
    parts: List[str] = []

    parts.append(f"Module: {entry['module']}")
    parts.append(f"Path: {entry['path']}")

    if entry.get("protocol_tag"):
        parts.append(f"Protocol: {entry['protocol_tag']}")

    if entry.get("category"):
        parts.append("Tags: " + ", ".join(entry["category"]))

    if entry.get("description"):
        parts.append("Description: " + entry["description"])

    leaf_count = int(entry.get("leaf_count", 0) or len(entry.get("leaf_names", [])))
    leaf_names: List[str] = list(entry.get("leaf_names", []))

    if leaf_names:
        sample = leaf_names[:max_leaves_in_text]
        parts.append(f"Example fields ({len(sample)}/{leaf_count}): " + ", ".join(sample))
        if leaf_count > max_leaves_in_text:
            parts.append(f"(+ {leaf_count - max_leaves_in_text} more fields not listed)")

    text = "\n".join(parts)
    if len(text) > max_text_chars:
        text = text[:max_text_chars]
    return text


def prepare_yang_entries(
    all_rows: List[Dict],
    *,
    max_leaves_in_text: int = 30,
    max_text_chars: int = 2000,
) -> List[Dict]:
    """
    Add id + search_text to each entry.
    """
    enriched: List[Dict] = []
    for i, row in enumerate(all_rows):
        r = dict(row)  # shallow copy (avoid mutating input)
        r["id"] = i
        r["search_text"] = build_search_text(
            r,
            max_leaves_in_text=max_leaves_in_text,
            max_text_chars=max_text_chars,
        )
        enriched.append(r)
    return enriched


# ----------------------------- JSONL Writing ----------------------------------


def write_jsonl(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------- Optional Stats ---------------------------------


def compute_text_length_stats(rows: List[Dict]) -> Dict[str, float]:
    lengths = [len(r.get("search_text", "") or "") for r in rows]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

    n = len(lengths)
    mean = sum(lengths) / n
    sorted_l = sorted(lengths)
    median = sorted_l[n // 2]
    var = sum((x - mean) ** 2 for x in lengths) / n
    std = var**0.5

    return {
        "count": float(n),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
    }


# --------------------------------- Main --------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build XR telemetry sensor catalog JSONL from YANG modules.")
    p.add_argument("--base-dir", type=Path, default=Path("data/yang/vendor/cisco/xr/701"),
                   help="Folder containing YANG files (recursively). Default: data/yang/vendor/cisco/xr/701")
    p.add_argument("--out-json", type=Path, default=Path("data/sensor_catalog.jsonl"),
                   help="Output JSONL path. Default: data/sensor_catalog.jsonl")
    p.add_argument("--min-leaves", type=int, default=2, help="Keep subtrees with at least this many leaf names. Default: 2")
    p.add_argument("--max-depth", type=int, default=None, help="Optional recursion depth limit.")
    # mutually exclusive flags to allow default = oper-only True, with an explicit --no-oper to disable
    group = p.add_mutually_exclusive_group()
    group.add_argument("--oper-only", dest="oper_only", action="store_true",
                       help="Only include operational modules (heuristic). (default)")
    group.add_argument("--no-oper", dest="oper_only", action="store_false",
                       help="Do not restrict to operational modules.")
    p.set_defaults(oper_only=True)

    p.add_argument("--max-leaves-in-text", type=int, default=30, help="Leaf names included in search_text.")
    p.add_argument("--max-text-chars", type=int, default=2000, help="Hard cap for search_text length.")
    p.add_argument("--stats", action="store_true", help="Print search_text length stats.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    base_dir: Path = args.base_dir.resolve()
    out_json: Path = args.out_json.resolve()

    logger.info("YANG base: %s", base_dir)
    logger.info("Output JSONL: %s", out_json)

    if not base_dir.exists():
        logger.error("Base dir does not exist: %s", base_dir)
        return 2

    _, modules = load_modules(base_dir)

    # Filter to operational modules if requested. This can be slow for many
    # modules (module_source_ref may access file metadata), so show progress
    # while we check each module.
    if args.oper_only:
        modules_used: List[statements.Statement] = []
        print(f"Filtering {len(modules)} parsed modules for operational ones...")
        for i, m in enumerate(modules, 1):
            try:
                if is_operational_module(m):
                    modules_used.append(m)
            except Exception as e:
                logger.debug("Error checking oper-only for module %s: %s", module_source_ref(m), e)

            if i % 100 == 0 or i == len(modules):
                print(f"  checked {i}/{len(modules)} modules, found {len(modules_used)} operational", flush=True)
    else:
        modules_used = modules

    logger.info("Modules parsed: %d", len(modules))
    logger.info("Modules used:   %d", len(modules_used))

    # Inform the user about module processing progress (flush so it's immediate)
    print(f"Processing {len(modules_used)} modules (this may take a while)...", flush=True)

    all_rows: List[Dict] = []
    for idx, mod in enumerate(modules_used, 1):
        mod_name = str(getattr(mod, 'arg', '') or '')
        if len(modules_used) <= 50 or idx % 10 == 0 or idx == len(modules_used):
            # Print sparse progress for large sets, or every module for small sets
            print(f"  [{idx}/{len(modules_used)}] module: {mod_name}", flush=True)

        rows = traverse_module(mod, min_leaves=args.min_leaves, max_depth=args.max_depth)
        all_rows.extend(rows)

    logger.info("Total catalog entries: %d", len(all_rows))

    all_rows = prepare_yang_entries(
        all_rows,
        max_leaves_in_text=args.max_leaves_in_text,
        max_text_chars=args.max_text_chars,
    )

    print(f"Writing {len(all_rows)} catalog entries to: {out_json} ...")
    write_jsonl(all_rows, out_json)
    print(f"Wrote catalog to: {out_json}")
    print(f"Entries: {len(all_rows)}")

    if args.stats:
        s = compute_text_length_stats(all_rows)
        print("search_text length stats:")
        print(
            f"  count={int(s['count'])}  min={int(s['min'])}  max={int(s['max'])}  "
            f"mean={s['mean']:.1f}  median={s['median']:.0f}  std={s['std']:.1f}"
        )

        none_count = sum(1 for r in all_rows if r.get("protocol_tag") is None)
        print(f"protocol_tag == None: {none_count} / {len(all_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
