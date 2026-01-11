#!/usr/bin/env python3
"""
build_sensor_catalog.py

Build a JSONL sensor catalog from Cisco IOS XR YANG modules (pyang).

Why this version is better
-------------------------
1) Much better "domain/protocol" tagging:
   - Classifies primarily by MODULE FAMILY (e.g., ipv4-bgp-oper => bgp)
   - Avoids false positives like infra-xtc-oper paths containing ".../bgp"
2) Faster traversal:
   - Memoizes subtree leaf sets to avoid repeated O(n^2) scanning
3) Adds metadata useful for deterministic filtering:
   - module_family, prefix, namespace, revision
4) Produces debug-friendly fields:
   - domain_confidence, domain_reasons

Output schema (superset of your current one)
--------------------------------------------
{
  "id": int,
  "module": str,
  "module_family": str,
  "prefix": str | null,
  "namespace": str | null,
  "revision": str | null,

  "path": str,
  "kind": "container" | "list",
  "key_leaves": list[str],

  "protocol_tag": str | null,      # kept for backward compatibility
  "domain": str | null,            # recommended new field
  "domain_confidence": float,      # 0..1
  "domain_reasons": list[str],     # why we tagged it that way

  "category": list[str],
  "leaf_names": list[str],
  "leaf_count": int,
  "description": str,
  "search_text": str
}

Usage
-----
python scripts/build_sensor_catalog.py \
  --base-dir data/yang/vendor/cisco/xr/701 \
  --out-json data/sensor_catalog.jsonl \
  --min-leaves 2 \
  --oper-only \
  -v

Notes
-----
- Requires: pyang
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyang import context, repository, statements

# ----------------------------- Logging ----------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ----------------------------- YANG Loading -----------------------------------

def load_modules(base_dir: Path) -> Tuple[context.Context, List[statements.Statement]]:
    repo = repository.FileRepository(str(base_dir))
    ctx = context.Context(repo)

    modules: List[statements.Statement] = []
    yang_files = sorted(base_dir.rglob("*.yang"))
    logger.info("Found %d .yang files under %s", len(yang_files), base_dir)
    print(f"Parsing {len(yang_files)} .yang files under {base_dir} ...")

    for i, p in enumerate(yang_files, 1):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            mod = ctx.add_module(p.name, text)
            if mod is not None and getattr(mod, "keyword", None) == "module":
                modules.append(mod)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", p, e)

        if i % 50 == 0 or i == len(yang_files):
            print(f"  parsed {i}/{len(yang_files)} files", flush=True)
    print("", flush=True)

    try:
        print("Validating parsed modules with pyang (this can take a while)...", flush=True)
        ctx.validate()
    except Exception as e:
        logger.warning("pyang ctx.validate() raised: %s", e)
        print(f"pyang validation raised an exception: {e}", flush=True)

    return ctx, modules


def module_source_ref(mod: statements.Statement) -> str:
    try:
        if mod.pos is not None and mod.pos.ref:
            return str(mod.pos.ref)
    except Exception:
        pass
    return str(mod.arg or "")


def is_operational_module(mod: statements.Statement) -> bool:
    ref = module_source_ref(mod).replace("\\", "/").lower()
    name = str(getattr(mod, "arg", "") or "").lower()
    # Common IOS XR op naming: *-oper.yang
    if "-oper.yang" in ref or name.endswith("-oper"):
        return True
    # Some trees use /oper/ in repo layout
    if "/oper/" in ref:
        return True
    return False


# -------------------------- Safe Tree Walk Helpers ----------------------------

def iter_children(stmt: statements.Statement) -> Sequence[statements.Statement]:
    ch = getattr(stmt, "i_children", None)
    return ch if ch is not None else ()


def is_container_or_list(stmt: statements.Statement) -> bool:
    return stmt.keyword in ("container", "list")


def is_leaf(stmt: statements.Statement) -> bool:
    return stmt.keyword in ("leaf", "leaf-list")


# ----------------------------- Metadata ---------------------------------------

def get_first_arg(stmt: statements.Statement, keyword: str) -> Optional[str]:
    try:
        sub = stmt.search_one(keyword)
        if sub is not None and getattr(sub, "arg", None):
            return str(sub.arg).strip()
    except Exception:
        return None
    return None


def extract_module_meta(mod: statements.Statement) -> Dict[str, Optional[str]]:
    prefix = get_first_arg(mod, "prefix")
    namespace = get_first_arg(mod, "namespace")

    # revision is a bit special: there can be multiple, take the latest (lexicographically works for YYYY-MM-DD)
    revs: List[str] = []
    try:
        for r in getattr(mod, "substmts", []) or []:
            if getattr(r, "keyword", None) == "revision" and getattr(r, "arg", None):
                revs.append(str(r.arg).strip())
    except Exception:
        pass
    revision = sorted(revs)[-1] if revs else None

    return {"prefix": prefix, "namespace": namespace, "revision": revision}


def module_family(module_name: str) -> str:
    """
    Compact family string used for filtering/routing.
    Examples:
      Cisco-IOS-XR-ipv4-bgp-oper -> ipv4-bgp-oper
      Cisco-IOS-XR-infra-xtc-oper -> infra-xtc-oper
    """
    m = module_name.strip()
    m = m.replace("Cisco-IOS-XR-", "")
    return m.lower()


# --------------------------- Domain Tagging -----------------------------------
# The goal: avoid "bgp appears somewhere" => domain=bgp.
# Rule: module family dominates. Path tokens are a fallback.

_DOMAIN_RULES = [
    # Strong protocol modules
    (re.compile(r"(?:^|-)ipv4-bgp-oper$"), "bgp", 1.00, "module_family=ipv4-bgp-oper"),
    (re.compile(r"(?:^|-)ipv6-bgp-oper$"), "bgp", 1.00, "module_family=ipv6-bgp-oper"),
    (re.compile(r"(?:^|-)bgp-oper$"), "bgp", 0.95, "module_family=bgp-oper"),
    (re.compile(r"(?:^|-)ospf(?:v3)?-oper$"), "ospf", 1.00, "module_family=ospf-oper"),
    (re.compile(r"(?:^|-)isis-?oper$"), "isis", 1.00, "module_family=isis-oper"),
    (re.compile(r"(?:^|-)mpls-.*-oper$"), "mpls", 0.95, "module_family=mpls-*"),
    (re.compile(r"(?:^|-)ldp-?oper$"), "ldp", 1.00, "module_family=ldp-oper"),
    (re.compile(r"(?:^|-)bfd-?oper$"), "bfd", 1.00, "module_family=bfd-oper"),
    (re.compile(r"(?:^|-)ifmgr-?oper$"), "interfaces", 1.00, "module_family=ifmgr-oper"),
    (re.compile(r"(?:^|-)ipv[46]-if-?oper$"), "interfaces", 1.00, "module_family=ipv*-if-oper"),
    (re.compile(r"(?:^|-)qos-?oper$"), "qos", 1.00, "module_family=qos-oper"),
    (re.compile(r"(?:^|-)acl-?oper$"), "acl", 1.00, "module_family=acl-oper"),
    (re.compile(r"(?:^|-)ethernet-span-?oper$"), "span", 1.00, "module_family=ethernet-span-oper"),

    # PCE / XTC / TE: must NOT become "bgp" just because subpaths mention it
    (re.compile(r"(?:^|-)infra-xtc-?oper$"), "pce", 1.00, "module_family=infra-xtc-oper"),
    (re.compile(r"(?:^|-)pce-.*-oper$"), "pce", 0.95, "module_family=pce-*"),
    (re.compile(r"(?:^|-)segment-routing-.*-oper$"), "sr", 0.95, "module_family=segment-routing-*"),
]

_PATH_FALLBACK = [
    (re.compile(r"(?:^|/|:)bgp(?:/|$)"), "bgp"),
    (re.compile(r"(?:^|/|:)ospf(?:/|$)"), "ospf"),
    (re.compile(r"(?:^|/|:)isis(?:/|$)"), "isis"),
    (re.compile(r"(?:^|/|:)mpls(?:/|$)"), "mpls"),
    (re.compile(r"(?:^|/|:)ldp(?:/|$)"), "ldp"),
    (re.compile(r"(?:^|/|:)interface(?:/|$)"), "interfaces"),
]


def classify_domain(module_name: str, path: str) -> Tuple[Optional[str], float, List[str]]:
    mf = module_family(module_name)
    reasons: List[str] = []

    for rx, dom, conf, reason in _DOMAIN_RULES:
        if rx.search(mf):
            reasons.append(reason)
            return dom, conf, reasons

    # fallback: path token match (lower confidence)
    p = path.lower()
    for rx, dom in _PATH_FALLBACK:
        if rx.search(p):
            reasons.append("path_fallback_match")
            return dom, 0.55, reasons

    return None, 0.0, reasons


def protocol_tag_compat(domain: Optional[str]) -> Optional[str]:
    """
    Keep your 'protocol_tag' field, but make it consistent and less noisy.
    """
    if domain is None:
        return None
    # Backward-compat mapping (your earlier tag set)
    mapping = {
        "bgp": "bgp",
        "ospf": "ospf",
        "isis": "isis",
        "mpls": "mpls",
        "ldp": "ldp",
        "bfd": "bfd",
        "interfaces": "interfaces",
        "acl": "acl",
        "qos": "qos",
        "pce": "pce",
        "sr": "sr",
        "span": "l2",  # you used l2 for ethernet-span
    }
    return mapping.get(domain, domain)


# --------------------------- Category Tagging ---------------------------------

_CATEGORY_RULES = [
    # (pattern, tag)
    (re.compile(r"(?:^|/|:)neighbor(s)?(?:/|$)"), "neighbors"),
    (re.compile(r"(?:^|/|:)peer(s)?(?:/|$)"), "neighbors"),
    (re.compile(r"(?:^|/|:)session(s)?(?:/|$)"), "sessions"),
    (re.compile(r"(?:^|/|:)process(?:/|$)"), "process"),
    (re.compile(r"(?:^|/|:)statistics|counter|counters|stats"), "stats"),
    (re.compile(r"(?:^|/|:)route(s)?|rib|prefix(es)?"), "routes"),
    (re.compile(r"(?:^|/|:)interface|intf"), "interfaces"),
    (re.compile(r"(?:^|/|:)topology|node(s)?|link(s)?"), "topology"),
    (re.compile(r"(?:^|/|:)policy|route-policy"), "policy"),
    (re.compile(r"(?:^|/|:)alarm|event"), "events"),
]


def guess_categories(path: str, leaf_names: List[str], description: str) -> List[str]:
    tags: set[str] = set()
    p = path.lower()
    leaves = " ".join(leaf_names).lower()
    desc = (description or "").lower()

    hay = f"{p} {desc} {leaves}"

    for rx, tag in _CATEGORY_RULES:
        if rx.search(hay):
            tags.add(tag)

    # State-ish / stats-ish leaf clues
    if re.search(r"\b(state|status|admin-state|oper-state|up|down)\b", hay):
        tags.add("state")
    if re.search(r"\b(packets|octets|bytes|drops|errors|counter)\b", hay):
        tags.add("stats")
    if re.search(r"\b(utilization|usage|percent|rate|bps|pps)\b", hay):
        tags.add("utilization")

    if not tags:
        tags.add("state")

    return sorted(tags)


# -------------------- Traversal & Catalog Building ----------------------------

def build_path(module_name: str, ancestors: List[str], current: str) -> str:
    elems = ancestors + [current]
    return f"{module_name}:{'/'.join(elems)}"


def list_keys(stmt: statements.Statement) -> List[str]:
    """
    Extract list keys if present.
    """
    try:
        k = stmt.search_one("key")
        if k is None or not getattr(k, "arg", None):
            return []
        # key arg is "k1 k2 ..."
        return [x.strip() for x in str(k.arg).split() if x.strip()]
    except Exception:
        return []


def traverse_module(
    mod: statements.Statement,
    *,
    min_leaves: int = 1,
    max_depth: Optional[int] = None,
    max_leaf_names_store: int = 200,
) -> List[Dict]:
    module_name = str(mod.arg)
    results: List[Dict] = []

    # Memoize subtree leaf sets per statement object-id
    leaf_memo: Dict[int, List[str]] = {}

    def subtree_leaf_names(stmt: statements.Statement) -> List[str]:
        sid = id(stmt)
        if sid in leaf_memo:
            return leaf_memo[sid]

        names: set[str] = set()

        def _walk(s: statements.Statement) -> None:
            if is_leaf(s):
                if getattr(s, "arg", None):
                    names.add(str(s.arg))
                return
            for ch in iter_children(s):
                _walk(ch)

        _walk(stmt)
        out = sorted(names)
        leaf_memo[sid] = out
        return out

    def _walk(stmt: statements.Statement, ancestors: List[str], depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return

        if is_container_or_list(stmt):
            leaf_names_full = subtree_leaf_names(stmt)
            leaf_count = len(leaf_names_full)

            if leaf_count >= min_leaves:
                desc = get_first_arg(stmt, "description") or ""
                path = build_path(module_name, ancestors, str(stmt.arg))

                dom, conf, reasons = classify_domain(module_name, path)
                proto = protocol_tag_compat(dom)
                cats = guess_categories(path, leaf_names_full, desc)

                # cap stored leaf names (but keep leaf_count)
                leaf_names_store = leaf_names_full[:max_leaf_names_store]

                results.append(
                    {
                        "module": module_name,
                        "module_family": module_family(module_name),
                        **extract_module_meta(mod),
                        "path": path,
                        "kind": stmt.keyword,
                        "key_leaves": list_keys(stmt) if stmt.keyword == "list" else [],
                        "protocol_tag": proto,  # backward compatible
                        "domain": dom,
                        "domain_confidence": float(conf),
                        "domain_reasons": reasons,
                        "category": cats,
                        "leaf_names": leaf_names_store,
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
    parts: List[str] = []
    parts.append(f"Module: {entry['module']}")
    parts.append(f"Family: {entry.get('module_family','')}")
    parts.append(f"Path: {entry['path']}")

    if entry.get("domain"):
        parts.append(f"Domain: {entry['domain']} (conf={entry.get('domain_confidence',0):.2f})")

    if entry.get("protocol_tag"):
        parts.append(f"ProtocolTag: {entry['protocol_tag']}")

    if entry.get("category"):
        parts.append("Category: " + ", ".join(entry["category"]))

    if entry.get("description"):
        parts.append("Description: " + entry["description"])

    keys = entry.get("key_leaves") or []
    if keys:
        parts.append("ListKeys: " + ", ".join(keys))

    leaf_count = int(entry.get("leaf_count", 0) or len(entry.get("leaf_names", [])))
    leaf_names: List[str] = list(entry.get("leaf_names", []))

    if leaf_names:
        sample = leaf_names[:max_leaves_in_text]
        parts.append(f"Fields ({len(sample)}/{leaf_count}): " + ", ".join(sample))
        if leaf_count > max_leaves_in_text:
            parts.append(f"(+ {leaf_count - max_leaves_in_text} more)")

    # Include namespace/prefix lightly (helps disambiguation for retrieval)
    if entry.get("prefix"):
        parts.append(f"Prefix: {entry['prefix']}")
    if entry.get("namespace"):
        parts.append(f"Namespace: {entry['namespace']}")
    if entry.get("revision"):
        parts.append(f"Revision: {entry['revision']}")

    text = "\n".join(parts)
    if len(text) > max_text_chars:
        text = text[:max_text_chars]
    return text


def prepare_entries(
    all_rows: List[Dict],
    *,
    max_leaves_in_text: int = 30,
    max_text_chars: int = 2000,
) -> List[Dict]:
    enriched: List[Dict] = []
    for i, row in enumerate(all_rows):
        r = dict(row)
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


# --------------------------------- Orchestration ------------------------------

def build_catalog(
    *,
    base_dir: Path,
    out_json: Path,
    min_leaves: int = 2,
    max_depth: Optional[int] = None,
    oper_only: bool = True,
    max_leaves_in_text: int = 30,
    max_text_chars: int = 2000,
    max_leaf_names_store: int = 200,
) -> int:
    if not base_dir.exists():
        logger.error("Base dir does not exist: %s", base_dir)
        return 2

    _, modules = load_modules(base_dir)

    if oper_only:
        modules_used: List[statements.Statement] = []
        print(f"Filtering {len(modules)} parsed modules for operational ones...")
        for i, m in enumerate(modules, 1):
            if is_operational_module(m):
                modules_used.append(m)
            if i % 100 == 0 or i == len(modules):
                print(f"  checked {i}/{len(modules)} modules, kept {len(modules_used)}", flush=True)
    else:
        modules_used = modules

    print(f"Processing {len(modules_used)} modules (this may take a while)...", flush=True)

    all_rows: List[Dict] = []
    for idx, mod in enumerate(modules_used, 1):
        mod_name = str(getattr(mod, "arg", "") or "")
        if len(modules_used) <= 50 or idx % 10 == 0 or idx == len(modules_used):
            print(f"  [{idx}/{len(modules_used)}] module: {mod_name}", flush=True)

        rows = traverse_module(
            mod,
            min_leaves=min_leaves,
            max_depth=max_depth,
            max_leaf_names_store=max_leaf_names_store,
        )
        all_rows.extend(rows)

    logger.info("Total catalog entries (pre-enrich): %d", len(all_rows))

    all_rows = prepare_entries(
        all_rows,
        max_leaves_in_text=max_leaves_in_text,
        max_text_chars=max_text_chars,
    )

    print(f"Writing {len(all_rows)} catalog entries to: {out_json} ...")
    write_jsonl(all_rows, out_json)
    print(f"Wrote catalog to: {out_json}")
    print(f"Entries: {len(all_rows)}")
    return 0
