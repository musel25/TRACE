## Catalog building (src/tracerag/catalog/builder.py)

Purpose: parse IOS XR YANG modules into a JSONL "sensor catalog" with
heuristic tags and a compact `search_text` field for embeddings.

### Output schema (per JSONL row)

```
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
```

### Key functions

`build_catalog(base_dir, out_json, min_leaves=2, max_depth=None, oper_only=True, max_leaves_in_text=30, max_text_chars=2000, stats=False) -> int`
- Input:
  - `base_dir` (Path): root containing YANG files.
  - `out_json` (Path): output JSONL path.
  - `min_leaves` (int): minimum leaf count per subtree.
  - `max_depth` (int | None): recursion depth cap.
  - `oper_only` (bool): filter to operational modules.
  - `max_leaves_in_text` (int): cap list of leaf names in `search_text`.
  - `max_text_chars` (int): hard cap for `search_text` length.
  - `stats` (bool): print text length stats.
- Output: int status code (0 for success).

`prepare_yang_entries(all_rows, max_leaves_in_text=30, max_text_chars=2000) -> List[Dict]`
- Adds `id` and `search_text` to each row.

`build_search_text(entry, max_leaves_in_text=30, max_text_chars=2000) -> str`
- Produces an embedding-friendly string from a single entry.

### Typical usage

```python
from pathlib import Path
from tracerag.catalog.builder import build_catalog

build_catalog(
    base_dir=Path("data/yang/vendor/cisco/xr/701"),
    out_json=Path("data/sensor_catalog.jsonl"),
    min_leaves=2,
    oper_only=True,
)
```
