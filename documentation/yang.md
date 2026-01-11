## YANG import (src/tracerag/yang/importer.py)

Purpose: import IOS XR YANG modules via git sparse checkout.

### Function

`import_yang_modules(repo_url=DEFAULT_REPO_URL, xr_version=DEFAULT_XR_VERSION, output_root=DEFAULT_OUTPUT_ROOT, tmp_dir=DEFAULT_TMP_DIR) -> Path`
- Input:
  - `repo_url` (str): source repo URL (default: YangModels/yang).
  - `xr_version` (str): IOS XR version folder (default: `701`).
  - `output_root` (Path): root folder to write under (default: `data/yang`).
  - `tmp_dir` (Path): temporary sparse checkout folder.
- Output: `Path` to the imported folder (e.g. `data/yang/vendor/cisco/xr/701`).
- Behavior:
  - Detects default branch.
  - Performs a sparse checkout for the versioned subdirectory.
  - Copies into `output_root` and removes the temporary checkout.

### Minimal usage

```python
from tracerag.yang.importer import import_yang_modules

out_dir = import_yang_modules(xr_version="701")
print(out_dir)
```
