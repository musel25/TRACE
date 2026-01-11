#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace.yang.importer import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPO_URL,
    DEFAULT_TMP_DIR,
    DEFAULT_XR_VERSION,
    import_yang_modules,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Cisco XR YANG modules via sparse checkout.")
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="YANG repo URL.")
    p.add_argument("--xr-version", default=DEFAULT_XR_VERSION, help="Cisco XR version folder.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output folder.")
    p.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR, help="Temporary sparse checkout folder.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import_yang_modules(
        repo_url=args.repo_url,
        xr_version=args.xr_version,
        output_root=args.output_root,
        tmp_dir=args.tmp_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
