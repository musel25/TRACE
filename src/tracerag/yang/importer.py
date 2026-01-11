from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/YangModels/yang.git"
DEFAULT_XR_VERSION = "701"
DEFAULT_REMOTE_SUBDIR = f"vendor/cisco/xr/{DEFAULT_XR_VERSION}"
DEFAULT_OUTPUT_ROOT = Path("data/yang")
DEFAULT_TMP_DIR = Path(".tmp_yang_sparse")


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str | None:
    if capture:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
    subprocess.run(cmd, cwd=cwd, check=True)
    return None


def detect_default_branch(repo_url: str) -> str:
    try:
        out = run(["git", "ls-remote", "--symref", repo_url, "HEAD"], capture=True)
        if not out:
            return "main"
        for line in out.splitlines():
            if line.startswith("ref:") and "HEAD" in line:
                # "ref: refs/heads/main\tHEAD"
                ref = line.split()[1]
                return ref.split("/")[-1]
    except Exception:
        pass
    return "main"


def import_yang_modules(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    xr_version: str = DEFAULT_XR_VERSION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tmp_dir: Path = DEFAULT_TMP_DIR,
) -> Path:
    remote_subdir = f"vendor/cisco/xr/{xr_version}"
    output_dir = output_root / remote_subdir

    if output_dir.exists():
        print(f"[✓] Target already exists: {output_dir}")
        return output_dir

    default_branch = detect_default_branch(repo_url)
    print(f"[*] Remote default branch detected: {default_branch}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("[*] Initializing sparse git repo...")
    run(["git", "init", tmp_dir.as_posix()])

    run(["git", "remote", "add", "origin", repo_url], cwd=tmp_dir)
    run(["git", "config", "core.sparseCheckout", "true"], cwd=tmp_dir)

    sparse_file = tmp_dir / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(f"{remote_subdir}/\n")

    print(f"[*] Fetching only required folder: {remote_subdir}/")
    run(["git", "fetch", "--depth=1", "origin", default_branch], cwd=tmp_dir)
    run(["git", "checkout", "FETCH_HEAD"], cwd=tmp_dir)

    src = tmp_dir / remote_subdir
    if not src.exists():
        raise FileNotFoundError(
            f"Folder not found after checkout: {src}\n"
            f"Check that the folder exists upstream and that xr_version='{xr_version}'."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Copying to: {output_dir}")
    shutil.copytree(src, output_dir)

    print("[✓] Done.")
    shutil.rmtree(tmp_dir)
    return output_dir

