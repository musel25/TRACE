import subprocess
import shutil
from pathlib import Path

REPO_URL = "https://github.com/YangModels/yang.git"

# Choose the IOS-XR version folder you want (e.g., "711", "701", "7101", etc.)
XR_VERSION = "701"

# IMPORTANT: In the upstream repo, it's "vendor/..." (no leading "yang/")
REMOTE_SUBDIR = f"vendor/cisco/xr/{XR_VERSION}"

# Locally, you want it under data/yang/...
OUTPUT_DIR = Path(f"data/yang/{REMOTE_SUBDIR}")

TMP_DIR = Path(".tmp_yang_sparse")

def run(cmd, cwd=None, capture=False):
    if capture:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout
    subprocess.run(cmd, cwd=cwd, check=True)

def detect_default_branch(repo_url: str) -> str:
    try:
        out = run(["git", "ls-remote", "--symref", repo_url, "HEAD"], capture=True)
        for line in out.splitlines():
            if line.startswith("ref:") and "HEAD" in line:
                # "ref: refs/heads/main\tHEAD"
                ref = line.split()[1]
                return ref.split("/")[-1]
    except Exception:
        pass
    return "main"

def main():
    if OUTPUT_DIR.exists():
        print(f"[✓] Target already exists: {OUTPUT_DIR}")
        return

    default_branch = detect_default_branch(REPO_URL)
    print(f"[*] Remote default branch detected: {default_branch}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print("[*] Initializing sparse git repo...")
    run(["git", "init", TMP_DIR.as_posix()])

    run(["git", "remote", "add", "origin", REPO_URL], cwd=TMP_DIR)
    run(["git", "config", "core.sparseCheckout", "true"], cwd=TMP_DIR)

    sparse_file = TMP_DIR / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(f"{REMOTE_SUBDIR}/\n")

    print(f"[*] Fetching only required folder: {REMOTE_SUBDIR}/")
    run(["git", "fetch", "--depth=1", "origin", default_branch], cwd=TMP_DIR)
    run(["git", "checkout", "FETCH_HEAD"], cwd=TMP_DIR)

    src = TMP_DIR / REMOTE_SUBDIR
    if not src.exists():
        raise FileNotFoundError(
            f"Folder not found after checkout: {src}\n"
            f"Check that the folder exists upstream and that XR_VERSION='{XR_VERSION}'."
        )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Copying to: {OUTPUT_DIR}")
    shutil.copytree(src, OUTPUT_DIR)

    print("[✓] Done.")
    shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    main()
