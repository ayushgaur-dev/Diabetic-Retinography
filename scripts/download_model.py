"""Download and verify EfficientNetB0 fine-tuned model artifact for SIH26038.

Downloads the exact model from Hugging Face Hub:
  Repo: manudaza/retinal-triage-efficientnetb0
  File: efficientnetb0_finetuned_patched.keras
Local destination:
  models/efficientnetb0_finetuned_patched.keras

Verifies SHA-256 integrity:
  2a993b3affc96051aedc79d7b4c20f1538c6c9199d91032b22cfcc4642a24dd4
"""

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ID = "manudaza/retinal-triage-efficientnetb0"
FILENAME = "efficientnetb0_finetuned_patched.keras"
EXPECTED_SHA256 = "2a993b3affc96051aedc79d7b4c20f1538c6c9199d91032b22cfcc4642a24dd4"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def download_model(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / FILENAME
    tmp_path = target_dir / f"{FILENAME}.tmp"

    if target_path.is_file():
        print(f"Checking existing model file: {target_path}")
        existing_hash = compute_sha256(target_path)
        if existing_hash == EXPECTED_SHA256:
            print(f"Existing model SHA-256 matches expected checksum ({existing_hash}). Download skipped.")
            return target_path
        print(f"Existing model checksum mismatch ({existing_hash} != {EXPECTED_SHA256}). Re-downloading...")
        target_path.unlink()

    print(f"Downloading {FILENAME} from Hugging Face ({REPO_ID})...")

    # Method 1: huggingface_hub if available
    downloaded = False
    try:
        from huggingface_hub import hf_hub_download
        downloaded_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(target_dir),
        )
        if Path(downloaded_file).resolve() != target_path.resolve():
            shutil.copy2(downloaded_file, target_path)
        downloaded = True
        print("Downloaded via huggingface_hub.")
    except Exception as e:
        print(f"huggingface_hub download attempt failed ({e}); trying direct streaming download...")

    # Method 2: Direct HTTP streaming download with redirect handling
    if not downloaded:
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"
        req = urllib.request.Request(url, headers={"User-Agent": "SIH26038-Deployment/1.0"})
        if tmp_path.exists():
            tmp_path.unlink()
        with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=1024 * 1024)
        tmp_path.replace(target_path)
        print("Downloaded via direct HTTPS stream.")

    # Verify SHA-256
    print("Verifying downloaded file checksum...")
    downloaded_hash = compute_sha256(target_path)
    if downloaded_hash != EXPECTED_SHA256:
        if target_path.exists():
            target_path.unlink()
        raise ValueError(
            f"SHA-256 checksum verification failed!\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Got:      {downloaded_hash}"
        )

    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"Model successfully verified: {target_path} ({file_size_mb:.2f} MB)")
    print(f"SHA-256: {downloaded_hash}")
    return target_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    try:
        download_model(models_dir)
        sys.exit(0)
    except Exception as err:
        print(f"ERROR: Model download failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
