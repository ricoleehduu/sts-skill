#!/usr/bin/env python3
"""
Multi-source data downloader for MICCAI STS Challenge Pre-Task data.

Download sources (tried in priority order):
  1. Huggingface  — dataset: Ricoooo/MICCAI-STS26-Challenge-Pre-Task
  2. Modelscope   — dataset: lizhii/MICCAI-STS26-Challenge-Pre-Task
  3. Google Drive — folder:  1lER9eIavr99g28aTO0kuxIcos_k9FBSx
  4. Baidu Netdisk — prints manual instructions (no API)

Usage:
    python scripts/download_data.py --task pretask-2026 --output ./data
    python scripts/download_data.py --task pretask-2026 --output ./data --source huggingface
    python scripts/download_data.py --task pretask-2026 --output ./data --source baidu
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "Ricoooo/MICCAI-STS26-Challenge-Pre-Task"
MS_DATASET_ID = "lizhii/MICCAI-STS26-Challenge-Pre-Task"
GDRIVE_FOLDER_ID = "1lER9eIavr99g28aTO0kuxIcos_k9FBSx"
BAIDU_URL = "https://pan.baidu.com/s/1U090bZnMGEJQaD3jwqaQuA"
BAIDU_CODE = "bm2u"

SUPPORTED_SOURCES = ("huggingface", "modelscope", "gdrive", "baidu")

EXPECTED_SUBDIRS = ("imgs", "masks")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".nii.gz", ".gz"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "INFO") -> None:
    """Print a formatted status message."""
    print(f"[{level}] {msg}", flush=True)


def _warn(msg: str) -> None:
    _log(msg, level="WARN")


def _err(msg: str) -> None:
    _log(msg, level="ERROR")


def _count_images(directory: Path) -> int:
    """Return the number of image files in *directory* (non-recursive)."""
    count = 0
    for f in directory.iterdir():
        if f.is_file():
            suffix = f.suffix.lower()
            if suffix == ".gz" and f.name.endswith(".nii.gz"):
                suffix = ".nii.gz"
            if suffix in IMAGE_EXTENSIONS:
                count += 1
    return count


def _find_zip_files(directory: Path) -> List[Path]:
    """Return all zip files in *directory* (non-recursive)."""
    return sorted(directory.glob("*.zip"))


def _extract_zips(directory: Path) -> None:
    """Extract all zip files found in *directory* into *directory* itself."""
    zips = _find_zip_files(directory)
    if not zips:
        _log("No zip files found to extract.")
        return

    for zpath in zips:
        _log(f"Extracting {zpath.name} ...")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(directory)
            _log(f"  Extracted: {zpath.name}")
            # Optionally remove the zip after extraction to save space.
            # Uncomment the next line if desired:
            # zpath.unlink()
        except zipfile.BadZipFile:
            _err(f"  Failed to extract {zpath.name}: bad zip file.")
        except Exception as exc:
            _err(f"  Failed to extract {zpath.name}: {exc}")


def _verify(output_dir: Path) -> bool:
    """Check that expected sub-directories exist and contain images."""
    _log("Verifying downloaded data ...")
    ok = True
    for sub in EXPECTED_SUBDIRS:
        sub_path = output_dir / sub
        if not sub_path.is_dir():
            _err(f"Expected directory not found: {sub_path}")
            ok = False
            continue
        n = _count_images(sub_path)
        if n == 0:
            _err(f"Directory exists but contains no images: {sub_path}")
            ok = False
        else:
            _log(f"  {sub}/  — {n} image(s) found.")
    return ok


# ---------------------------------------------------------------------------
# Source: Huggingface
# ---------------------------------------------------------------------------


def _download_huggingface(output_dir: Path) -> bool:
    """Download dataset from Hugging Face Hub. Returns True on success."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        _warn("huggingface_hub is not installed. Install with: pip install huggingface_hub")
        return False

    _log(f"Downloading from Hugging Face Hub: {HF_REPO_ID} ...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        _log("Hugging Face download complete.")
        return True
    except Exception as exc:
        _err(f"Hugging Face download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Source: Modelscope
# ---------------------------------------------------------------------------


def _download_modelscope(output_dir: Path) -> bool:
    """Download dataset from ModelScope. Returns True on success."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download  # type: ignore
    except ImportError:
        _warn("modelscope is not installed. Install with: pip install modelscope")
        return False

    _log(f"Downloading from ModelScope: {MS_DATASET_ID} ...")
    try:
        ms_snapshot_download(
            MS_DATASET_ID,
            local_dir=str(output_dir),
        )
        _log("ModelScope download complete.")
        return True
    except Exception as exc:
        _err(f"ModelScope download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Source: Google Drive
# ---------------------------------------------------------------------------


def _download_gdrive(output_dir: Path) -> bool:
    """Download folder from Google Drive via gdown. Returns True on success."""
    try:
        import gdown  # type: ignore[import-untyped]
    except ImportError:
        _warn("gdown is not installed. Install with: pip install gdown")
        return False

    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    _log(f"Downloading from Google Drive folder: {url} ...")
    try:
        gdown.download_folder(url, output=str(output_dir), quiet=False)
        _log("Google Drive download complete.")
        return True
    except Exception as exc:
        _err(f"Google Drive download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Source: Baidu Netdisk (manual instructions)
# ---------------------------------------------------------------------------


def _download_baidu(output_dir: Path) -> bool:
    """Print manual download instructions for Baidu Netdisk.

    There is no public API for Baidu Netdisk, so we simply guide the user.
    Returns False to indicate that automated download is not possible.
    """
    print()
    print("=" * 64)
    print("  Baidu Netdisk Download Instructions")
    print("=" * 64)
    print()
    print(f"  Link  : {BAIDU_URL}")
    print(f"  Code  : {BAIDU_CODE}")
    print()
    print(f"  After downloading, please place the data under:")
    print(f"    {output_dir.resolve()}")
    print()
    print("  The folder should contain 'imgs/' and 'masks/' sub-directories.")
    print("=" * 64)
    print()
    return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_source(source: Optional[str]) -> List[str]:
    """Return an ordered list of sources to try.

    If *source* is specified, return it as a single-element list after
    validating. Otherwise return the full priority list.
    """
    if source is None:
        return list(SUPPORTED_SOURCES)
    source = source.lower().strip()
    if source not in SUPPORTED_SOURCES:
        _err(f"Unknown source '{source}'. Choose from: {', '.join(SUPPORTED_SOURCES)}")
        sys.exit(1)
    return [source]


DOWNLOADERS = {
    "huggingface": _download_huggingface,
    "modelscope": _download_modelscope,
    "gdrive": _download_gdrive,
    "baidu": _download_baidu,
}


def run(task: str, output: str, source: Optional[str] = None) -> None:
    """Main entry point: download, extract, verify."""
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Task   : {task}")
    _log(f"Output : {output_dir}")

    sources = _resolve_source(source)

    downloaded = False
    for src in sources:
        _log(f"Trying source: {src}")
        fn = DOWNLOADERS[src]
        if fn(output_dir):
            downloaded = True
            break
        _warn(f"Source '{src}' failed or is unavailable, trying next ...")

    if not downloaded:
        _err("All download sources failed. Please download the data manually.")
        sys.exit(1)

    # --- Post-download steps ---------------------------------------------------

    _log("Post-download processing ...")

    # 1. Extract any zip files that were downloaded.
    _extract_zips(output_dir)

    # 2. Some sources place data inside a nested directory; flatten if needed.
    for sub in EXPECTED_SUBDIRS:
        nested = output_dir / task / sub
        target = output_dir / sub
        if nested.is_dir() and not target.is_dir():
            _log(f"Moving {nested} -> {target}")
            shutil.move(str(nested), str(target))

    # 3. Verify integrity.
    if _verify(output_dir):
        _log("Data downloaded and verified successfully.")
    else:
        _warn("Verification found issues. Please check the data manually.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download MICCAI STS Challenge Pre-Task data from multiple sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_data.py --task pretask-2026 --output ./data\n"
            "  python scripts/download_data.py --task pretask-2026 --output ./data --source huggingface\n"
        ),
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task identifier, e.g. pretask-2026",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--source", "-s",
        choices=SUPPORTED_SOURCES,
        default=None,
        help=(
            "Download source. If omitted, sources are tried in priority order: "
            "huggingface -> modelsope -> gdrive -> baidu"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(task=args.task, output=args.output, source=args.source)


if __name__ == "__main__":
    main()
