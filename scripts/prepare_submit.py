#!/usr/bin/env python3
"""
Submission packager for MICCAI STS 2026 Codabench competitions.

Usage:
    python scripts/prepare_submit.py --masks <masks_dir> --task <task-name> [--output <output.zip>]

Examples:
    python scripts/prepare_submit.py --masks data/test/masks/ --task pretask-2026
    python scripts/prepare_submit.py --masks data/test/masks/ --task pretask-2026 --output my_submission.zip
"""

import argparse
import os
import struct
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Task registry — add new tasks here
# ---------------------------------------------------------------------------
TASK_CONFIG = {
    "pretask-2026": {
        "output_name": "task_pre_validation_data.zip",
        "archive_root": "data/test/masks",
        "expected_count": 50,
        "description": "Pre-Task 2026 (50 test masks)",
    },
    # "task1-2026": {
    #     "output_name": "task1_validation.zip",
    #     "archive_root": "data/test/masks",
    #     "expected_count": 120,
    #     "description": "Task 1 2026",
    # },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_png(filepath: str) -> bool:
    """Check whether *filepath* is a valid PNG by reading its magic bytes
    and verifying the IHDR chunk is parseable.

    Returns True for any file that at least looks like a real PNG image.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(24)
            if len(header) < 24:
                return False
            # PNG signature: 8 bytes
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                return False
            # IHDR chunk length (4 bytes, big-endian) should be 13
            ihdr_len = struct.unpack(">I", header[8:12])[0]
            if ihdr_len != 13:
                return False
            # IHDR chunk type
            if header[12:16] != b"IHDR":
                return False
            # Width and height (4 bytes each, big-endian), must be > 0
            width = struct.unpack(">I", header[16:20])[0]
            height = struct.unpack(">I", header[20:24])[0]
            if width == 0 or height == 0:
                return False
            return True
    except (OSError, struct.error):
        return False


def collect_png_files(masks_dir: Path):
    """Return a sorted list of *.png paths under *masks_dir* (non-recursive)."""
    return sorted(p for p in masks_dir.iterdir() if p.suffix.lower() == ".png")


def print_banner(text: str):
    width = max(len(text) + 4, 50)
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(msg: str):
    print(f"  -> {msg}")


def print_ok(msg: str):
    print(f"  [OK] {msg}")


def print_warn(msg: str):
    print(f"  [WARN] {msg}")


def print_error(msg: str):
    print(f"  [ERROR] {msg}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    masks_dir: Path = args.masks.resolve()
    task_name: str = args.task

    print_banner("STS 2026 Submission Packager")

    # ---- Validate task ----
    if task_name not in TASK_CONFIG:
        print_error(f"Unknown task: '{task_name}'")
        print_step(f"Supported tasks: {', '.join(TASK_CONFIG.keys())}")
        return 1

    config = TASK_CONFIG[task_name]
    print_step(f"Task: {config['description']}")

    # ---- Validate input directory ----
    if not masks_dir.exists():
        print_error(f"Masks directory does not exist: {masks_dir}")
        return 1
    if not masks_dir.is_dir():
        print_error(f"Masks path is not a directory: {masks_dir}")
        return 1

    # ---- Collect PNG files ----
    print_step(f"Scanning: {masks_dir}")
    png_files = collect_png_files(masks_dir)

    if not png_files:
        print_error("No PNG files found in the masks directory.")
        return 1

    print_step(f"Found {len(png_files)} PNG file(s)")

    # ---- Check for non-PNG image files that may have been mis-named ----
    non_png = [
        p.name
        for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() != ".png"
    ]
    if non_png:
        print_warn(
            f"{len(non_png)} non-PNG file(s) in directory (will be ignored): "
            + ", ".join(non_png[:10])
            + (" ..." if len(non_png) > 10 else "")
        )

    # ---- Validate PNG integrity ----
    print_step("Validating PNG files ...")
    bad_pngs = []
    for fp in png_files:
        if not is_valid_png(str(fp)):
            bad_pngs.append(fp.name)

    if bad_pngs:
        print_warn(
            f"{len(bad_pngs)} file(s) do not appear to be valid PNG images: "
            + ", ".join(bad_pngs[:10])
            + (" ..." if len(bad_pngs) > 10 else "")
        )
        print_warn("These files will still be included in the archive.")
    else:
        print_ok(f"All {len(png_files)} PNG files passed validation.")

    # ---- File count check ----
    expected = config.get("expected_count")
    if expected is not None:
        if len(png_files) == expected:
            print_ok(f"File count matches expected ({expected}).")
        elif abs(len(png_files) - expected) <= 5:
            print_warn(
                f"File count is {len(png_files)}, expected ~{expected}. "
                "Please double-check."
            )
        else:
            print_warn(
                f"File count is {len(png_files)}, expected ~{expected}. "
                "This looks suspicious — verify your masks directory."
            )

    # ---- Determine output path ----
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path.cwd() / config["output_name"]

    # ---- Package ----
    archive_prefix = config["archive_root"]  # e.g. "data/test/masks"
    print_step(f"Packaging into: {output_path}")
    print_step(f"Archive structure: {archive_prefix}/*.png")

    try:
        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in png_files:
                arcname = f"{archive_prefix}/{fp.name}"
                zf.write(str(fp), arcname)
    except OSError as exc:
        print_error(f"Failed to create zip archive: {exc}")
        return 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print_ok(f"Archive created: {output_path} ({size_mb:.2f} MB)")
    print_step(f"Contains {len(png_files)} file(s)")

    # ---- Summary ----
    print()
    print_banner("Done")
    print_step(f"Submission file ready: {output_path}")
    print_step("Upload this zip to your Codabench competition page.")
    print()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Package predicted masks into a Codabench submission zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported tasks:\n"
            + "\n".join(
                f"  {name:16s}  {cfg['description']}"
                for name, cfg in TASK_CONFIG.items()
            )
        ),
    )
    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Directory containing predicted mask PNG files.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),
        help="Target competition task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output zip file path. "
            "Defaults to the task-specific name in the current directory."
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
