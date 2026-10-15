#!/usr/bin/env python3
"""
CricketCoach AI — Dataset Preparation and Manifest Generator
===========================================================
Prepares and validates cricket batting video datasets (e.g., CricketVision,
CricShot10, or custom local video folders).

Usage:
    # 1. Process and validate a local dataset folder (e.g. downloaded CricketVision):
    python ml_pipeline/prepare_dataset.py --input-dir /path/to/dataset --output-manifest dataset/manifest.csv

    # 2. Check current workspace dataset status:
    python ml_pipeline/prepare_dataset.py --status

    # 3. Create canonical folder structure for manual video collection:
    python ml_pipeline/prepare_dataset.py --create
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "ml_pipeline"))

from dataset_preparer import (
    CANONICAL_SHOT_CLASSES,
    CricketDatasetPreparer,
    SPLIT_DIRECTORY_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = project_root / "dataset" / "shots"


def create_dataset_structure():
    """Create the empty canonical dataset directory structure."""
    for shot in CANONICAL_SHOT_CLASSES:
        d = DEFAULT_DATASET_DIR / shot
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(
                f"Place labeled {shot.replace('_', ' ')} batting videos here.\n"
                "Supported formats: mp4, avi, mov, webm, mkv\n"
                "Minimum 5 videos recommended for training, 15+ for better accuracy.\n"
                "Ensure:\n"
                "  - The batsman is clearly visible full-body\n"
                "  - Good lighting\n"
                "  - Side-on or front-on angle preferred\n"
                "  - Video length: 2-10 seconds per shot\n"
            )

    print(f"Canonical dataset structure created at: {DEFAULT_DATASET_DIR}")
    print("\nNext steps:")
    print("  1. Add videos to each shot folder, or point to an external dataset with --input-dir")
    print("  2. Run verification: python ml_pipeline/prepare_dataset.py --status")
    print("  3. Run training:     python ml_pipeline/train_model.py")


def report_dataset():
    """Inspect and report the current workspace dataset status."""
    print("=== CricketCoach AI Dataset Status ===")
    print(f"Checking directory: {DEFAULT_DATASET_DIR}")

    if not DEFAULT_DATASET_DIR.exists():
        print("\n✗ Dataset directory does not exist yet.")
        print("  Run with --create to generate the folder structure, or use --input-dir to scan an existing dataset.")
        return

    # Check for manifest first
    manifest_candidates = list(project_root.glob("**/cricket_dataset_manifest.csv")) + list(
        project_root.glob("**/manifest.csv")
    )
    if manifest_candidates:
        print(f"\n✓ Found prepared manifest: {manifest_candidates[0]}")

    total = 0
    class_dirs = []
    for path in sorted(DEFAULT_DATASET_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.lower() in SPLIT_DIRECTORY_NAMES:
            class_dirs.extend(child for child in sorted(path.iterdir()) if child.is_dir())
        else:
            class_dirs.append(path)
    print("\nShot Class Breakdown:")
    if not class_dirs:
        print("  No class folders found.")
    for class_dir in class_dirs:
        videos = [
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v")
        ]
        status = "✓" if len(videos) >= 5 else ("⚠" if len(videos) > 0 else "✗")
        print(f"  {status} {class_dir.name:20s}: {len(videos):3d} videos")
        total += len(videos)

    print(f"\nTotal videos found in shots/ folders: {total}")
    if total < 10:
        print("\n⚠ Notice: Fewer than 10 total videos found.")
        print("  Training requires at least 5 usable videos per target class and a leak-safe split.")
        print("  See docs/DATASET_GUIDE.md for instructions on using CricketVision or other datasets.")
    else:
        print("\n✓ Sufficient data found. Ready to train: python ml_pipeline/train_model.py")


def main():
    parser = argparse.ArgumentParser(
        description="CricketCoach AI — Dataset Preparation and Validation Tool"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to dataset directory containing videos and annotation files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where manifest CSV and summary report should be saved",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Optional specific output path for manifest CSV (e.g. dataset/manifest.csv)",
    )
    parser.add_argument(
        "--train-val-test",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        help="Split ratio for train, validation, and test (default: 0.70 0.15 0.15)",
    )
    parser.add_argument(
        "--label-mapping",
        type=str,
        default=None,
        help="Optional JSON object mapping original labels to explicit target labels",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create canonical empty dataset folder structure in workspace",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report status of dataset currently present in workspace",
    )

    args = parser.parse_args()

    if args.create:
        create_dataset_structure()
        return

    if args.status:
        report_dataset()
        return

    # If input-dir is provided, run full dataset preparation and validation
    input_path = args.input_dir
    if not input_path:
        # If dataset/ exists, default to scanning it; otherwise show status
        if DEFAULT_DATASET_DIR.exists() or (project_root / "dataset").exists():
            input_path = str(project_root / "dataset")
        else:
            report_dataset()
            return

    output_dir = args.output_dir or (project_root / "dataset")
    label_mapping = None
    if args.label_mapping:
        with open(args.label_mapping, "r", encoding="utf-8") as mapping_file:
            label_mapping = json.load(mapping_file)
        if not isinstance(label_mapping, dict):
            parser.error("--label-mapping must reference a JSON object of source: target labels")

    preparer = CricketDatasetPreparer(
        input_dir=input_path,
        output_dir=output_dir,
        train_val_test_ratio=tuple(args.train_val_test),
        label_mapping=label_mapping,
    )

    logger.info("Starting dataset preparation and OpenCV validation...")
    manifest_path, summary = preparer.run()

    print("\n" + "=" * 60)
    print(" CRICKETCOACH AI — DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Scanned Directory: {summary['dataset_directory']}")
    print(f"Total Video Clips: {summary['total_clips']}")
    print(f"Valid Videos:      {summary['valid_clips']}")
    print(f"Invalid Videos:    {summary['invalid_clips']}")
    print(f"Duplicates:        {summary['duplicate_clips']}")
    print(f"Unlabelled Clips:  {summary['unlabelled_clips']}")
    print("-" * 60)
    print("Model Target Class Distribution:")
    for shot, cnt in summary["target_class_distribution"].items():
        print(f"  {shot:20s}: {cnt:4d} clips")
    print("Usable videos by target class:")
    for shot, cnt in summary["usable_class_distribution"].items():
        print(f"  {shot:20s}: {cnt:4d} videos")
    if summary["label_mapping"]:
        print(
            "Explicit mapping reduced classes: "
            f"{summary['class_mapping_reduced_classes']} "
            f"({summary['original_class_count']} original -> {summary['target_class_count']} target)"
        )
    print("-" * 60)
    print(f"Imbalance Ratio (Max/Min): {summary['class_imbalance']['imbalance_ratio_max_to_min']}x")
    print(f"Shannon Entropy (norm):    {summary['class_imbalance']['normalized_shannon_entropy']}")
    print(f"Split Breakdown:           {summary['split_distribution']}")
    print("=" * 60)
    print(f"Manifest CSV saved to:      {manifest_path}")
    print(f"Summary JSON saved to:      {output_dir}/manifest_summary.json")
    print(f"Markdown report saved to:   {output_dir}/dataset_validation_report.md")
    print("=" * 60)
    print("\nTo train the classifier with this manifest, run:")
    print(f"  python ml_pipeline/train_model.py --manifest {manifest_path}\n")


if __name__ == "__main__":
    main()
