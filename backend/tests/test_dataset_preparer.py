"""
Unit tests for the CricketCoach AI Dataset Preparation & Validation Engine.
Tests:
- OpenCV video validation (synthetic video vs zero-byte corrupt file)
- CricketVision VIA JSON annotation parsing
- CSV metadata parsing
- Duplicate detection via SHA-256 content hash
- Group-aware train/val/test split generation (leakage prevention)
- Class imbalance metrics (Entropy, Imbalance ratio)
- Manifest CSV, summary JSON, and Markdown report export
"""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ml_pipeline"))
sys.path.insert(0, str(project_root / "backend"))

from ml_pipeline.dataset_preparer import (
    CANONICAL_SHOT_CLASSES,
    CricketDatasetPreparer,
    compute_file_hash,
    extract_source_id,
    normalize_detailed_label,
    normalize_shot_label,
    validate_video_with_opencv,
)


def create_synthetic_video(filepath: Path, width=64, height=64, fps=25, num_frames=10) -> bool:
    """Helper to generate a minimal valid MP4 video using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    if not out.isOpened():
        return False
    for i in range(num_frames):
        # Draw a synthetic colored frame
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 10, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return filepath.exists() and filepath.stat().st_size > 0


class TestVideoValidation:
    def test_valid_video_metadata(self, tmp_path):
        video_path = tmp_path / "valid_test.mp4"
        success = create_synthetic_video(video_path, width=128, height=128, fps=30, num_frames=15)
        assert success, "Failed to create synthetic test video"

        info = validate_video_with_opencv(video_path)
        assert info["is_valid"] is True
        assert info["validation_note"] == "ok"
        assert info["width"] == 128
        assert info["height"] == 128
        assert info["frame_count"] == 15
        assert info["fps"] == 30.0
        assert info["duration_seconds"] > 0.4

    def test_zero_byte_video_is_invalid(self, tmp_path):
        corrupt_path = tmp_path / "zero_byte.mp4"
        corrupt_path.write_bytes(b"")
        info = validate_video_with_opencv(corrupt_path)
        assert info["is_valid"] is False
        assert info["validation_note"] == "zero_byte_file"

    def test_nonexistent_file_is_invalid(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.mp4"
        info = validate_video_with_opencv(missing_path)
        assert info["is_valid"] is False
        assert info["validation_note"] == "file_not_found"

    def test_corrupt_content_is_invalid(self, tmp_path):
        corrupt_path = tmp_path / "corrupt.mp4"
        corrupt_path.write_bytes(b"random corrupt header bytes that are not video")
        info = validate_video_with_opencv(corrupt_path)
        assert info["is_valid"] is False


class TestLabelNormalization:
    def test_canonical_mappings(self):
        # Off Drive -> cover_drive
        canon, orig = normalize_shot_label("OffDrive")
        assert canon == "cover_drive"
        assert orig == "OffDrive"

        # On Drive -> straight_drive
        canon, orig = normalize_shot_label("On Drive")
        assert canon == "straight_drive"

        # Keep the legacy alias for compatibility; target_label stores the detailed class.
        canon, orig = normalize_shot_label("Hook")
        assert canon == "pull"

        # Square Drive / Cut -> cut
        canon, orig = normalize_shot_label("cut_shot")
        assert canon == "cut"

        # Block / Defense -> defensive
        canon, orig = normalize_shot_label("block")
        assert canon == "defensive"

        # Leg Glance / Flick -> flick
        canon, orig = normalize_shot_label("leg_glance")
        assert canon == "flick"

    def test_empty_label_does_not_invent(self):
        canon, orig = normalize_shot_label(None)
        assert canon is None
        assert orig is None

        canon, orig = normalize_shot_label("")
        assert canon is None
        assert orig is None


class TestSourceIdAndLeakagePrevention:
    def test_source_id_extraction(self):
        assert extract_source_id("match01_clip03.mp4") == "match01"
        assert extract_source_id("P1_V1_clip05.mp4") == "P1_V1"
        assert extract_source_id("IND_vs_AUS_2020_001.mp4") == "IND_vs_AUS_2020"
        assert extract_source_id("single_clip.mp4") == ""

    def test_detailed_folder_label_normalization_preserves_class(self):
        assert normalize_detailed_label("Straight Drive") == "straight_drive"
        assert normalize_detailed_label("Reverse Sweep") == "reverse_sweep"
        assert normalize_detailed_label("Hook") == "hook"


class TestDuplicateDetection:
    def test_duplicate_file_hash(self, tmp_path):
        f1 = tmp_path / "vid1.mp4"
        f2 = tmp_path / "vid2.mp4"
        create_synthetic_video(f1, num_frames=5)
        # Duplicate file contents
        shutil.copyfile(f1, f2)

        h1 = compute_file_hash(f1)
        h2 = compute_file_hash(f2)
        assert h1 == h2
        assert len(h1) == 64


class TestCricketVisionViaJsonParser:
    def test_parse_via_json(self, tmp_path):
        dataset_dir = tmp_path / "cricketvision_sample"
        dataset_dir.mkdir()

        # Create 2 synthetic videos
        v1 = dataset_dir / "stroke_01.mp4"
        v2 = dataset_dir / "stroke_02.mp4"
        create_synthetic_video(v1, num_frames=10)
        create_synthetic_video(v2, num_frames=10)

        # Create sample CricketVision VIA JSON
        via_json = {
            "project": {"pid": "cv_sample"},
            "attribute": {
                "1": {"aname": "TEMPORAL-SEGMENTS"},
                "2": {"aname": "Head_Score"},
                "7": {"aname": "Handedness"},
                "8": {"aname": "FootType"},
                "9": {"aname": "StrokeType"},
            },
            "_via_img_metadata": {
                "stroke_01.mp4": {
                    "filename": "stroke_01.mp4",
                    "z": [1.2, 2.5],
                    "av": {
                        "1": "Execution",
                        "2": "8",
                        "7": "0",  # Right
                        "8": "0",  # FrontFoot
                        "9": "0",  # OffDrive -> cover_drive
                    },
                },
                "stroke_02.mp4": {
                    "filename": "stroke_02.mp4",
                    "z": [2.0, 3.1],
                    "av": {
                        "1": "FollowThrough",
                        "2": "6",
                        "7": "1",  # Left
                        "8": "1",  # BackFoot
                        "9": "4",  # Hook remains distinct from Pull
                    },
                },
            },
        }

        with open(dataset_dir / "annotations.json", "w") as f:
            json.dump(via_json, f)

        preparer = CricketDatasetPreparer(input_dir=str(dataset_dir), output_dir=str(dataset_dir))
        manifest_path, summary = preparer.run()

        assert manifest_path.exists()
        assert summary["total_clips"] == 2
        assert summary["valid_clips"] == 2
        assert summary["invalid_clips"] == 0

        # Check parsed classes
        assert "cover_drive" in summary["class_distribution"]
        assert "pull" in summary["class_distribution"]
        assert "hook" in summary["target_class_distribution"]

        # Read manifest CSV and verify
        with open(manifest_path, "r") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        r1 = next(r for r in reader if r["filename"] == "stroke_01.mp4")
        assert r1["shot_label"] == "cover_drive"
        assert r1["target_label"] == "cover_drive"
        assert r1["original_label"] == "OffDrive"
        assert r1["batting_hand"] == "right"
        assert r1["foot_type"] == "front_foot"
        assert "overall_mean" in r1["technique_quality"]
        r2 = next(r for r in reader if r["filename"] == "stroke_02.mp4")
        assert r2["original_label"] == "Hook"
        assert r2["shot_label"] == "pull"  # Existing CricketVision integration field
        assert r2["target_label"] == "hook"  # Detailed model target


class TestCricShotFolderDiscovery:
    def test_split_paths_override_repeated_basenames_and_class_number_is_not_source(self, tmp_path):
        dataset_dir = tmp_path / "cricketshot"
        train_path = dataset_dir / "train" / "cover" / "cover_0001.avi"
        test_path = dataset_dir / "test" / "cover" / "cover_0001.avi"
        square_path = dataset_dir / "train" / "square_cut" / "square_cut_0015.avi"
        train_path.parent.mkdir(parents=True)
        test_path.parent.mkdir(parents=True)
        square_path.parent.mkdir(parents=True)
        assert create_synthetic_video(train_path, num_frames=5)
        assert create_synthetic_video(test_path, num_frames=7)
        assert create_synthetic_video(square_path, num_frames=8)

        manifest_path, _ = CricketDatasetPreparer(
            input_dir=str(dataset_dir), output_dir=str(tmp_path / "prepared")
        ).run()

        rows = {row["original_relative_path"]: row for row in csv.DictReader(manifest_path.open())}
        train = rows["train/cover/cover_0001.avi"]
        test = rows["test/cover/cover_0001.avi"]
        assert train["target_label"] == test["target_label"] == "cover"
        assert train["split"] == "train"
        assert test["split"] == "test"
        assert train["source_id"] == test["source_id"] == ""
        square = rows["train/square_cut/square_cut_0015.avi"]
        assert square["target_label"] == "square_cut"
        assert square["source_id"] == ""

    def test_detailed_labels_paths_and_usable_counts(self, tmp_path):
        dataset_dir = tmp_path / "cricshot10k"
        cover = dataset_dir / "shots" / "Cover Drive"
        hook = dataset_dir / "shots" / "Hook"
        reverse = dataset_dir / "shots" / "Reverse Sweep"
        for folder in (cover, hook, reverse):
            folder.mkdir(parents=True)
        assert create_synthetic_video(cover / "sample_a.mp4", num_frames=5)
        assert create_synthetic_video(hook / "sample_b.avi", num_frames=6)
        (reverse / "README.txt").write_text("class documentation")

        manifest_path, summary = CricketDatasetPreparer(
            input_dir=str(dataset_dir), output_dir=str(tmp_path / "prepared")
        ).run()

        rows = {row["original_label"]: row for row in csv.DictReader(manifest_path.open())}
        assert rows["Cover Drive"]["target_label"] == "cover_drive"
        assert rows["Hook"]["target_label"] == "hook"
        assert rows["Hook"]["shot_label"] == "hook"
        assert rows["Hook"]["original_relative_path"] == "shots/Hook/sample_b.avi"
        assert rows["Cover Drive"]["original_relative_path"] == "shots/Cover Drive/sample_a.mp4"
        assert summary["usable_class_distribution"] == {
            "cover_drive": 1,
            "hook": 1,
            "reverse_sweep": 0,
        }

    def test_explicit_label_mapping_only_changes_listed_labels(self, tmp_path):
        dataset_dir = tmp_path / "cricshot10k"
        for label, frames in (("Cover Drive", 5), ("Straight Drive", 6), ("Hook", 7)):
            folder = dataset_dir / "shots" / label
            folder.mkdir(parents=True)
            assert create_synthetic_video(folder / f"clip{frames}.mp4", num_frames=frames)

        manifest_path, summary = CricketDatasetPreparer(
            input_dir=str(dataset_dir),
            output_dir=str(tmp_path / "prepared"),
            label_mapping={"Cover Drive": "drive", "Straight Drive": "drive"},
        ).run()
        rows = list(csv.DictReader(manifest_path.open()))
        labels = {row["original_label"]: row["target_label"] for row in rows}
        assert labels == {"Cover Drive": "drive", "Straight Drive": "drive", "Hook": "hook"}
        assert next(row for row in rows if row["original_label"] == "Cover Drive")["shot_label"] == "cover_drive"
        assert summary["class_mapping_reduced_classes"] is True
        assert summary["original_class_count"] == 3
        assert summary["target_class_count"] == 2

    def test_manifest_counts_invalid_and_exact_duplicates_as_unusable(self, tmp_path):
        dataset_dir = tmp_path / "invalids" / "shots" / "Pull"
        dataset_dir.mkdir(parents=True)
        good = dataset_dir / "good.mp4"
        assert create_synthetic_video(good, num_frames=5)
        shutil.copyfile(good, dataset_dir / "duplicate.mp4")
        (dataset_dir / "empty.mp4").write_bytes(b"")
        (dataset_dir / "corrupt.mkv").write_bytes(b"not a video")

        _, summary = CricketDatasetPreparer(
            input_dir=str(tmp_path / "invalids"), output_dir=str(tmp_path / "prepared")
        ).run()
        assert summary["invalid_clips"] == 2
        assert summary["duplicate_clips"] == 1
        assert summary["usable_class_distribution"]["pull"] == 1


class TestClassImbalanceCalculation:
    def test_imbalance_metrics_computed(self, tmp_path):
        dataset_dir = tmp_path / "imbalance_dataset"
        dataset_dir.mkdir()

        # Create 3 cover drives, 1 pull shot
        shots = ["cover_drive", "cover_drive", "cover_drive", "pull"]
        for idx, shot in enumerate(shots):
            sdir = dataset_dir / "shots" / shot
            sdir.mkdir(parents=True, exist_ok=True)
            vpath = sdir / f"clip_{idx}.mp4"
            create_synthetic_video(vpath, num_frames=5)

        preparer = CricketDatasetPreparer(input_dir=str(dataset_dir), output_dir=str(dataset_dir))
        manifest_path, summary = preparer.run()

        assert summary["class_distribution"]["cover_drive"] == 3
        assert summary["class_distribution"]["pull"] == 1
        # Max / Min = 3 / 1 = 3.0x
        assert summary["class_imbalance"]["imbalance_ratio_max_to_min"] == 3.0
        # Entropy should be between 0 and 1
        assert 0.0 < summary["class_imbalance"]["normalized_shannon_entropy"] < 1.0


class TestGroupedSplitNoDataLeakage:
    def test_same_match_clips_never_split_across_train_and_test(self, tmp_path):
        dataset_dir = tmp_path / "match_dataset"
        dataset_dir.mkdir()

        # 4 clips from Match01, 4 clips from Match02, 4 clips from Match03
        for match_id in ("match01", "match02", "match03"):
            for clip_idx in range(4):
                vpath = dataset_dir / f"{match_id}_clip{clip_idx}.mp4"
                create_synthetic_video(vpath, num_frames=5)

        preparer = CricketDatasetPreparer(
            input_dir=str(dataset_dir),
            output_dir=str(dataset_dir),
            train_val_test_ratio=(0.60, 0.20, 0.20),
        )
        manifest_path, summary = preparer.run()

        with open(manifest_path, "r") as f:
            records = list(csv.DictReader(f))

        # Check each match: all clips of that match MUST have the exact same split!
        for match_id in ("match01", "match02", "match03"):
            match_splits = {r["split"] for r in records if r["source_id"] == match_id}
            assert len(match_splits) == 1, f"Data leakage detected for {match_id}: {match_splits}"
