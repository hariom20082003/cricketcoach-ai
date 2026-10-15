"""Tests for manifest labels, safe splitting, feature schema, and model metadata."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ml_pipeline"))
sys.path.insert(0, str(project_root / "backend"))

from app.ml.classifier import ShotClassifier
from app.ml.pose_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_NAME,
    FEATURE_SCHEMA_VERSION,
)
from ml_pipeline.train_model import check_dataset, check_manifest, train_and_evaluate


def test_directory_discovery_supports_cricshot_detailed_folders(tmp_path):
    shots = tmp_path / "cricshot10k" / "shots"
    for label in ("Cover Drive", "Reverse Sweep"):
        folder = shots / label
        folder.mkdir(parents=True)
        for idx in range(4):
            (folder / f"clip{idx}.mp4").write_bytes(b"fixture")

    report = check_dataset(str(shots.parent))
    assert report["classes"] == {"cover_drive": 4, "reverse_sweep": 4}
    assert not report["valid"]
    assert any("minimum 5" in error for error in report["errors"])


def test_missing_class_reports_a_clear_classification_error(tmp_path):
    shots = tmp_path / "shots"
    (shots / "Hook").mkdir(parents=True)
    report = check_dataset(str(shots))
    assert report["classes"] == {"hook": 0}
    assert any("At least two" in error for error in report["errors"])


def test_manifest_readiness_uses_target_label_and_excludes_duplicates(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["is_valid", "shot_label", "target_label", "validation_note", "split"]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ("drive", "hook"):
            for idx in range(5):
                writer.writerow({
                    "is_valid": "True", "shot_label": "legacy", "target_label": label,
                    "split": "train" if idx < 3 else "test",
                })
        writer.writerow({
            "is_valid": "True", "shot_label": "hook", "target_label": "hook",
            "validation_note": "duplicate_of:earlier.mp4",
        })

    report = check_manifest(str(manifest))
    assert report["valid"]
    assert report["classes"] == {"drive": 5, "hook": 5}
    assert report["examples_per_split"] == {"train": 6, "test": 4}


def test_manifest_readiness_rejects_unknown_provenance_and_missing_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        fields = ["is_valid", "shot_label", "source_id", "split"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ("hook", "pull"):
            for idx in range(5):
                writer.writerow({"is_valid": "True", "shot_label": label, "split": "unassigned"})

    report = check_manifest(str(manifest))
    assert not report["valid"]
    assert any("Unknown provenance" in error for error in report["errors"])


def test_training_rejects_insufficient_samples_and_unknown_provenance(tmp_path):
    X = np.zeros((8, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array([0] * 4 + [1] * 4, dtype=np.int32)
    with pytest.raises(ValueError, match="at least 5"):
        train_and_evaluate(X, y, str(tmp_path), ["hook", "pull"], splits=["train"] * 6 + ["test"] * 2)

    X = np.zeros((10, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    with pytest.raises(ValueError, match="Unknown provenance"):
        train_and_evaluate(X, y, str(tmp_path), ["hook", "pull"], source_ids=[""] * 10)


def test_training_metadata_and_backend_loader_share_feature_schema(tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, len(FEATURE_NAMES))).astype(np.float32)
    X[5:, 0] += 4
    y = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    splits = ["train"] * 3 + ["test"] * 2 + ["train"] * 3 + ["test"] * 2

    metadata = train_and_evaluate(
        X,
        y,
        str(tmp_path),
        ["hook", "pull"],
        splits=splits,
        label_mapping={"Cover Drive": "drive"},
    )
    assert metadata["feature_schema"] == {
        "name": FEATURE_SCHEMA_NAME,
        "version": FEATURE_SCHEMA_VERSION,
        "features": FEATURE_NAMES,
        "dimension": len(FEATURE_NAMES),
    }
    assert metadata["preprocessing"] == {"scaler": "StandardScaler", "feature_dtype": "float32"}
    assert metadata["class_labels"] == metadata["supported_shots"] == ["hook", "pull"]
    assert metadata["examples_per_split"]["train"] == 6
    assert metadata["examples_per_split"]["test"] == 4
    assert metadata["evaluation_warning"]

    classifier = ShotClassifier(model_dir=str(tmp_path))
    assert classifier.is_available
    assert classifier.supported_shots == ["hook", "pull"]
    prediction = classifier.predict(X[0])
    assert prediction["model_available"]
    assert prediction["shot_label"] in {"hook", "pull"}
    assert set(prediction["all_scores"]) == {"hook", "pull"}
    assert not classifier.predict(np.zeros(3))["model_available"]

    # A feature-schema mismatch disables the artifact instead of loading silently.
    metadata_path = tmp_path / "model_meta.json"
    saved = json.loads(metadata_path.read_text())
    saved["supported_shots"] = ["other", "classes"]
    metadata_path.write_text(json.dumps(saved))
    wrong_classes = ShotClassifier(model_dir=str(tmp_path))
    assert not wrong_classes.is_available

    saved["supported_shots"] = saved["class_labels"]
    saved["feature_schema"]["version"] += 1
    metadata_path.write_text(json.dumps(saved))
    incompatible = ShotClassifier(model_dir=str(tmp_path))
    assert not incompatible.is_available
