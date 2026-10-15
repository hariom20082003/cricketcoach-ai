#!/usr/bin/env python3
"""
CricketCoach AI — Shot Classifier Training Pipeline
====================================================
Trains a scikit-learn Random Forest classifier on cricket batting shots.

Supports two data ingestion modes:
1. Manifest CSV mode (recommended):
       python ml_pipeline/train_model.py --manifest dataset/cricket_dataset_manifest.csv
   - Uses pre-validated videos and official splits
   - Uses source_id grouping to prevent data leakage across matches/recordings
2. Directory-based mode:
       python ml_pipeline/train_model.py --dataset-dir dataset/shots/

The trained model files are saved to backend/app/ml/models/ and activated automatically.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "ml_pipeline"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHOT_CLASSES = [
    "cover_drive",
    "straight_drive",
    "pull",
    "cut",
    "sweep",
    "flick",
    "defensive",
]
MIN_SAMPLES_PER_CLASS = 5
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")


def normalize_split(value: Optional[str]) -> str:
    value = (value or "unassigned").strip().lower()
    return {"training": "train", "validation": "val", "valid": "val", "testing": "test"}.get(value, value)

DEFAULT_DATASET_DIR = str(project_root / "dataset" / "shots")
DEFAULT_OUTPUT_DIR = str(project_root / "backend" / "app" / "ml" / "models")


def check_dataset(dataset_dir: str) -> dict:
    """Discover detailed class folders and report their video counts."""
    report = {"classes": {}, "total": 0, "valid": True, "errors": []}

    if not os.path.exists(dataset_dir):
        report["valid"] = False
        report["errors"].append(f"Dataset directory not found: {dataset_dir}")
        return report

    from dataset_preparer import SPLIT_DIRECTORY_NAMES, normalize_detailed_label

    root = Path(dataset_dir)
    if (root / "shots").is_dir():
        shots_root = root / "shots"
    elif root.name.lower() == "shots":
        shots_root = root
    else:
        shots_roots = sorted(path for path in root.rglob("shots") if path.is_dir())
        shots_root = shots_roots[0] if shots_roots else root

    class_dirs = []
    for child in sorted(shots_root.iterdir()) if shots_root.is_dir() else []:
        if not child.is_dir():
            continue
        if child.name.lower() in SPLIT_DIRECTORY_NAMES:
            class_dirs.extend(path for path in sorted(child.iterdir()) if path.is_dir())
        else:
            class_dirs.append(child)

    for class_dir in class_dirs:
        shot_class = normalize_detailed_label(class_dir.name)
        if not shot_class:
            continue
        videos = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        report["classes"][shot_class] = len(videos)
        report["total"] += len(videos)

    if len(report["classes"]) < 2:
        report["valid"] = False
        report["errors"].append("At least two labeled class folders are required for classification.")
    insufficient = {
        label: count for label, count in report["classes"].items()
        if count < MIN_SAMPLES_PER_CLASS
    }
    if insufficient:
        report["valid"] = False
        report["errors"].append(
            f"Insufficient samples per class (minimum {MIN_SAMPLES_PER_CLASS} before evaluation): "
            + ", ".join(f"{label}={count}" for label, count in sorted(insufficient.items()))
        )

    return report


def extract_features_from_manifest(
    manifest_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    """
    Extracts features from all valid videos listed in the manifest CSV.
    Returns:
        (X, y, filenames, splits, source_ids, class_names)
    """
    try:
        from app.ml.pose_extractor import POSE_MAX_FRAMES, extract_feature_vector, extract_pose_from_video
    except ImportError:
        logger.error("Cannot import pose extractor. Run with backend/ in PYTHONPATH.")
        sys.exit(1)

    mpath = Path(manifest_path).resolve()
    if not mpath.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        sys.exit(1)

    records: List[Dict[str, str]] = []
    with open(mpath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    logger.info("Loaded %d records from manifest: %s", len(records), mpath.name)

    valid_records = []
    for row in records:
        target_label = (row.get("target_label") or row.get("shot_label") or "").strip()
        note = row.get("validation_note", "") or ""
        if (
            str(row.get("is_valid", "")).strip().lower() in ("true", "1")
            and target_label
            and "duplicate_of:" not in note
        ):
            row["_training_target_label"] = target_label
            valid_records.append(row)

    if not valid_records:
        raise ValueError("Manifest contains no usable, labeled, non-duplicate videos.")
    target_classes = sorted({r["_training_target_label"] for r in valid_records})

    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

    X, y, filenames, splits, source_ids = [], [], [], [], []

    manifest_parent = mpath.parent
    for idx, r in enumerate(valid_records, 1):
        rel_path = r["video_path"]
        # Resolve path relative to manifest parent or workspace root
        video_full_path = manifest_parent / rel_path
        if not video_full_path.exists():
            video_full_path = project_root / rel_path
        if not video_full_path.exists():
            # Try absolute path
            video_full_path = Path(rel_path)

        if not video_full_path.exists():
            logger.warning("Video file not found at %s — skipping", video_full_path)
            continue

        hand = r.get("batting_hand", "right").lower()
        if hand not in ("left", "right"):
            hand = "right"

        try:
            logger.info(
                "[%d/%d] Extracting features: %s (%s)",
                idx,
                len(valid_records),
                r["filename"],
                r["shot_label"],
            )
            pose_result = extract_pose_from_video(str(video_full_path), max_frames=POSE_MAX_FRAMES)

            if not pose_result.get("landmarks_detected"):
                logger.warning("No landmarks detected in %s — skipping", r["filename"])
                continue

            feature_vec = extract_feature_vector(pose_result, batting_hand=hand)
            if feature_vec is None:
                logger.warning("Could not extract feature vector from %s — skipping", r["filename"])
                continue

            X.append(feature_vec)
            y.append(class_to_idx[r["_training_target_label"]])
            filenames.append(str(video_full_path))
            splits.append(normalize_split(r.get("split")))
            source_ids.append((r.get("source_id") or "").strip())

        except Exception as e:
            logger.error("Error extracting features from %s: %s", r["filename"], e)

    if not X:
        raise ValueError("No pose feature vectors could be extracted from usable manifest videos.")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        filenames,
        splits,
        source_ids,
        target_classes,
    )


def extract_features_from_dataset(
    dataset_dir: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Extract feature vectors from directory structure dataset/shots/<class>/video.mp4.
    """
    try:
        from app.ml.pose_extractor import POSE_MAX_FRAMES, extract_feature_vector, extract_pose_from_video
    except ImportError:
        logger.error("Cannot import pose extractor. Run with backend/ in PYTHONPATH.")
        sys.exit(1)

    from dataset_preparer import SPLIT_DIRECTORY_NAMES, extract_source_id, normalize_detailed_label

    root = Path(dataset_dir)
    if (root / "shots").is_dir():
        shots_root = root / "shots"
    elif root.name.lower() == "shots":
        shots_root = root
    else:
        shots_roots = sorted(path for path in root.rglob("shots") if path.is_dir())
        shots_root = shots_roots[0] if shots_roots else root
    class_dirs = []
    for child in sorted(shots_root.iterdir()) if shots_root.is_dir() else []:
        if child.is_dir() and child.name.lower() in SPLIT_DIRECTORY_NAMES:
            class_dirs.extend(path for path in sorted(child.iterdir()) if path.is_dir())
        elif child.is_dir():
            class_dirs.append(child)

    discovered = []
    for class_dir in class_dirs:
        label = normalize_detailed_label(class_dir.name)
        if label:
            discovered.append((label, class_dir))
    target_classes = sorted({label for label, _ in discovered})
    class_to_idx = {label: idx for idx, label in enumerate(target_classes)}
    X, y, filenames, source_ids = [], [], [], []

    for shot_class, class_dir in discovered:
        videos = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
        logger.info("Processing class '%s': %d videos", shot_class, len(videos))

        for video_file in videos:
            video_path = str(video_file)
            try:
                pose_result = extract_pose_from_video(video_path, max_frames=POSE_MAX_FRAMES)
                if not pose_result.get("landmarks_detected"):
                    continue

                feature_vec = extract_feature_vector(pose_result)
                if feature_vec is None:
                    continue

                X.append(feature_vec)
                y.append(class_to_idx[shot_class])
                filenames.append(video_path)
                source_ids.append(extract_source_id(video_file.name))
                logger.info("  ✓ %s", os.path.basename(video_path))

            except Exception as e:
                logger.error("Error processing %s: %s", video_path, e)

    if not X:
        raise ValueError("No pose feature vectors extracted. Check the class folders and videos.")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        filenames,
        source_ids,
        target_classes,
    )


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    target_classes: List[str],
    splits: Optional[List[str]] = None,
    source_ids: Optional[List[str]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Train a Random Forest classifier with leak-free train/test evaluation.
    Preserves official split if available; otherwise uses GroupShuffleSplit on source_id.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from app.ml.pose_extractor import (
        FEATURE_NAMES,
        FEATURE_SCHEMA_NAME,
        FEATURE_SCHEMA_VERSION,
        POSE_MAX_FRAMES,
    )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature schema mismatch: expected a matrix with {len(FEATURE_NAMES)} columns, "
            f"got shape {X.shape}."
        )
    if len(X) != len(y) or not len(X) or not np.isfinite(X).all():
        raise ValueError("Training features and labels must be non-empty, aligned, and finite.")
    if not target_classes or np.any(y < 0) or np.any(y >= len(target_classes)):
        raise ValueError("Training labels do not match the configured target classes.")
    if splits is not None and len(splits) != len(X):
        raise ValueError("Manifest split values do not align with the extracted examples.")
    if source_ids is not None and len(source_ids) != len(X):
        raise ValueError("Source IDs do not align with the extracted examples.")

    unique_classes, counts = np.unique(y, return_counts=True)
    class_dist = {target_classes[int(i)]: int(cnt) for i, cnt in zip(unique_classes, counts)}
    missing_targets = [name for i, name in enumerate(target_classes) if i not in unique_classes]
    if missing_targets:
        raise ValueError("No usable feature examples for target classes: " + ", ".join(missing_targets))
    insufficient = {name: count for name, count in class_dist.items() if count < MIN_SAMPLES_PER_CLASS}
    if insufficient:
        raise ValueError(
            f"Insufficient samples per target class; need at least {MIN_SAMPLES_PER_CLASS} "
            "usable videos before a train/evaluation split: "
            + ", ".join(f"{name}={count}" for name, count in sorted(insufficient.items()))
        )
    logger.info("Dataset: %d samples, %d features, %d classes", len(X), X.shape[1], len(unique_classes))
    logger.info("Class distribution: %s", class_dist)

    # Preserve complete official splits, and reject source recordings spread
    # across official train/evaluation partitions.
    has_official_split = splits is not None and "train" in splits and any(s in ("test", "val") for s in splits)
    if has_official_split:
        logger.info("Using designated train/validation/test splits from manifest.")
        official_group_splits = {}
        if source_ids is not None:
            for source_id, split in zip(source_ids, splits):
                if source_id and split in ("train", "val", "test"):
                    official_group_splits.setdefault(source_id, set()).add(split)
        leaked_groups = [key for key, value in official_group_splits.items() if len(value) > 1]
        if leaked_groups:
            raise ValueError(
                "Official manifest split separates clips from the same source recording: "
                + ", ".join(leaked_groups[:5])
            )
        train_mask = np.array([s == "train" for s in splits], dtype=bool)
        test_mask = np.array([s in ("test", "val") for s in splits], dtype=bool)
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Official manifest split must contain both train and validation/test examples.")
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        split_method = "official_manifest_split"
    elif source_ids is not None and all(source_ids) and len(set(source_ids)) > 1:
        logger.info("Searching grouped splits by source_id to prevent data leakage...")
        split_result = None
        for seed in range(42, 142):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            train_idx, test_idx = next(gss.split(X, y, groups=source_ids))
            train_counts = np.bincount(y[train_idx], minlength=len(target_classes))
            test_counts = np.bincount(y[test_idx], minlength=len(target_classes))
            if np.all(train_counts >= 2) and np.all(test_counts >= 1):
                split_result = (train_idx, test_idx)
                break
        if split_result is None:
            raise ValueError(
                "Could not create a source-grouped split with every target class represented "
                "in training and evaluation. Add independent source recordings or provide an official split."
            )
        train_idx, test_idx = split_result
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        split_method = "group_shuffle_split_by_source_id"
    else:
        raise ValueError(
            "Cannot make a leak-safe evaluation split: provide official train/val/test splits "
            "or reliable source_id values for every clip. Unknown provenance is not treated as independent."
        )

    train_counts = np.bincount(y_train, minlength=len(target_classes))
    test_counts = np.bincount(y_test, minlength=len(target_classes))
    bad_train = [target_classes[i] for i, count in enumerate(train_counts) if count < 2]
    bad_test = [target_classes[i] for i, count in enumerate(test_counts) if count < 1]
    if bad_train or bad_test:
        details = []
        if bad_train:
            details.append("fewer than 2 training examples: " + ", ".join(bad_train))
        if bad_test:
            details.append("no evaluation examples: " + ", ".join(bad_test))
        raise ValueError("Insufficient examples in requested split (" + "; ".join(details) + ").")

    logger.info(
        "Split result (%s): %d train samples, %d test samples",
        split_method,
        len(X_train),
        len(X_test),
    )

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest (CPU-friendly baseline)
    logger.info("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    class_indices = list(range(len(target_classes)))
    cm = confusion_matrix(y_test, y_pred, labels=class_indices).tolist()
    report = classification_report(
        y_test,
        y_pred,
        labels=class_indices,
        target_names=target_classes,
        output_dict=True,
        zero_division=0,
    )

    logger.info("Test set metrics:")
    logger.info("  Accuracy:   %.3f", accuracy)
    logger.info("  Precision:  %.3f (macro)", precision)
    logger.info("  Recall:     %.3f (macro)", recall)
    logger.info("  F1:         %.3f (macro)", f1)

    # Save trained model artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "shot_classifier.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    labels_path = os.path.join(output_dir, "label_classes.npy")
    meta_path = os.path.join(output_dir, "model_meta.json")

    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    active_labels = [target_classes[i] for i in sorted(np.unique(y))]
    np.save(labels_path, np.array(active_labels))

    evaluation_warning = None
    if any(count <= MIN_SAMPLES_PER_CLASS for count in counts):
        evaluation_warning = (
            "Evaluation metrics from five or fewer usable clips per class are highly uncertain "
            "and do not establish real-world performance."
        )
        logger.warning(evaluation_warning)

    examples_per_split = {}
    if splits is not None:
        for split in ("train", "val", "test", "unassigned"):
            examples_per_split[split] = int(sum(value == split for value in splits))
    else:
        examples_per_split = {"train": int(len(X_train)), "evaluation": int(len(X_test))}

    meta = {
        "name": "Cricket Shot Classifier (Random Forest)",
        "version": "2.0",
        "model_type": "sklearn_random_forest",
        "feature_schema": {
            "name": FEATURE_SCHEMA_NAME,
            "version": FEATURE_SCHEMA_VERSION,
            "features": FEATURE_NAMES,
            "dimension": len(FEATURE_NAMES),
        },
        "preprocessing": {"scaler": "StandardScaler", "feature_dtype": "float32"},
        "pose_extraction": {"max_frames": POSE_MAX_FRAMES},
        "label_mapping": label_mapping or {},
        "class_labels": active_labels,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "split_method": split_method,
        "supported_shots": active_labels,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "examples_per_split": examples_per_split,
        "examples_per_class": class_dist,
        "evaluation_warning": evaluation_warning,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model saved to: %s", output_dir)
    return meta


def check_manifest(manifest_path: str) -> dict:
    """Validate usable class counts without extracting pose features."""
    path = Path(manifest_path)
    if not path.is_file():
        return {
            "valid": False,
            "classes": {},
            "split_method": None,
            "examples_per_split": {},
            "errors": [f"Manifest not found: {path}"],
        }
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        rows = list(csv.DictReader(stream))
    usable_rows = []
    counts: Dict[str, int] = {}
    for row in rows:
        label = (row.get("target_label") or row.get("shot_label") or "").strip()
        if (
            label
            and str(row.get("is_valid", "")).strip().lower() in ("true", "1")
            and "duplicate_of:" not in (row.get("validation_note") or "")
        ):
            row["_target"] = label
            row["_split"] = normalize_split(row.get("split"))
            usable_rows.append(row)
            counts[label] = counts.get(label, 0) + 1
    errors = []
    if len(counts) < 2:
        errors.append("At least two target classes with usable videos are required.")
    insufficient = {label: count for label, count in counts.items() if count < MIN_SAMPLES_PER_CLASS}
    if insufficient:
        errors.append(
            f"Insufficient usable samples per class (minimum {MIN_SAMPLES_PER_CLASS}): "
            + ", ".join(f"{label}={count}" for label, count in sorted(insufficient.items()))
        )
    split_method = None
    split_counts = {}
    has_official_split = any(r["_split"] == "train" for r in usable_rows) and any(
        r["_split"] in ("val", "test") for r in usable_rows
    )
    if has_official_split:
        split_method = "official_manifest_split"
        source_splits: Dict[str, set] = {}
        for row in usable_rows:
            split = row["_split"]
            if split in ("train", "val", "test"):
                split_counts[split] = split_counts.get(split, 0) + 1
                source_id = (row.get("source_id") or "").strip()
                if source_id:
                    source_splits.setdefault(source_id, set()).add(split)
        leaked = [source for source, values in source_splits.items() if len(values) > 1]
        if leaked:
            errors.append("Official split separates source recordings: " + ", ".join(leaked[:5]))
        train_by_class = {label: 0 for label in counts}
        eval_by_class = {label: 0 for label in counts}
        for row in usable_rows:
            if row["_split"] == "train":
                train_by_class[row["_target"]] += 1
            elif row["_split"] in ("val", "test"):
                eval_by_class[row["_target"]] += 1
        bad_train = [label for label, count in train_by_class.items() if count < 2]
        bad_eval = [label for label, count in eval_by_class.items() if count < 1]
        if bad_train:
            errors.append("Official split has fewer than 2 training examples for: " + ", ".join(bad_train))
        if bad_eval:
            errors.append("Official split has no evaluation examples for: " + ", ".join(bad_eval))
    else:
        source_ids = [(row.get("source_id") or "").strip() for row in usable_rows]
        if len(source_ids) == len(usable_rows) and source_ids and all(source_ids) and len(set(source_ids)) > 1:
            from sklearn.model_selection import GroupShuffleSplit

            y_values = np.array([sorted(counts).index(row["_target"]) for row in usable_rows])
            found_split = False
            for seed in range(42, 142):
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                train_idx, eval_idx = next(splitter.split(np.zeros(len(y_values)), y_values, groups=source_ids))
                train_counts = np.bincount(y_values[train_idx], minlength=len(counts))
                eval_counts = np.bincount(y_values[eval_idx], minlength=len(counts))
                if np.all(train_counts >= 2) and np.all(eval_counts >= 1):
                    split_counts = {"train": len(train_idx), "evaluation": len(eval_idx)}
                    found_split = True
                    break
            if found_split:
                split_method = "group_shuffle_split_by_source_id"
            else:
                errors.append("Source-grouped splits cannot represent every class in train and evaluation.")
        else:
            errors.append(
                "Unknown provenance prevents a leak-safe split: provide official "
                "train/val/test values or reliable source_id values for every clip."
            )
    return {
        "valid": not errors,
        "classes": counts,
        "split_method": split_method,
        "examples_per_split": split_counts,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CricketCoach AI shot classifier")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset manifest CSV (e.g. dataset/cricket_dataset_manifest.csv)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Path to labeled folder dataset. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save model files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate dataset/manifest, do not train",
    )
    args = parser.parse_args()

    logger.info("=== CricketCoach AI — Training Pipeline ===")
    logger.info("Output dir: %s", args.output_dir)

    # 1. Manifest mode
    if args.manifest:
        logger.info("Mode: Manifest CSV (%s)", args.manifest)
        report = check_manifest(args.manifest)
        if args.check_only:
            for cls, count in report["classes"].items():
                logger.info("  %s: %d usable videos", cls, count)
            if report["split_method"]:
                logger.info(
                    "  Split method: %s (%s)",
                    report["split_method"],
                    report["examples_per_split"],
                )
            for error in report["errors"]:
                logger.error(error)
            sys.exit(0 if report["valid"] else 1)

        if not report["valid"]:
            for error in report["errors"]:
                logger.error("Training stopped: %s", error)
            sys.exit(2)

        try:
            X, y, filenames, splits, source_ids, target_classes = extract_features_from_manifest(
                args.manifest
            )
            mapping = {}
            summary_path = Path(args.manifest).resolve().parent / "manifest_summary.json"
            if summary_path.exists():
                with summary_path.open(encoding="utf-8") as summary_file:
                    summary = json.load(summary_file)
                    mapping = dict(summary.get("observed_label_mapping", {}))
                    mapping.update(summary.get("label_mapping", {}))
            metrics = train_and_evaluate(
                X,
                y,
                args.output_dir,
                target_classes=target_classes,
                splits=splits,
                source_ids=source_ids,
                label_mapping=mapping,
            )
        except (ValueError, FileNotFoundError) as error:
            logger.error("Training stopped: %s", error)
            sys.exit(2)

    # 2. Directory mode
    else:
        logger.info("Mode: Directory scanning (%s)", args.dataset_dir)
        report = check_dataset(args.dataset_dir)
        logger.info("Dataset report:")
        for cls, count in report["classes"].items():
            status = "✓" if count >= 5 else ("⚠" if count > 0 else "✗")
            logger.info("  %s %-20s: %d videos", status, cls, count)
        logger.info("  Total: %d videos", report["total"])

        if args.check_only:
            sys.exit(0 if report["valid"] else 1)

        if not report["valid"]:
            logger.error("\nDataset validation failed. Provide labeled videos before training.")
            logger.error("See docs/DATASET_GUIDE.md for dataset preparation instructions.")
            sys.exit(1)

        try:
            X, y, filenames, source_ids, target_classes = extract_features_from_dataset(args.dataset_dir)
            metrics = train_and_evaluate(
                X,
                y,
                args.output_dir,
                target_classes=target_classes,
                source_ids=source_ids,
            )
        except (ValueError, FileNotFoundError) as error:
            logger.error("Training stopped: %s", error)
            sys.exit(2)

    logger.info("\n=== Training complete ===")
    logger.info("Accuracy: %.1f%%", metrics["accuracy"] * 100)
    logger.info("F1 (macro): %.3f", metrics["f1_macro"])
    logger.info("\nRestart the backend to activate the new model.")


if __name__ == "__main__":
    main()
