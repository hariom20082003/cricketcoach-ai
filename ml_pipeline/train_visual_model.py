#!/usr/bin/env python3
"""Train and evaluate the 10-class visual-temporal CricketCoach model."""
import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "ml_pipeline"))

from app.ml.visual_classifier import (  # noqa: E402
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    DISPLAY_NAMES,
    EfficientNetTemporalModel,
    FRAME_COUNT,
    IMAGE_SIZE,
    LABELS,
    META_FILENAME,
    WEIGHTS_FILENAME,
    VisualShotClassifier,
    read_video_frames,
)
from train_model import check_manifest, normalize_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_visual_model")

DEFAULT_MANIFEST = PROJECT_ROOT / "dataset/prepared/cricket-shot/cricket_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backend/app/ml/models"
DATASET_ID = "rokmr/cricket-shot"
DATASET_URL = "https://huggingface.co/datasets/rokmr/cricket-shot"
PREPROCESSING = {
    "version": 1,
    "decoder": "OpenCV VideoCapture",
    "color_space": "RGB",
    "frame_sampling": "uniform_linspace_including_endpoints",
    "frames_per_video": FRAME_COUNT,
    "resize_and_crop": "EfficientNet_B0_Weights.IMAGENET1K_V1.transforms",
    "image_size": IMAGE_SIZE,
}


def _load_manifest(manifest_path: Path):
    rows_by_split: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8", errors="ignore", newline="") as stream:
        for row in csv.DictReader(stream):
            label = (row.get("target_label") or row.get("shot_label") or "").strip()
            note = row.get("validation_note") or ""
            if (
                not label
                or row.get("is_valid", "").strip().lower() not in {"true", "1"}
                or "duplicate_of:" in note
            ):
                continue
            split = normalize_split(row.get("split"))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Clip has no official split: {row.get('video_path')}")
            if label not in LABELS:
                raise ValueError(f"Unexpected target label {label!r}; expected the dataset's 10 labels")

            raw_path = Path(row.get("video_path", ""))
            candidates = [raw_path] if raw_path.is_absolute() else [manifest_path.parent / raw_path, PROJECT_ROOT / raw_path]
            video_path = next((path.resolve() for path in candidates if path.is_file()), None)
            if video_path is None:
                raise FileNotFoundError(f"Video file referenced by manifest is missing: {raw_path}")
            item = dict(row)
            item["target_label"] = label
            item["split"] = split
            item["resolved_video_path"] = str(video_path)
            rows_by_split[split].append(item)

    found = {row["target_label"] for rows in rows_by_split.values() for row in rows}
    if found != set(LABELS):
        raise ValueError(f"Manifest class labels do not match the configured 10 labels: {sorted(found)}")
    for split in ("train", "val", "test"):
        counts = Counter(row["target_label"] for row in rows_by_split[split])
        missing = [label for label in LABELS if counts[label] == 0]
        if missing:
            raise ValueError(f"{split} split is missing classes: {missing}")
    return rows_by_split


def _select_device(torch, name: str):
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but this PyTorch build cannot use Apple's Metal backend")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_embeddings(torch, model, rows, label_to_index, transform, device, video_batch_size):
    """Cache frozen EfficientNet features for one split; only compact vectors stay in memory."""
    sequences = []
    targets = []
    for start in range(0, len(rows), video_batch_size):
        batch = rows[start : start + video_batch_size]
        video_tensors = [
            read_video_frames(row["resolved_video_path"], transform=transform)
            for row in batch
        ]
        videos = torch.stack(video_tensors)
        batch_size, timesteps, channels, height, width = videos.shape
        with torch.inference_mode():
            embeddings = model.encode_frames(
                videos.reshape(batch_size * timesteps, channels, height, width).to(device)
            )
            embeddings = embeddings.reshape(batch_size, timesteps, -1).cpu()
        sequences.append(embeddings)
        targets.extend(label_to_index[row["target_label"]] for row in batch)
        if (start // video_batch_size + 1) % 20 == 0 or start + len(batch) == len(rows):
            logger.info("Encoded %d/%d %s clips", start + len(batch), len(rows), batch[0]["split"])
    return torch.cat(sequences), torch.tensor(targets, dtype=torch.long)


def _evaluate_logits(torch, model, features, targets, device, batch_size, labels):
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    model.eval()
    predictions = []
    truths = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = features[start : start + batch_size].to(device)
            logits = model.classify_embeddings(x)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(targets[start : start + batch_size].tolist())
    return {
        "accuracy": float(accuracy_score(truths, predictions)),
        "precision_macro": float(precision_score(
            truths, predictions, labels=list(range(len(labels))), average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            truths, predictions, labels=list(range(len(labels))), average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(truths, predictions, labels=list(range(len(labels))), average="macro", zero_division=0)),
        "per_class": classification_report(
            truths, predictions, labels=list(range(len(labels))), output_dict=True,
            target_names=labels, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truths, predictions, labels=list(range(len(labels)))
        ).astype(int).tolist(),
        "truths": truths,
        "predictions": predictions,
    }


def _write_json(path: Path, value):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def train(args):
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torchvision.models import EfficientNet_B0_Weights

    manifest_path = Path(args.manifest).resolve()
    readiness = check_manifest(str(manifest_path))
    if not readiness["valid"]:
        raise ValueError("Training readiness check failed: " + "; ".join(readiness["errors"]))
    if readiness["split_method"] != "official_manifest_split":
        raise ValueError("This training command requires the dataset's official train/val/test split")
    rows_by_split = _load_manifest(manifest_path)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = _select_device(torch, args.device)
    logger.info("Device: %s; PyTorch %s; TorchVision %s", device, torch.__version__, __import__("torchvision").__version__)
    logger.info("Usable examples by split: %s", {key: len(value) for key, value in rows_by_split.items()})
    logger.info("Class mapping: %s", label_to_index)

    start_time = time.perf_counter()
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = EfficientNetTemporalModel(num_classes=len(LABELS), hidden_size=args.hidden_size, weights=weights)
    model.to(device)
    model.eval()
    transform = weights.transforms()
    preprocessing = {
        **PREPROCESSING,
        "resize_short_side": int(transform.resize_size[0]),
        "center_crop": [int(value) for value in transform.crop_size],
        "interpolation": str(transform.interpolation),
        "mean": [float(value) for value in transform.mean],
        "std": [float(value) for value in transform.std],
    }
    feature_sets = {}
    target_sets = {}
    for split in ("train", "val"):
        feature_sets[split], target_sets[split] = _extract_embeddings(
            torch, model, rows_by_split[split], label_to_index, transform, device, args.video_batch_size
        )

    train_features = feature_sets["train"]
    train_targets = target_sets["train"]
    val_features = feature_sets["val"]
    val_targets = target_sets["val"]
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    optimizer = torch.optim.AdamW(
        list(model.temporal.parameters()) + list(model.classifier.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify_embeddings(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_seen += len(y)

        val_metrics = _evaluate_logits(
            torch, model, val_features, val_targets, device, args.batch_size, LABELS
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_seen),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["f1_macro"],
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_accuracy=%.4f val_macro_f1=%.4f",
            epoch, args.epochs, epoch_record["train_loss"],
            epoch_record["val_accuracy"], epoch_record["val_macro_f1"],
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logger.info("Early stopping after %d epochs without validation macro-F1 improvement", args.patience)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    validation_metrics = _evaluate_logits(
        torch, model, val_features, val_targets, device, args.batch_size, LABELS
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_FILENAME
    temp_weights = weights_path.with_suffix(weights_path.suffix + ".tmp")
    torch.save(best_state, temp_weights)
    temp_weights.replace(weights_path)
    del model
    del feature_sets, target_sets, train_features, train_targets, val_features, val_targets
    if device.type == "mps":
        torch.mps.empty_cache()

    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    class_counts_by_split = {
        split: dict(sorted(Counter(row["target_label"] for row in rows).items()))
        for split, rows in rows_by_split.items()
    }
    metadata = {
        "name": "CricketShot EfficientNet-B0 + BiGRU",
        "version": "1.0.0",
        "model_type": ARCHITECTURE,
        "architecture": ARCHITECTURE,
        "architecture_version": ARCHITECTURE_VERSION,
        "weights_file": WEIGHTS_FILENAME,
        "class_labels": LABELS,
        "supported_shots": LABELS,
        "class_index": label_to_index,
        "display_names": DISPLAY_NAMES,
        "frames_per_video": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
        "hidden_size": args.hidden_size,
        "preprocessing": preprocessing,
        "pretrained_encoder": "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "training_device": str(device),
        "dataset": {
            "id": DATASET_ID,
            "url": DATASET_URL,
            "declared_license": "Apache-2.0",
            "footage_rights_note": (
                "The dataset card declares Apache-2.0. Its acknowledgment references CricShot10; "
                "underlying match footage remains subject to its original rights holders. "
                "This local research artifact does not establish commercial redistribution rights."
            ),
        },
        "split_method": readiness["split_method"],
        "source_match_provenance_available": False,
        "duplicate_clips_excluded": int(json.loads(
            (manifest_path.parent / "manifest_summary.json").read_text(encoding="utf-8")
        ).get("duplicate_clips", 0)) if (manifest_path.parent / "manifest_summary.json").is_file() else None,
        "examples_per_split": counts,
        "class_counts_per_split": class_counts_by_split,
        "training_samples": counts["train"],
        "validation_samples": counts["val"],
        "test_samples": counts["test"],
        "training_config": {
            "seed": args.seed,
            "max_epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.patience,
            "frozen_encoder": True,
            "temporal_head": "one-layer bidirectional GRU",
        },
        "training_time_seconds": time.perf_counter() - start_time,
        "validation_metrics": {
            key: value for key, value in validation_metrics.items()
            if key not in {"truths", "predictions"}
        },
        "validation_history": history,
        "test_metrics": None,
        "test_inference_latency_ms": None,
        "confidence_note": "Softmax class scores are uncalibrated and should not be interpreted as calibrated probabilities.",
    }
    meta_path = output_dir / META_FILENAME
    _write_json(meta_path, metadata)

    # Evaluate every held-out test clip through the exact backend video inference path.
    classifier = VisualShotClassifier(model_dir=str(output_dir))
    if not classifier.is_available:
        raise RuntimeError(f"Saved visual artifact failed to load: {classifier.load_error}")
    y_true = []
    y_pred = []
    latencies_ms = []
    for index, row in enumerate(rows_by_split["test"], 1):
        started = time.perf_counter()
        prediction = classifier.predict_video(row["resolved_video_path"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if prediction.get("shot_label") not in label_to_index:
            raise RuntimeError(
                f"Production inference failed on held-out clip {row['resolved_video_path']}: "
                f"{prediction.get('error', 'no class predicted')}"
            )
        y_true.append(label_to_index[row["target_label"]])
        y_pred.append(label_to_index[prediction["shot_label"]])
        latencies_ms.append(elapsed_ms)
        if index % 25 == 0 or index == len(rows_by_split["test"]):
            logger.info("Production test inference: %d/%d clips", index, len(rows_by_split["test"]))

    test_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(__import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(
            y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0
        )),
        "recall_macro": float(__import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(
            y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(LABELS))), output_dict=True,
            target_names=LABELS, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))
        ).astype(int).tolist(),
    }
    metadata["test_metrics"] = test_metrics
    metadata["test_inference_latency_ms"] = {
        "mean": float(np.mean(latencies_ms)),
        "median": float(np.median(latencies_ms)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "n_videos": len(latencies_ms),
    }
    _write_json(meta_path, metadata)

    # Re-load the final two-file artifact and make one real inference after reload.
    reloaded = VisualShotClassifier(model_dir=str(output_dir))
    if not reloaded.is_available:
        raise RuntimeError(f"Final backend reload failed: {reloaded.load_error}")
    sample_row = rows_by_split["test"][0]
    sample_prediction = reloaded.predict_video(sample_row["resolved_video_path"])
    if sample_prediction.get("shot_label") not in label_to_index:
        raise RuntimeError("Final backend reload did not predict a class for a held-out video")
    metadata["post_save_inference_verified"] = True
    _write_json(meta_path, metadata)

    logger.info("Saved weights: %s", weights_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info("Held-out production test metrics: %s", {
        key: test_metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    })
    logger.info("Held-out production inference latency (ms/video): %s", metadata["test_inference_latency_ms"])
    logger.info("Class metrics: %s", json.dumps(test_metrics["per_class"], indent=2))
    logger.info("Confusion matrix label order: %s", LABELS)
    logger.info("Confusion matrix: %s", test_metrics["confusion_matrix"])
    logger.info("Post-save backend inference smoke test passed")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, default=FRAME_COUNT, help="Must match the versioned backend artifact contract")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--video-batch-size", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.frames != FRAME_COUNT:
        parser.error(f"--frames must be {FRAME_COUNT} for the backend artifact contract")
    if min(args.epochs, args.patience, args.batch_size, args.video_batch_size, args.hidden_size) < 1:
        parser.error("Epochs, patience, batch sizes, and hidden size must all be positive")
    metadata = train(args)
    print("\n=== Visual-temporal training complete ===")
    print(json.dumps({
        "weights": str(Path(args.output_dir).resolve() / WEIGHTS_FILENAME),
        "metadata": str(Path(args.output_dir).resolve() / META_FILENAME),
        "device": metadata["training_device"],
        "best_epoch": metadata["training_config"]["best_epoch"],
        "training_time_seconds": round(metadata["training_time_seconds"], 2),
        "test_metrics": metadata["test_metrics"],
        "test_inference_latency_ms": metadata["test_inference_latency_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
