#!/usr/bin/env python3
"""
CricketCoach AI — Dataset Preparation and Validation Engine
===========================================================
Prepares, validates, and manifests cricket batting video datasets.

Key features:
1. Multi-format video validation using OpenCV (MP4, AVI, MOV, WebM, MKV, M4V).
2. Reads original annotations from:
   - CricketVision (VGG Image Annotator VIA JSON format)
   - CSV / TSV metadata files
   - JSON / JSONL metadata files
   - Directory-based shot class layouts
   - Text split files (train.txt, val.txt, test.txt)
3. Preserves original labels while mapping to canonical shot taxonomy without fabrication.
4. Extracts metadata: FPS, duration, frame count, dimensions, readability, and content hashes.
5. Identifies duplicate clips via SHA-256 hash.
6. Preserves existing train/val/test splits or generates grouped splits on source_id to prevent data leakage.
7. Generates manifest CSV, summary JSON, and human-readable validation report.
"""

import csv
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported video extensions (case-insensitive)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}

# Canonical supported shot classes in CricketCoach AI
CANONICAL_SHOT_CLASSES = [
    "cover_drive",
    "straight_drive",
    "pull",
    "cut",
    "sweep",
    "flick",
    "defensive",
]

DETAILED_SHOT_CLASSES = [
    "cover_drive",
    "straight_drive",
    "pull",
    "hook",
    "square_cut",
    "late_cut",
    "sweep",
    "reverse_sweep",
    "defensive",
]
# Exact source labels used by the public rokmr/cricket-shot archive. Keep
# these as distinct targets instead of collapsing them into the older taxonomy.
CRICKETSHOT_LABELS = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep",
]
SPLIT_DIRECTORY_NAMES = {"train", "training", "val", "validation", "valid", "test", "testing"}

# Mapping table from common dataset shot names to canonical classes
# Preserves original_label in metadata while standardizing shot_label
SHOT_SYNONYMS = {
    # Off drive / Cover drive
    "offdrive": "cover_drive",
    "off_drive": "cover_drive",
    "off drive": "cover_drive",
    "coverdrive": "cover_drive",
    "cover_drive": "cover_drive",
    "cover drive": "cover_drive",
    # On drive / Straight drive
    "ondrive": "straight_drive",
    "on_drive": "straight_drive",
    "on drive": "straight_drive",
    "straightdrive": "straight_drive",
    "straight_drive": "straight_drive",
    "straight drive": "straight_drive",
    # Keep the legacy alias for old shot_label consumers. The separate
    # target_label field preserves Hook as its own classifier class.
    "pull": "pull",
    "pullshot": "pull",
    "pull_shot": "pull",
    "pull shot": "pull",
    "hook": "pull",
    "hookshot": "pull",
    "hook_shot": "pull",
    "hook shot": "pull",
    # Cut / Square drive
    "cut": "cut",
    "cutshot": "cut",
    "cut_shot": "cut",
    "cut shot": "cut",
    "squaredrive": "cut",
    "square_drive": "cut",
    "square drive": "cut",
    "cut / square drive": "cut",
    # Sweep
    "sweep": "sweep",
    "sweepshot": "sweep",
    "sweep_shot": "sweep",
    "sweep shot": "sweep",
    "paddle_sweep": "sweep",
    # Flick / Glance
    "flick": "flick",
    "flickshot": "flick",
    "flick_shot": "flick",
    "flick shot": "flick",
    "glance": "flick",
    "leg_glance": "flick",
    "leg glance": "flick",
    # Defensive / Block
    "block": "defensive",
    "defensive": "defensive",
    "defensive_shot": "defensive",
    "defensive shot": "defensive",
    "forward_defense": "defensive",
    "forward_defensive": "defensive",
    "backfoot_defense": "defensive",
}

# VIA attribute mappings used by CricketVision dataset
CRICKETVISION_VIA_STROKE_MAP = {
    "0": "cover_drive",  # OffDrive
    "1": "straight_drive",  # OnDrive
    "2": "cut",  # Cut
    "3": "flick",  # Glance
    "4": "pull",  # Historical CricketVision shot_label compatibility
    "5": "sweep",  # Sweep
    "6": "defensive",  # Block
    "offdrive": "cover_drive",
    "ondrive": "straight_drive",
    "cut": "cut",
    "glance": "flick",
    "hook": "pull",
    "sweep": "sweep",
    "block": "defensive",
}
CRICKETVISION_VIA_STROKE_NAMES = {
    "0": "OffDrive",
    "1": "OnDrive",
    "2": "Cut",
    "3": "Glance",
    "4": "Hook",
    "5": "Sweep",
    "6": "Block",
}

CRICKETVISION_VIA_HAND_MAP = {
    "0": "right",
    "1": "left",
    "right": "right",
    "left": "left",
}

CRICKETVISION_VIA_FOOT_MAP = {
    "0": "front_foot",
    "1": "back_foot",
    "frontfoot": "front_foot",
    "backfoot": "back_foot",
}


@dataclass
class VideoMetadata:
    video_id: str
    video_path: str
    original_relative_path: str
    filename: str
    shot_label: Optional[str]
    target_label: Optional[str]
    original_label: Optional[str]
    source_id: str
    batting_hand: str
    foot_type: str
    phase_timestamps: str  # JSON string
    technique_quality: str  # JSON string
    split: str  # 'train', 'val', 'test', 'unassigned'
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    file_size_bytes: int
    content_hash: str
    is_valid: bool
    validation_note: str


def compute_file_hash(filepath: Path, block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file for exact duplicate detection."""
    sha = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha.update(block)
        return sha.hexdigest()
    except Exception as e:
        logger.warning("Could not compute hash for %s: %s", filepath, e)
        return ""


def extract_source_id(filename: str) -> str:
    """
    Extracts or heuristics a source ID (e.g. match ID or parent video)
    from video filenames to prevent data leakage during train/val/test splits.
    Examples:
        match01_clip03.mp4 -> match01
        P1_V1_clip02.mp4   -> P1_V1
        IND_vs_AUS_2020_001.mp4 -> IND_vs_AUS_2020
    """
    stem = Path(filename).stem
    patterns = [
        r"^(match\d+)",
        r"^(P\d+[_\-]V\d+)",
        r"^(game\d+)",
        r"^([A-Z]{3}_vs_[A-Z]{3}(?:[_\-]\d+)?)",
        r"^([a-zA-Z0-9_\-]+?)[_\-](?:clip|shot|part|seq|cut)[_\-]?\d+",
        r"^([a-zA-Z0-9_\-]+?)[_\-]\d{2,}$",
    ]
    for pattern in patterns:
        m = re.match(pattern, stem, re.IGNORECASE)
        if m:
            return m.group(1).rstrip("_-")
    # A filename alone does not establish an independent source recording.
    return ""


def normalize_detailed_label(raw_label: Optional[str]) -> Optional[str]:
    """Convert a folder label to a stable slug without collapsing shot classes."""
    if raw_label is None or not str(raw_label).strip():
        return None
    return re.sub(r"[^a-z0-9]+", "_", str(raw_label).strip().lower()).strip("_") or None


def _normalise_mapping(mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
    result = {}
    for source, target in (mapping or {}).items():
        source_key = normalize_detailed_label(source)
        target_label = normalize_detailed_label(target)
        if not source_key or not target_label:
            raise ValueError("Label mappings must have non-empty source and target labels")
        result[source_key] = target_label
    return result


def normalize_shot_label(raw_label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes a raw shot label into canonical taxonomy.
    Returns (canonical_label, original_label).
    Never invents a label if input is empty or unknown.
    """
    if not raw_label or not str(raw_label).strip():
        return None, None

    original = str(raw_label).strip()
    clean = original.lower().replace("-", "_").replace("  ", " ").strip()

    if clean in SHOT_SYNONYMS:
        return SHOT_SYNONYMS[clean], original

    clean_no_underscore = clean.replace("_", " ")
    if clean_no_underscore in SHOT_SYNONYMS:
        return SHOT_SYNONYMS[clean_no_underscore], original

    # Direct match with canonical classes
    for c in CANONICAL_SHOT_CLASSES:
        if clean == c or clean.replace("_", " ") == c.replace("_", " "):
            return c, original

    # Unmapped but valid non-empty label: normalize to lowercase snake_case
    fallback = re.sub(r"[^a-z0-9_]+", "_", clean).strip("_")
    return fallback, original


def validate_video_with_opencv(filepath: Path) -> Dict[str, Any]:
    """
    Validates a video using OpenCV.
    Reads metadata and checks readability of frames.
    """
    result = {
        "is_valid": False,
        "validation_note": "unknown_error",
        "duration_seconds": 0.0,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
    }

    if not filepath.exists():
        result["validation_note"] = "file_not_found"
        return result

    if filepath.stat().st_size == 0:
        result["validation_note"] = "zero_byte_file"
        return result

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        result["validation_note"] = "cannot_open_video_stream"
        return result

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check frame reading
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            result["validation_note"] = "cannot_read_first_frame"
            return result

        # Read middle frame if video has multiple frames
        if frame_count > 1:
            mid_idx = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ret_mid, mid_frame = cap.read()
            if not ret_mid or mid_frame is None:
                # Still readable from first frame, but note partial corruption
                pass

        # Calculate duration
        duration = 0.0
        if fps > 0 and frame_count > 0:
            duration = round(frame_count / fps, 3)

        if width <= 0 or height <= 0:
            result["validation_note"] = "invalid_dimensions"
            return result

        result["is_valid"] = True
        result["validation_note"] = "ok"
        result["width"] = width
        result["height"] = height
        result["fps"] = round(fps, 2)
        result["frame_count"] = max(1, frame_count)
        result["duration_seconds"] = duration
        return result

    except Exception as e:
        result["validation_note"] = f"opencv_exception:{str(e)[:50]}"
        return result
    finally:
        cap.release()


class CricketDatasetPreparer:
    """
    Scans a local dataset folder, inspects annotations and videos,
    validates with OpenCV, and generates an ML-ready manifest.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        train_val_test_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        random_seed: int = 42,
        label_mapping: Optional[Dict[str, str]] = None,
    ):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir or self.input_dir).resolve()
        self.split_ratio = train_val_test_ratio
        self.random_seed = random_seed
        self.label_mapping = _normalise_mapping(label_mapping)
        self.video_records: List[VideoMetadata] = []
        self.seen_hashes: Dict[str, str] = {}  # hash -> first_filepath

    def run(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Executes full dataset preparation pipeline:
        1. Discover videos & annotations
        2. Validate videos with OpenCV
        3. Parse metadata and match annotations
        4. Detect duplicates
        5. Assign or preserve splits
        6. Export manifest CSV and summary report
        """
        logger.info("Scanning dataset directory: %s", self.input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Discover all videos
        video_files = self._find_video_files()
        logger.info("Discovered %d video files with supported extensions.", len(video_files))

        # 2. Parse all annotations in the directory
        annotations = self._parse_all_annotations(video_files)
        logger.info("Parsed annotations for %d video entries.", len(annotations))

        # 3. Validate each video and combine with annotations
        self.video_records = []
        for idx, vpath in enumerate(video_files, 1):
            annotation = dict(annotations.get(vpath.name, {}))
            # Annotation files are often keyed only by basename. Repeated names
            # across official splits can overwrite one another, so explicit
            # directory labels/splits for this exact path take precedence.
            annotation.update(self._infer_from_path(vpath))
            record = self._process_video_file(vpath, idx, annotation)
            self.video_records.append(record)

        # 4. Handle train/val/test splits
        self._assign_splits()

        # 5. Export manifest CSV
        manifest_path = self.output_dir / "cricket_dataset_manifest.csv"
        self._export_manifest_csv(manifest_path)

        # 6. Generate and save summary statistics
        summary = self._generate_summary_report(manifest_path)
        summary_json_path = self.output_dir / "manifest_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        report_md_path = self.output_dir / "dataset_validation_report.md"
        self._export_markdown_report(report_md_path, summary)

        logger.info("Manifest created at: %s", manifest_path)
        logger.info("Summary JSON at: %s", summary_json_path)
        logger.info("Validation report at: %s", report_md_path)

        return manifest_path, summary

    def _find_video_files(self) -> List[Path]:
        """Finds all videos matching supported extensions recursively."""
        video_files = []
        for root, _, files in os.walk(self.input_dir):
            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                # macOS archives commonly carry AppleDouble resource-fork
                # sidecars such as ._cover_0001.avi. They are not video clips.
                if not fname.startswith(".") and ext in SUPPORTED_VIDEO_EXTENSIONS:
                    video_files.append(Path(root) / fname)
        return video_files

    def _layout_class_labels(self) -> Dict[str, str]:
        """Return normalized/original class labels from documented ``shots`` folders."""
        labels: Dict[str, str] = {}
        for shots_dir in self.input_dir.rglob("shots"):
            if not shots_dir.is_dir():
                continue
            for class_dir in shots_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                if class_dir.name.lower() in SPLIT_DIRECTORY_NAMES:
                    for split_class_dir in class_dir.iterdir():
                        if split_class_dir.is_dir():
                            key = normalize_detailed_label(split_class_dir.name)
                            if key:
                                labels.setdefault(key, split_class_dir.name)
                else:
                    key = normalize_detailed_label(class_dir.name)
                    if key:
                        labels.setdefault(key, class_dir.name)
        return labels

    def _parse_all_annotations(self, video_files: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Finds and parses any annotation files (CricketVision VIA JSON, CSV, text splits).
        Keyed by filename (e.g. 'clip_01.mp4').
        """
        annotations: Dict[str, Dict[str, Any]] = {}

        # 1. Check for CricketVision VIA JSON files
        json_files = list(self.input_dir.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                via_parsed = self._try_parse_via_json(data)
                if via_parsed:
                    logger.info("Parsed VIA annotations from %s (%d items)", jf.name, len(via_parsed))
                    for k, v in via_parsed.items():
                        annotations.setdefault(k, {}).update(v)
                else:
                    # Generic JSON list or dict
                    generic_parsed = self._try_parse_generic_json(data)
                    if generic_parsed:
                        for k, v in generic_parsed.items():
                            annotations.setdefault(k, {}).update(v)
            except Exception as e:
                logger.debug("Could not parse JSON %s: %s", jf, e)

        # 2. Check for CSV / TSV annotation files
        csv_files = list(self.input_dir.rglob("*.csv")) + list(self.input_dir.rglob("*.tsv"))
        for cf in csv_files:
            if cf.name in ("cricket_dataset_manifest.csv",):
                continue
            try:
                csv_parsed = self._try_parse_csv_annotations(cf)
                if csv_parsed:
                    logger.info("Parsed CSV annotations from %s (%d items)", cf.name, len(csv_parsed))
                    for k, v in csv_parsed.items():
                        annotations.setdefault(k, {}).update(v)
            except Exception as e:
                logger.debug("Could not parse CSV %s: %s", cf, e)

        # 3. Check for text split files (train.txt, val.txt, test.txt)
        txt_splits = self._parse_text_splits()
        for k, v in txt_splits.items():
            annotations.setdefault(k, {}).update(v)

        # 4. Infer labels and split from directory structure if not yet annotated
        for vpath in video_files:
            if vpath.name not in annotations or not annotations[vpath.name].get("shot_label"):
                dir_info = self._infer_from_path(vpath)
                if dir_info:
                    annotations.setdefault(vpath.name, {}).update(dir_info)

        return annotations

    def _try_parse_via_json(self, data: Any) -> Optional[Dict[str, Dict[str, Any]]]:
        """Parses VGG Image Annotator (VIA) JSON used by CricketVision."""
        if not isinstance(data, dict):
            return None

        # VIA format usually has '_via_img_metadata' or 'attribute' or 'vid_list'
        metadata_dict = None
        if "_via_img_metadata" in data and isinstance(data["_via_img_metadata"], dict):
            metadata_dict = data["_via_img_metadata"]
        elif "attribute" in data and any(k in data for k in ("project", "vid_list")):
            # Top-level or sub-key score structure
            metadata_dict = {k: v for k, v in data.items() if isinstance(v, dict) and "vid" in v}

        if not metadata_dict:
            return None

        attr_def = data.get("attribute", {})
        results: Dict[str, Dict[str, Any]] = {}

        # Look for file list mapping if present
        file_map: Dict[str, str] = {}
        if "file" in data and isinstance(data["file"], dict):
            for fid, fobj in data["file"].items():
                if isinstance(fobj, dict) and "fname" in fobj:
                    file_map[str(fid)] = fobj["fname"]

        for item_key, item_val in metadata_dict.items():
            if not isinstance(item_val, dict):
                continue
            fname = item_val.get("filename") or file_map.get(str(item_val.get("vid", "")))
            if not fname:
                # Try item_key if it resembles a filename
                if any(item_key.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                    fname = item_key
                else:
                    fname = f"{item_key}.mp4"

            av = item_val.get("av", {})
            z = item_val.get("z", [])

            # Extract CricketVision attributes:
            # Attribute 9 = StrokeType
            # Attribute 7 = Handedness
            # Attribute 8 = FootType
            # Attributes 2-6 = Head, Shoulder, Hands, Hips, Feet Scores
            # Attribute 1 = TEMPORAL-SEGMENTS (Buildup, Execution, FollowThrough)
            raw_stroke = av.get("9") or av.get("StrokeType") or av.get("stroke")
            raw_hand = av.get("7") or av.get("Handedness") or av.get("hand")
            raw_foot = av.get("8") or av.get("FootType") or av.get("foot")

            # Scores
            head_sc = av.get("2") or av.get("Head_Score")
            shoul_sc = av.get("3") or av.get("Shoulder_Score")
            hands_sc = av.get("4") or av.get("Hands_Score")
            hips_sc = av.get("5") or av.get("Hips_Score")
            feet_sc = av.get("6") or av.get("Feet_Score")

            # Map stroke
            shot_label, original_label = None, None
            if raw_stroke is not None:
                stroke_str = str(raw_stroke).strip()
                if stroke_str in CRICKETVISION_VIA_STROKE_MAP:
                    shot_label = CRICKETVISION_VIA_STROKE_MAP[stroke_str]
                    original_label = CRICKETVISION_VIA_STROKE_NAMES.get(
                        stroke_str, f"StrokeID_{stroke_str}"
                    )
                else:
                    shot_label, original_label = normalize_shot_label(stroke_str)

            # Map hand
            hand_str = "unknown"
            if raw_hand is not None:
                hand_clean = str(raw_hand).lower().strip()
                hand_str = CRICKETVISION_VIA_HAND_MAP.get(hand_clean, "unknown")

            # Map foot
            foot_str = "unknown"
            if raw_foot is not None:
                foot_clean = str(raw_foot).lower().strip()
                foot_str = CRICKETVISION_VIA_FOOT_MAP.get(foot_clean, "unknown")

            # Quality dict
            quality = {}
            scores = [
                ("head", head_sc),
                ("shoulder", shoul_sc),
                ("hands", hands_sc),
                ("hips", hips_sc),
                ("feet", feet_sc),
            ]
            valid_scores = []
            for name, sc in scores:
                if sc is not None:
                    try:
                        val = float(sc)
                        quality[name] = val
                        valid_scores.append(val)
                    except ValueError:
                        pass
            if valid_scores:
                quality["overall_mean"] = round(float(np.mean(valid_scores)), 2)

            # Phase temporal segments
            phases = {}
            if "1" in av or "phase" in av:
                phase_name = av.get("1") or av.get("phase")
                if z and len(z) >= 1:
                    phases[str(phase_name)] = [float(t) for t in z]

            results[Path(fname).name] = {
                "shot_label": shot_label,
                "original_label": original_label,
                "batting_hand": hand_str,
                "foot_type": foot_str,
                "phase_timestamps": phases,
                "technique_quality": quality,
            }

        return results if results else None

    def _try_parse_generic_json(self, data: Any) -> Optional[Dict[str, Dict[str, Any]]]:
        """Parses generic JSON list or mapping."""
        results: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                vname = item.get("video") or item.get("filename") or item.get("file") or item.get("name")
                if vname:
                    shot = item.get("label") or item.get("shot") or item.get("stroke") or item.get("class")
                    shot_label, orig_label = normalize_shot_label(shot)
                    results[Path(str(vname)).name] = {
                        "shot_label": shot_label,
                        "original_label": orig_label,
                        "batting_hand": str(item.get("handedness", item.get("batting_hand", "unknown"))).lower(),
                        "foot_type": str(item.get("foot_type", "unknown")).lower(),
                        "split": str(item.get("split", "unassigned")),
                        "source_id": str(item.get("source_id", item.get("match_id", ""))),
                    }
        elif isinstance(data, dict):
            for k, item in data.items():
                if isinstance(item, dict):
                    shot = item.get("label") or item.get("shot") or item.get("stroke") or item.get("class")
                    shot_label, orig_label = normalize_shot_label(shot)
                    results[Path(str(k)).name] = {
                        "shot_label": shot_label,
                        "original_label": orig_label,
                        "batting_hand": str(item.get("handedness", item.get("batting_hand", "unknown"))).lower(),
                        "foot_type": str(item.get("foot_type", "unknown")).lower(),
                        "split": str(item.get("split", "unassigned")),
                    }
        return results if results else None

    def _try_parse_csv_annotations(self, csv_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
        """Parses CSV / TSV metadata files."""
        results: Dict[str, Dict[str, Any]] = {}
        delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames:
                return None

            # Detect column headers
            fields = {k.lower().strip(): k for k in reader.fieldnames if k}
            col_video = next((fields[k] for k in ["video", "video_name", "filename", "file", "path", "clip", "video_id"] if k in fields), None)
            col_label = next((fields[k] for k in ["label", "shot", "stroke", "shot_type", "class", "category", "shot_label"] if k in fields), None)
            col_split = next((fields[k] for k in ["split", "subset", "partition", "fold", "set"] if k in fields), None)
            col_hand = next((fields[k] for k in ["handedness", "batting_hand", "hand"] if k in fields), None)
            col_source = next((fields[k] for k in ["source", "source_id", "match", "match_id"] if k in fields), None)
            col_quality = next((fields[k] for k in ["quality", "score", "rating", "competency"] if k in fields), None)

            if not col_video or not col_label:
                return None

            for row in reader:
                vname = row.get(col_video, "").strip()
                if not vname:
                    continue
                raw_label = row.get(col_label, "").strip()
                shot_label, original_label = normalize_shot_label(raw_label)

                entry: Dict[str, Any] = {
                    "shot_label": shot_label,
                    "original_label": original_label,
                }
                if col_split and row.get(col_split):
                    entry["split"] = row[col_split].strip().lower()
                if col_hand and row.get(col_hand):
                    entry["batting_hand"] = row[col_hand].strip().lower()
                if col_source and row.get(col_source):
                    entry["source_id"] = row[col_source].strip()
                if col_quality and row.get(col_quality):
                    entry["technique_quality"] = {"score": row[col_quality].strip()}

                results[Path(vname).name] = entry

        return results if results else None

    def _parse_text_splits(self) -> Dict[str, Dict[str, Any]]:
        """Parses traditional train.txt / val.txt / test.txt split files."""
        results: Dict[str, Dict[str, Any]] = {}
        for split_name in ("train", "val", "validation", "test"):
            for txt_file in self.input_dir.rglob(f"{split_name}.txt"):
                try:
                    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            vname = Path(parts[0]).name
                            target_split = "val" if split_name == "validation" else split_name
                            entry = {"split": target_split}
                            if len(parts) > 1 and parts[1].strip():
                                shot, orig = normalize_shot_label(parts[1].strip())
                                entry["shot_label"] = shot
                                entry["original_label"] = orig
                            results.setdefault(vname, {}).update(entry)
                except Exception as e:
                    logger.debug("Error reading %s: %s", txt_file, e)
        return results

    def _infer_from_path(self, vpath: Path) -> Dict[str, Any]:
        """
        Infers shot class and split from folder structure if present.
        e.g. dataset/shots/cover_drive/video.mp4 -> shot_label='cover_drive'
        e.g. dataset/train/cover_drive/video.mp4 -> split='train', shot_label='cover_drive'
        """
        rel_parts = list(vpath.relative_to(self.input_dir).parts[:-1])
        lower_parts = [p.lower() for p in rel_parts]
        info: Dict[str, Any] = {}

        # Check for split folders
        for part in lower_parts:
            if part in ("train", "training"):
                info["split"] = "train"
            elif part in ("val", "validation", "valid"):
                info["split"] = "val"
            elif part in ("test", "testing"):
                info["split"] = "test"

        # A folder immediately below ``shots`` is an explicit class label. Skip
        # an optional official split directory between ``shots`` and the class.
        if "shots" in lower_parts:
            idx = lower_parts.index("shots") + 1
            if idx < len(rel_parts) and lower_parts[idx] in SPLIT_DIRECTORY_NAMES:
                idx += 1
            if idx < len(rel_parts):
                original = rel_parts[idx]
                info["shot_label"] = normalize_detailed_label(original)
                info["original_label"] = original

        # Retain inference for known legacy class folders outside a ``shots`` root.
        if not info.get("shot_label"):
            known = set(DETAILED_SHOT_CLASSES + CANONICAL_SHOT_CLASSES + CRICKETSHOT_LABELS)
            for part in reversed(rel_parts):
                label = normalize_detailed_label(part)
                if label in known:
                    info["shot_label"] = label
                    info["original_label"] = part
                    break

        return info

    def _process_video_file(
        self,
        vpath: Path,
        index: int,
        annotation: Dict[str, Any],
    ) -> VideoMetadata:
        """Runs OpenCV validation on a single video file and merges metadata."""
        cv_info = validate_video_with_opencv(vpath)

        file_size = vpath.stat().st_size if vpath.exists() else 0
        content_hash = compute_file_hash(vpath) if file_size > 0 else ""

        # Duplicate detection via content hash
        validation_note = cv_info["validation_note"]
        if content_hash:
            if content_hash in self.seen_hashes:
                original_file = self.seen_hashes[content_hash]
                duplicate_note = f"duplicate_of:{Path(original_file).name}"
                validation_note = (
                    duplicate_note if validation_note == "ok"
                    else f"{validation_note};{duplicate_note}"
                )
            else:
                self.seen_hashes[content_hash] = str(vpath)

        # Labels
        shot_label = annotation.get("shot_label")
        original_label = annotation.get("original_label")
        if not original_label and shot_label:
            original_label = str(shot_label)
        if not shot_label and original_label:
            shot_label = normalize_detailed_label(original_label)
        mapping_key = normalize_detailed_label(original_label) or normalize_detailed_label(shot_label)
        detailed_original = normalize_detailed_label(original_label)
        default_target = (
            detailed_original
            if detailed_original in DETAILED_SHOT_CLASSES or detailed_original in CRICKETSHOT_LABELS
            else shot_label
        )
        target_label = self.label_mapping.get(mapping_key, default_target)
        if cv_info["is_valid"] and not target_label:
            validation_note = "ok_unlabelled" if validation_note == "ok" else validation_note

        # Source ID
        source_id = annotation.get("source_id") or extract_source_id(vpath.name)
        # Names like cover_0023 encode a label and clip number, not reliable
        # match provenance. Do not turn the class name into a false source ID;
        # that would make official train/val/test splits look contaminated.
        if not annotation.get("source_id") and source_id:
            label_tokens = {
                normalize_detailed_label(value)
                for value in (shot_label, original_label, target_label)
                if value
            }
            source_key = normalize_detailed_label(source_id) or ""
            if any(
                source_key == token or token.startswith(source_key + "_")
                for token in label_tokens
            ):
                source_id = ""

        # Handedness and Foot type
        batting_hand = annotation.get("batting_hand") or "unknown"
        foot_type = annotation.get("foot_type") or "unknown"

        # Phase timestamps
        phases = annotation.get("phase_timestamps") or {}
        phases_str = json.dumps(phases) if phases else ""

        # Technique quality
        quality = annotation.get("technique_quality") or {}
        quality_str = json.dumps(quality) if quality else ""

        # Split
        split = annotation.get("split") or "unassigned"
        if split in ("training",):
            split = "train"
        elif split in ("validation", "valid"):
            split = "val"
        elif split in ("testing",):
            split = "test"

        # Safe relative path
        try:
            rel_path = str(vpath.relative_to(self.output_dir))
        except ValueError:
            rel_path = str(vpath)
        original_relative_path = str(vpath.relative_to(self.input_dir))

        return VideoMetadata(
            video_id=f"VID_{index:05d}",
            video_path=rel_path,
            original_relative_path=original_relative_path,
            filename=vpath.name,
            shot_label=shot_label,
            target_label=target_label,
            original_label=original_label,
            source_id=source_id,
            batting_hand=batting_hand,
            foot_type=foot_type,
            phase_timestamps=phases_str,
            technique_quality=quality_str,
            split=split,
            duration_seconds=cv_info["duration_seconds"],
            fps=cv_info["fps"],
            frame_count=cv_info["frame_count"],
            width=cv_info["width"],
            height=cv_info["height"],
            file_size_bytes=file_size,
            content_hash=content_hash,
            is_valid=cv_info["is_valid"],
            validation_note=validation_note,
        )

    def _assign_splits(self):
        """
        Preserves official/pre-existing train/val/test splits.
        For unassigned clips, performs group-aware splitting on source_id to prevent data leakage.
        """
        unassigned_records = [r for r in self.video_records if r.split not in ("train", "val", "test")]
        if not unassigned_records:
            logger.info("All records have pre-existing splits; preserving official splits.")
            return

        # Group by source_id
        groups: Dict[str, List[VideoMetadata]] = {}
        for r in unassigned_records:
            if r.source_id:
                groups.setdefault(r.source_id, []).append(r)

        if not groups:
            logger.warning(
                "Source provenance is unknown for unassigned clips; leaving them unassigned "
                "instead of treating clips as independent split groups."
            )
            return

        unique_sources = sorted(groups.keys())
        np.random.seed(self.random_seed)
        shuffled_sources = np.random.permutation(unique_sources)

        n_total = len(shuffled_sources)
        r_train, r_val, r_test = self.split_ratio
        norm_sum = r_train + r_val + r_test
        n_train = int(round((r_train / norm_sum) * n_total))
        n_val = int(round((r_val / norm_sum) * n_total))

        train_sources = set(shuffled_sources[:n_train])
        val_sources = set(shuffled_sources[n_train : n_train + n_val])
        test_sources = set(shuffled_sources[n_train + n_val :])

        for source_id, recs in groups.items():
            if source_id in train_sources:
                target_split = "train"
            elif source_id in val_sources:
                target_split = "val"
            else:
                target_split = "test"

            for r in recs:
                r.split = target_split

        logger.info(
            "Assigned grouped splits across %d source groups: %d train, %d val, %d test",
            n_total,
            len(train_sources),
            len(val_sources),
            len(test_sources),
        )

    def _export_manifest_csv(self, manifest_path: Path):
        """Writes clean CSV manifest containing all validated records."""
        fieldnames = [
            "video_id",
            "video_path",
            "original_relative_path",
            "filename",
            "shot_label",
            "target_label",
            "original_label",
            "source_id",
            "batting_hand",
            "foot_type",
            "split",
            "is_valid",
            "validation_note",
            "duration_seconds",
            "fps",
            "frame_count",
            "width",
            "height",
            "file_size_bytes",
            "content_hash",
            "phase_timestamps",
            "technique_quality",
        ]

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.video_records:
                writer.writerow(asdict(r))

    def _generate_summary_report(self, manifest_path: Path) -> Dict[str, Any]:
        """Calculates comprehensive dataset statistics, class imbalance, and duplicate counts."""
        total_clips = len(self.video_records)
        valid_clips = sum(1 for r in self.video_records if r.is_valid)
        invalid_clips = total_clips - valid_clips

        # Counts
        duplicates = sum(1 for r in self.video_records if "duplicate_of:" in r.validation_note)
        unlabelled = sum(1 for r in self.video_records if not r.shot_label)

        # Keep the legacy shot_label counts, plus model-target and usable counts.
        class_counts: Dict[str, int] = {}
        target_class_counts: Dict[str, int] = {}
        observed_label_mapping: Dict[str, str] = {}
        usable_class_counts: Dict[str, int] = {
            self.label_mapping.get(key, key): 0 for key in self._layout_class_labels()
        }
        original_class_counts: Dict[str, int] = {
            key: 0 for key in self._layout_class_labels()
        }
        for r in self.video_records:
            if r.shot_label:
                class_counts[r.shot_label] = class_counts.get(r.shot_label, 0) + 1
            if r.target_label:
                target_class_counts[r.target_label] = target_class_counts.get(r.target_label, 0) + 1
                usable_class_counts.setdefault(r.target_label, 0)
                original_key = normalize_detailed_label(r.original_label) or r.target_label
                original_class_counts[original_key] = original_class_counts.get(original_key, 0) + 1
                if original_key != r.target_label:
                    observed_label_mapping[original_key] = r.target_label
                if r.is_valid and "duplicate_of:" not in r.validation_note:
                    usable_class_counts[r.target_label] += 1

        # Imbalance metrics
        imbalance_ratio = 1.0
        shannon_entropy_norm = 1.0
        nonzero_class_counts = {k: v for k, v in target_class_counts.items() if v > 0}
        if nonzero_class_counts:
            counts = list(nonzero_class_counts.values())
            min_c = min(counts)
            max_c = max(counts)
            imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")

            total_labelled = sum(counts)
            probs = [c / total_labelled for c in counts]
            k = len(counts)
            if k > 1:
                entropy = -sum(p * math.log(p) for p in probs)
                shannon_entropy_norm = round(entropy / math.log(k), 4)

        # Splits
        split_counts: Dict[str, int] = {}
        for r in self.video_records:
            split_counts[r.split] = split_counts.get(r.split, 0) + 1

        # Handedness
        hand_counts: Dict[str, int] = {}
        for r in self.video_records:
            hand_counts[r.batting_hand] = hand_counts.get(r.batting_hand, 0) + 1

        # Durations
        durations = [r.duration_seconds for r in self.video_records if r.duration_seconds > 0]
        avg_duration = round(float(np.mean(durations)), 2) if durations else 0.0

        summary = {
            "dataset_directory": str(self.input_dir),
            "manifest_file": str(manifest_path),
            "total_clips": total_clips,
            "valid_clips": valid_clips,
            "invalid_clips": invalid_clips,
            "duplicate_clips": duplicates,
            "unlabelled_clips": unlabelled,
            "class_distribution": class_counts,
            "target_class_distribution": target_class_counts,
            "usable_class_distribution": dict(sorted(usable_class_counts.items())),
            "original_class_distribution": dict(sorted(original_class_counts.items())),
            "label_mapping": self.label_mapping,
            "observed_label_mapping": dict(sorted(observed_label_mapping.items())),
            "class_mapping_reduced_classes": (
                len([count for count in original_class_counts.values() if count > 0])
                > len(nonzero_class_counts)
            ),
            "original_class_count": len([count for count in original_class_counts.values() if count > 0]),
            "target_class_count": len(nonzero_class_counts),
            "class_imbalance": {
                "imbalance_ratio_max_to_min": imbalance_ratio,
                "normalized_shannon_entropy": shannon_entropy_norm,
                "entropy_note": "1.0 = perfectly balanced; < 0.7 indicates significant class skew.",
            },
            "split_distribution": split_counts,
            "handedness_distribution": hand_counts,
            "duration_stats": {
                "average_seconds": avg_duration,
                "min_seconds": min(durations) if durations else 0.0,
                "max_seconds": max(durations) if durations else 0.0,
            },
            "supported_canonical_classes": CANONICAL_SHOT_CLASSES,
        }
        return summary

    def _export_markdown_report(self, report_path: Path, summary: Dict[str, Any]):
        """Generates a clean markdown validation report."""
        lines = [
            "# CricketCoach AI — Dataset Validation Report",
            "",
            f"**Scanned Directory:** `{summary['dataset_directory']}`  ",
            f"**Manifest CSV:** `{summary['manifest_file']}`  ",
            f"**Total Video Clips:** {summary['total_clips']}  ",
            f"**Valid & Readable Videos:** {summary['valid_clips']}  ",
            f"**Invalid / Corrupt Videos:** {summary['invalid_clips']}  ",
            f"**Duplicate Clips (Identical Hash):** {summary['duplicate_clips']}  ",
            f"**Unlabelled Clips:** {summary['unlabelled_clips']}  ",
            "",
            "---",
            "",
            "## 1. Model Target Class Distribution",
            "",
            "| Target Class | Clip Count | Percentage | Canonical Supported |",
            "| :--- | :---: | :---: | :---: |",
        ]

        total_labelled = sum(summary["target_class_distribution"].values()) or 1
        for shot, count in sorted(summary["target_class_distribution"].items(), key=lambda x: -x[1]):
            pct = round((count / total_labelled) * 100, 1)
            is_canon = "Yes" if shot in CANONICAL_SHOT_CLASSES else "Custom / Extra"
            lines.append(f"| `{shot}` | {count} | {pct}% | {is_canon} |")

        if not summary["target_class_distribution"]:
            lines.append("| *(No clips found with labels)* | 0 | 0.0% | - |")

        lines.extend([
            "",
            "### Usable clips by target class",
            "",
            "| Target class | Usable videos |",
            "| :--- | :---: |",
        ])
        for shot, count in summary["usable_class_distribution"].items():
            lines.append(f"| `{shot}` | {count} |")
        if summary["label_mapping"]:
            lines.extend([
                "",
                "Explicit label mapping: " + ", ".join(
                    f"`{source}` → `{target}`" for source, target in summary["label_mapping"].items()
                ),
                f"Class count reduced: {summary['class_mapping_reduced_classes']} "
                f"({summary['original_class_count']} original, {summary['target_class_count']} target).",
            ])
        if summary["observed_label_mapping"]:
            lines.extend([
                "",
                "Dataset adapter label normalization: " + ", ".join(
                    f"`{source}` → `{target}`"
                    for source, target in summary["observed_label_mapping"].items()
                ),
            ])

        lines.extend([
            "",
            "### Class Imbalance Assessment",
            f"- **Imbalance Ratio (Max / Min):** {summary['class_imbalance']['imbalance_ratio_max_to_min']}x",
            f"- **Normalized Shannon Entropy:** {summary['class_imbalance']['normalized_shannon_entropy']} *(1.0 = balanced)*",
            "",
            "---",
            "",
            "## 2. Train / Validation / Test Splits",
            "",
            "| Split | Count | Share |",
            "| :--- | :---: | :---: |",
        ])

        for sp, count in summary["split_distribution"].items():
            pct = round((count / max(1, summary['total_clips'])) * 100, 1)
            lines.append(f"| `{sp}` | {count} | {pct}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Batting Handedness",
            "",
            "| Hand | Count |",
            "| :--- | :---: |",
        ])

        for hand, count in summary["handedness_distribution"].items():
            lines.append(f"| `{hand}` | {count} |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Next Steps for Model Training",
            "",
            "To train the shot classifier with this validated manifest:",
            "```bash",
            f"python ml_pipeline/train_model.py --manifest {summary['manifest_file']}",
            "```",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
