"""
Shot classifier using scikit-learn.
Loads a trained model if available, otherwise gracefully reports unavailability.
"""
import logging
import os
from typing import Dict, Optional, List

import numpy as np
from app.ml.pose_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_NAME,
    FEATURE_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "shot_classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.npy")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

SUPPORTED_SHOTS = [
    "cover_drive",
    "straight_drive",
    "pull",
    "hook",
    "cut",
    "square_cut",
    "late_cut",
    "sweep",
    "reverse_sweep",
    "flick",
    "defensive",
]

SHOT_DISPLAY_NAMES = {
    "cover_drive": "Cover Drive",
    "straight_drive": "Straight Drive",
    "pull": "Pull Shot",
    "hook": "Hook",
    "cut": "Cut Shot",
    "square_cut": "Square Cut",
    "late_cut": "Late Cut",
    "sweep": "Sweep",
    "reverse_sweep": "Reverse Sweep",
    "flick": "Flick",
    "defensive": "Defensive Shot",
}


class ShotClassifier:
    def __init__(self, model_dir: Optional[str] = None):
        model_dir = model_dir or MODEL_DIR
        self._model_path = os.path.join(model_dir, "shot_classifier.joblib")
        self._scaler_path = os.path.join(model_dir, "scaler.joblib")
        self._labels_path = os.path.join(model_dir, "label_classes.npy")
        self._meta_path = os.path.join(model_dir, "model_meta.json")
        self._model = None
        self._scaler = None
        self._classes: Optional[List[str]] = None
        self._meta: Optional[Dict] = None
        self._loaded = False
        self._load()

    def _load(self):
        """Load only complete artifacts matching the active feature schema."""
        try:
            import json
            import joblib

            required_paths = (
                self._model_path,
                self._scaler_path,
                self._labels_path,
                self._meta_path,
            )
            missing = [path for path in required_paths if not os.path.isfile(path)]
            if missing:
                logger.info("Model artifacts incomplete (%s) — predictions disabled", ", ".join(missing))
                return

            model = joblib.load(self._model_path)
            scaler = joblib.load(self._scaler_path)
            classes = np.load(self._labels_path, allow_pickle=True).tolist()
            with open(self._meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            schema = meta.get("feature_schema", {})
            expected_schema = {
                "name": FEATURE_SCHEMA_NAME,
                "version": FEATURE_SCHEMA_VERSION,
                "features": FEATURE_NAMES,
                "dimension": len(FEATURE_NAMES),
            }
            if schema != expected_schema:
                raise ValueError("Model feature schema is missing or incompatible")
            if meta.get("supported_shots") != classes or meta.get("class_labels") != classes:
                raise ValueError("Model metadata class labels do not match label_classes.npy")
            if not meta.get("version"):
                raise ValueError("Model artifact version metadata is missing")
            if not classes or len(set(classes)) != len(classes):
                raise ValueError("Model class labels are empty or duplicated")
            if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
                raise ValueError("Model feature dimension does not match the active feature schema")
            if getattr(scaler, "n_features_in_", None) != len(FEATURE_NAMES):
                raise ValueError("Scaler feature dimension does not match the active feature schema")
            if scaler.__class__.__name__ != "StandardScaler":
                raise ValueError("Unsupported model scaler; expected StandardScaler")
            if not callable(getattr(model, "predict_proba", None)):
                raise ValueError("Loaded model does not support probability predictions")
            model_classes = [int(value) for value in getattr(model, "classes_", [])]
            if model_classes != list(range(len(classes))):
                raise ValueError("Model class indices do not match stored class labels")
            if meta.get("preprocessing") != {"scaler": "StandardScaler", "feature_dtype": "float32"}:
                raise ValueError("Model preprocessing metadata is missing or unsupported")

            self._model = model
            self._scaler = scaler
            self._classes = classes
            self._meta = meta
            self._loaded = True
            logger.info("Shot classifier loaded. Classes: %s", self._classes)

        except Exception as e:
            logger.warning("Failed to load model: %s", e)
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_meta(self) -> Optional[Dict]:
        return self._meta

    @property
    def supported_shots(self) -> List[str]:
        return self._classes or SUPPORTED_SHOTS

    def predict(self, feature_vector: np.ndarray) -> Dict:
        """
        Run inference. Returns:
        {
            "shot_label": str | None,
            "display_name": str | None,
            "confidence": float | None,
            "all_scores": {label: score},
            "model_available": bool,
        }
        """
        if not self.is_available:
            return {
                "shot_label": None,
                "display_name": None,
                "confidence": None,
                "all_scores": {},
                "model_available": False,
            }

        try:
            x = np.asarray(feature_vector, dtype=np.float32)
            if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
                raise ValueError(
                    f"Unsupported feature vector; expected {len(FEATURE_NAMES)} finite values"
                )
            x = self._scaler.transform(x.reshape(1, -1))

            proba = self._model.predict_proba(x)[0]
            predicted_idx = int(np.argmax(proba))
            shot_label = self._classes[predicted_idx]
            confidence = float(proba[predicted_idx])

            all_scores = {
                self._classes[i]: float(proba[i]) for i in range(len(self._classes))
            }

            return {
                "shot_label": shot_label,
                "display_name": SHOT_DISPLAY_NAMES.get(
                    shot_label, shot_label.replace("_", " ").title()
                ),
                "confidence": confidence,
                "all_scores": all_scores,
                "model_available": True,
            }

        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            return {
                "shot_label": None,
                "display_name": None,
                "confidence": None,
                "all_scores": {},
                "model_available": False,
                "error": str(e),
            }


# Singleton instance
_classifier: Optional[ShotClassifier] = None


def get_classifier() -> ShotClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ShotClassifier()
    return _classifier
