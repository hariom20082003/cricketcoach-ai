"""Hugging Face ZeroGPU inference client."""
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.ml.visual_classifier import LABELS, META_FILENAME

logger = logging.getLogger(__name__)


class HuggingFaceZeroGPUClassifier:
    """Use a private Gradio Space for inference without loading weights on Render."""

    def __init__(self):
        self.space_id = settings.HF_ZERO_GPU_SPACE_ID.strip()
        self._token = settings.HF_TOKEN.strip()
        self.load_error: Optional[str] = None
        self._meta: Optional[Dict] = None

        if self.space_id and not self._token:
            self.load_error = "HF_TOKEN is required for the private inference Space"
            return

        metadata_path = Path(__file__).resolve().parent / "models" / META_FILENAME
        if metadata_path.is_file():
            import json

            with metadata_path.open(encoding="utf-8") as stream:
                self._meta = json.load(stream)

    @property
    def is_available(self) -> bool:
        return bool(self.space_id and self._token)

    @property
    def model_meta(self) -> Optional[Dict]:
        return self._meta

    @property
    def supported_shots(self) -> List[str]:
        if self._meta:
            return list(self._meta.get("class_labels", LABELS))
        return list(LABELS)

    @property
    def device_name(self) -> str:
        return "Hugging Face ZeroGPU"

    def predict_video(self, video_path: str) -> Dict:
        result = {
            "shot_label": None,
            "display_name": None,
            "confidence": None,
            "all_scores": {},
            "model_available": self.is_available,
        }
        if not self.is_available:
            return result
        if not Path(video_path).is_file():
            result["error"] = "Video file is missing"
            return result

        try:
            from gradio_client import Client

            client = Client(self.space_id, hf_token=self._token, verbose=False)
            prediction = client.predict(video_path, api_name="/predict")
            if not isinstance(prediction, dict):
                raise ValueError("Inference Space returned an invalid response")
            label = prediction.get("shot_label")
            scores = prediction.get("all_scores")
            confidence = prediction.get("confidence")
            valid_scores = (
                isinstance(scores, dict)
                and set(scores) == set(self.supported_shots)
                and all(
                    isinstance(score, (int, float))
                    and math.isfinite(score)
                    and 0 <= score <= 1
                    for score in scores.values()
                )
                and math.isclose(sum(scores.values()), 1.0, abs_tol=1e-4)
            )
            if (
                prediction.get("model_available") is not True
                or label not in self.supported_shots
                or not valid_scores
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or not math.isclose(confidence, scores[label], abs_tol=1e-6)
            ):
                raise ValueError("Inference Space did not return a valid shot prediction")
            result.update(
                shot_label=label,
                display_name=prediction.get("display_name"),
                confidence=float(confidence),
                all_scores=scores,
            )
        except Exception as exc:
            logger.exception("Hugging Face ZeroGPU inference failed: %s", exc)
            result["error"] = "Remote shot inference failed; check the Space queue and logs."
        return result
