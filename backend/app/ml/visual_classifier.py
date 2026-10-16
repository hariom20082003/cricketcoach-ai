"""Visual-temporal cricket shot classifier, independent of pose features."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from app.config import settings

MODEL_DIR = Path(settings.VISUAL_MODEL_DIR or Path(__file__).resolve().parent / "models")
WEIGHTS_FILENAME = "cricketshot_temporal.pt"
META_FILENAME = "visual_model_meta.json"
ARCHITECTURE = "efficientnet_b0_frozen_gru"
ARCHITECTURE_VERSION = 1
FRAME_COUNT = 8
IMAGE_SIZE = 224
LABELS = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep",
]
DISPLAY_NAMES = {
    "cover": "Cover Drive",
    "defense": "Defense Shot",
    "flick": "Flick Shot",
    "hook": "Hook Shot",
    "late_cut": "Late Cut",
    "lofted": "Lofted Drive",
    "pull": "Pull Shot",
    "square_cut": "Square Cut",
    "straight": "Straight Drive",
    "sweep": "Sweep Shot",
}

try:
    import torch
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # Keep the API importable if optional ML dependencies are absent.
    torch = None
    nn = None
    EfficientNet_B0_Weights = None
    efficientnet_b0 = None
    TORCH_IMPORT_ERROR = str(exc)


if nn is not None:
    class EfficientNetTemporalModel(nn.Module):
        """Frozen ImageNet EfficientNet frame encoder with a trained bidirectional GRU."""

        def __init__(self, num_classes: int, hidden_size: int = 128, weights=None):
            super().__init__()
            self.encoder = efficientnet_b0(weights=weights)
            self.encoder.classifier = nn.Identity()
            self.encoder.requires_grad_(False)
            self.temporal = nn.GRU(
                input_size=1280,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(0.35)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)

        def train(self, mode: bool = True):
            super().train(mode)
            # The pretrained encoder is frozen; preserve its ImageNet BatchNorm
            # statistics while fitting only the temporal head.
            self.encoder.eval()
            return self

        def encode_frames(self, frame_batch):
            return self.encoder(frame_batch)

        def classify_embeddings(self, sequence_embeddings):
            _, hidden = self.temporal(sequence_embeddings)
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(pooled))

        def forward(self, videos):
            batch, timesteps, channels, height, width = videos.shape
            embeddings = self.encode_frames(videos.reshape(batch * timesteps, channels, height, width))
            embeddings = embeddings.reshape(batch, timesteps, -1)
            return self.classify_embeddings(embeddings)
else:
    class EfficientNetTemporalModel:  # pragma: no cover - dependency-missing fallback
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"PyTorch and TorchVision are unavailable: {TORCH_IMPORT_ERROR}")


def _device():
    if torch is None:
        return None
    requested = settings.CRICKETCOACH_DEVICE.strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transform():
    if torch is None:
        raise RuntimeError(f"PyTorch and TorchVision are unavailable: {TORCH_IMPORT_ERROR}")
    # Pin the weight enum rather than relying on DEFAULT, whose target can change.
    return EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()


def read_video_frames(video_path: str, transform=None, frame_count: int = FRAME_COUNT):
    """Uniformly sample a fixed number of RGB frames and apply ImageNet preprocessing."""
    if torch is None:
        raise RuntimeError(f"PyTorch and TorchVision are unavailable: {TORCH_IMPORT_ERROR}")
    path = Path(video_path)
    if not path.is_file():
        raise ValueError("Video file is missing")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("OpenCV could not open the video stream")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0 or width * height > 4096 * 4096:
            raise ValueError("Video dimensions are missing or exceed the supported limit")
        if total_frames <= 0:
            raise ValueError("Video contains no readable frames")
        indices = np.linspace(0, total_frames - 1, frame_count).round().astype(int)
        transform = transform or make_transform()
        frames = []
        for index in indices:
            # Some AVI headers over-report frame_count by one. If a requested
            # tail frame is beyond the decoded stream, seek back to the nearest
            # readable preceding frame rather than failing the whole clip.
            candidate = int(index)
            ok, bgr = False, None
            while candidate >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                ok, bgr = cap.read()
                if ok and bgr is not None:
                    break
                candidate -= 1
            if not ok or bgr is None:
                raise ValueError(f"Could not decode sampled frame {int(index)}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(rgb)))
        return torch.stack(frames)
    finally:
        cap.release()


class VisualShotClassifier:
    """Loads a versioned visual-temporal artifact and predicts directly from video."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.weights_path = self.model_dir / WEIGHTS_FILENAME
        self.meta_path = self.model_dir / META_FILENAME
        self._model = None
        self._meta: Optional[Dict] = None
        self._device = _device()
        self._transform = None
        self.load_error: Optional[str] = TORCH_IMPORT_ERROR
        self._load()

    def _load(self):
        if torch is None:
            logger.warning("Visual classifier unavailable: %s", self.load_error)
            return
        try:
            self._download_private_artifacts_if_needed()
            if not self.weights_path.is_file() or not self.meta_path.is_file():
                self.load_error = "Visual model artifact or metadata file is missing"
                return
            with self.meta_path.open(encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("architecture") != ARCHITECTURE or meta.get("architecture_version") != ARCHITECTURE_VERSION:
                raise ValueError("Unsupported visual model architecture")
            labels = meta.get("class_labels")
            if labels != LABELS:
                raise ValueError("Visual model labels do not match the configured CricketShot class map")
            if meta.get("frames_per_video") != FRAME_COUNT or meta.get("image_size") != IMAGE_SIZE:
                raise ValueError("Visual model preprocessing dimensions are incompatible")
            preprocessing = meta.get("preprocessing", {})
            if (
                preprocessing.get("version") != 1
                or preprocessing.get("frame_sampling") != "uniform_linspace_including_endpoints"
                or preprocessing.get("resize_and_crop") != "EfficientNet_B0_Weights.IMAGENET1K_V1.transforms"
                or preprocessing.get("image_size") != IMAGE_SIZE
            ):
                raise ValueError("Unsupported visual model preprocessing version")
            model = EfficientNetTemporalModel(num_classes=len(labels))
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(self._device)
            model.eval()
            self._model = model
            self._meta = meta
            self._transform = make_transform()
            self.load_error = None
            logger.info("Visual shot classifier loaded on %s with classes %s", self._device, labels)
        except Exception as exc:
            self._model = None
            self._meta = None
            self.load_error = str(exc)
            logger.exception("Could not load visual shot classifier: %s", exc)

    def _download_private_artifacts_if_needed(self):
        if self.weights_path.is_file() and self.meta_path.is_file():
            return
        if settings.VIDEO_STORAGE_BACKEND.strip().lower() != "s3":
            return
        from app.services.object_storage import download_private_object

        prefix = settings.MODEL_ARTIFACT_PREFIX.strip("/")
        for filename, destination in (
            (WEIGHTS_FILENAME, self.weights_path),
            (META_FILENAME, self.meta_path),
        ):
            if not destination.is_file():
                download_private_object(f"{prefix}/{filename}", str(destination))

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def model_meta(self) -> Optional[Dict]:
        return self._meta

    @property
    def supported_shots(self) -> List[str]:
        return list(self._meta.get("class_labels", [])) if self._meta else list(LABELS)

    @property
    def device_name(self) -> str:
        return str(self._device) if self._device is not None else "unavailable"

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
        try:
            videos = read_video_frames(video_path, transform=self._transform).unsqueeze(0)
            with torch.inference_mode():
                logits = self._model(videos.to(self._device))
                if self._device.type == "mps":
                    torch.mps.synchronize()
                probabilities = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
            predicted = int(np.argmax(probabilities))
            label = self.supported_shots[predicted]
            result.update({
                "shot_label": label,
                "display_name": DISPLAY_NAMES.get(label, label.replace("_", " ").title()),
                "confidence": float(probabilities[predicted]),
                "all_scores": {
                    name: float(probabilities[index])
                    for index, name in enumerate(self.supported_shots)
                },
            })
        except Exception as exc:
            logger.warning("Visual prediction failed for %s: %s", video_path, exc)
            result["error"] = str(exc)
        return result


_visual_classifier: Optional[VisualShotClassifier] = None
_remote_visual_classifier = None


def get_visual_classifier():
    global _visual_classifier, _remote_visual_classifier
    if settings.HF_ZERO_GPU_SPACE_ID.strip():
        if _remote_visual_classifier is None:
            from app.ml.remote_visual_classifier import HuggingFaceZeroGPUClassifier

            _remote_visual_classifier = HuggingFaceZeroGPUClassifier()
        return _remote_visual_classifier
    if _visual_classifier is None:
        _visual_classifier = VisualShotClassifier()
    return _visual_classifier
