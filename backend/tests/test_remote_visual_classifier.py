import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.ml.remote_visual_classifier import HuggingFaceZeroGPUClassifier
from app.ml import visual_classifier

VALID_SCORES = {
    label: 0.8 if label == "cover" else 0.2 / 9
    for label in visual_classifier.LABELS
}


def test_remote_classifier_is_unavailable_without_private_space(monkeypatch):
    monkeypatch.setattr(settings, "HF_ZERO_GPU_SPACE_ID", "")
    monkeypatch.setattr(settings, "HF_TOKEN", "")

    classifier = HuggingFaceZeroGPUClassifier()

    assert not classifier.is_available
    assert classifier.predict_video("/missing/video.mp4")["model_available"] is False


def test_classifier_factory_uses_remote_space_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "HF_ZERO_GPU_SPACE_ID", "demo/cricketcoach-inference")
    monkeypatch.setattr(settings, "HF_TOKEN", "private-read-token")
    monkeypatch.setattr(visual_classifier, "_remote_visual_classifier", None)

    classifier = visual_classifier.get_visual_classifier()

    assert isinstance(classifier, HuggingFaceZeroGPUClassifier)
    monkeypatch.setattr(visual_classifier, "_remote_visual_classifier", None)


def test_remote_classifier_calls_private_space_and_validates_prediction(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    calls = {}

    class FakeClient:
        def __init__(self, space_id, hf_token, verbose):
            calls["client"] = (space_id, hf_token, verbose)

        def predict(self, path, api_name):
            calls["predict"] = (path, api_name)
            return {
                "model_available": True,
                "shot_label": "cover",
                "display_name": "Cover Drive",
                "confidence": 0.8,
                "all_scores": VALID_SCORES,
            }

    fake_module = ModuleType("gradio_client")
    fake_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "gradio_client", fake_module)
    monkeypatch.setattr(settings, "HF_ZERO_GPU_SPACE_ID", "demo/cricketcoach-inference")
    monkeypatch.setattr(settings, "HF_TOKEN", "private-read-token")

    result = HuggingFaceZeroGPUClassifier().predict_video(str(video_path))

    assert result["shot_label"] == "cover"
    assert result["model_available"] is True
    assert calls["client"] == ("demo/cricketcoach-inference", "private-read-token", False)
    assert calls["predict"] == (str(video_path), "/predict")


@pytest.mark.parametrize(
    "prediction",
    [
        None,
        {"model_available": False},
        {
            "model_available": True,
            "shot_label": "not-a-shot",
            "confidence": 0.9,
            "all_scores": {},
        },
    ],
)
def test_remote_classifier_surfaces_invalid_space_responses(
    monkeypatch, tmp_path, prediction
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return prediction

    fake_module = ModuleType("gradio_client")
    fake_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "gradio_client", fake_module)
    monkeypatch.setattr(settings, "HF_ZERO_GPU_SPACE_ID", "demo/cricketcoach-inference")
    monkeypatch.setattr(settings, "HF_TOKEN", "private-read-token")

    result = HuggingFaceZeroGPUClassifier().predict_video(str(video_path))

    assert result["model_available"] is True
    assert result.get("shot_label") is None
    assert result["error"]
