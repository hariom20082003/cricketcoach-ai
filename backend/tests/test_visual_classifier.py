"""Tests for versioned video-frame inference and safe artifact loading."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.visual_classifier import LABELS, TORCH_IMPORT_ERROR, VisualShotClassifier


def test_missing_visual_artifact_is_safe_and_unavailable(tmp_path):
    classifier = VisualShotClassifier(model_dir=str(tmp_path))
    assert not classifier.is_available
    assert classifier.predict_video(str(tmp_path / "missing.avi"))["model_available"] is False


def test_saved_model_predicts_a_real_heldout_video():
    if TORCH_IMPORT_ERROR:
        pytest.skip(f"PyTorch inference dependencies are unavailable: {TORCH_IMPORT_ERROR}")
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "backend/app/ml/models"
    video_path = project_root / "dataset/external/cricket-shot/cricketshot/test/sweep/sweep_0001.avi"
    if not video_path.is_file() or not (model_dir / "cricketshot_temporal.pt").is_file():
        pytest.skip("The local trained artifact and downloaded held-out sample are required")

    classifier = VisualShotClassifier(model_dir=str(model_dir))
    assert classifier.is_available, classifier.load_error
    result = classifier.predict_video(str(video_path))
    assert result["model_available"] is True
    assert result["shot_label"] in LABELS
    assert set(result["all_scores"]) == set(LABELS)
    assert sum(result["all_scores"].values()) == pytest.approx(1.0, abs=1e-5)
