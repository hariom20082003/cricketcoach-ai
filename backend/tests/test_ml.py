"""Tests for pose extraction, feature vector extraction, and technique coaching."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.pose_extractor import (
    calculate_angle, extract_feature_vector, _compute_joint_angles,
    _aggregate_angles, _compute_movement_features, FEATURE_NAMES
)
from app.ml.technique_coach import generate_technique_observations
from app.ml.classifier import get_classifier


class TestAngleCalculation:
    def test_right_angle(self):
        """90 degree angle at B."""
        a = [0.0, 1.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 0.0, 0.0]
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1.0

    def test_straight_line(self):
        """180 degree angle (straight line)."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [2.0, 0.0, 0.0]
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 1.0

    def test_45_degree(self):
        """45 degree angle."""
        a = [0.0, 1.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 1.0, 0.0]
        angle = calculate_angle(a, b, c)
        assert 40.0 < angle < 50.0


class TestFeatureExtraction:
    def _make_mock_pose_result(self, landmarks_detected=True, detection_confidence=0.85):
        """Create a mock pose result for testing."""
        mock_angles = {
            "left_elbow": {"mean": 130.0, "std": 10.0, "min": 110.0, "max": 150.0, "n_frames": 10},
            "right_elbow": {"mean": 120.0, "std": 12.0, "min": 100.0, "max": 140.0, "n_frames": 10},
            "left_knee": {"mean": 150.0, "std": 5.0, "min": 140.0, "max": 160.0, "n_frames": 10},
            "right_knee": {"mean": 145.0, "std": 6.0, "min": 135.0, "max": 155.0, "n_frames": 10},
            "left_shoulder_raise": {"mean": 40.0, "std": 8.0, "min": 30.0, "max": 50.0, "n_frames": 10},
            "right_shoulder_raise": {"mean": 35.0, "std": 7.0, "min": 25.0, "max": 45.0, "n_frames": 10},
            "left_hip": {"mean": 160.0, "std": 4.0, "min": 155.0, "max": 165.0, "n_frames": 10},
            "right_hip": {"mean": 158.0, "std": 5.0, "min": 150.0, "max": 165.0, "n_frames": 10},
            "spine_angle": {"mean": 12.0, "std": 3.0, "min": 8.0, "max": 18.0, "n_frames": 10},
        }
        mock_movement = {
            "wrist_velocity_mean": 0.05,
            "wrist_velocity_max": 0.12,
            "head_stability_x": 0.02,
            "head_stability_y": 0.01,
            "normalized_stance_width_mean": 1.5,
            "normalized_stance_width_std": 0.1,
        }
        return {
            "landmarks_detected": landmarks_detected,
            "detection_confidence": detection_confidence,
            "joint_angles": mock_angles,
            "movement_features": mock_movement,
            "pose_landmarks": [],
        }

    def test_feature_vector_right_hand(self):
        pose_result = self._make_mock_pose_result()
        vec = extract_feature_vector(pose_result, batting_hand="right")
        assert vec is not None
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 18  # Expected feature count
        assert len(FEATURE_NAMES) == len(vec)
        assert not np.any(np.isnan(vec))

    def test_feature_vector_left_hand(self):
        pose_result = self._make_mock_pose_result()
        vec = extract_feature_vector(pose_result, batting_hand="left")
        assert vec is not None
        assert len(vec) == 18

    def test_feature_vector_no_landmarks(self):
        pose_result = self._make_mock_pose_result(landmarks_detected=False)
        vec = extract_feature_vector(pose_result)
        assert vec is None

    def test_aggregate_angles(self):
        frames = [
            {"left_elbow": 130.0, "right_elbow": 120.0, "spine_angle": None},
            {"left_elbow": 140.0, "right_elbow": 110.0, "spine_angle": None},
            {"left_elbow": 135.0, "right_elbow": 125.0, "spine_angle": 15.0},
        ]
        result = _aggregate_angles(frames)
        assert "left_elbow" in result
        assert abs(result["left_elbow"]["mean"] - 135.0) < 0.1
        assert result["spine_angle"]["n_frames"] == 1  # Only 1 non-None frame


class TestTechniqueCoach:
    def _make_mock_pose_result(self, landmarks_detected=True, detection_confidence=0.85):
        return {
            "landmarks_detected": landmarks_detected,
            "detection_confidence": detection_confidence,
            "joint_angles": {
                "left_elbow": {"mean": 130.0, "std": 10.0, "min": 110.0, "max": 150.0, "n_frames": 10},
                "right_elbow": {"mean": 120.0, "std": 12.0, "min": 100.0, "max": 140.0, "n_frames": 10},
                "left_knee": {"mean": 150.0, "std": 5.0, "min": 140.0, "max": 160.0, "n_frames": 10},
                "right_knee": {"mean": 145.0, "std": 6.0, "min": 135.0, "max": 155.0, "n_frames": 10},
                "spine_angle": {"mean": 12.0, "std": 3.0, "min": 8.0, "max": 18.0, "n_frames": 10},
            },
            "movement_features": {
                "wrist_velocity_mean": 0.05,
                "wrist_velocity_max": 0.12,
                "head_stability_x": 0.02,
                "normalized_stance_width_mean": 1.5,
                "normalized_stance_width_std": 0.1,
            },
        }

    def test_observations_generated(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result, batting_hand="right")
        assert len(obs) >= 1

    def test_all_observations_have_required_fields(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result, batting_hand="right")
        required_fields = ["category", "observation", "reliability", "explanation", "source_type", "severity"]
        for o in obs:
            for field in required_fields:
                assert field in o, f"Missing field '{field}' in observation: {o}"

    def test_no_landmarks_returns_single_observation(self):
        pose_result = self._make_mock_pose_result(landmarks_detected=False)
        obs = generate_technique_observations(pose_result)
        assert len(obs) == 1
        assert obs[0]["category"] == "pose_detection"
        assert obs[0]["severity"] == "warning"

    def test_low_confidence_adds_caveat(self):
        pose_result = self._make_mock_pose_result(detection_confidence=0.4)
        obs = generate_technique_observations(pose_result)
        categories = [o["category"] for o in obs]
        assert "data_quality" in categories

    def test_methodology_disclaimer_always_present(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result)
        categories = [o["category"] for o in obs]
        assert "methodology" in categories

    def test_source_types_are_valid(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result)
        valid_types = {"measured_feature", "heuristic", "validated_reference"}
        for o in obs:
            assert o["source_type"] in valid_types, f"Invalid source_type: {o['source_type']}"

    def test_reliability_values_are_valid(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result)
        for o in obs:
            assert o["reliability"] in {"high", "medium", "low"}

    def test_severity_values_are_valid(self):
        pose_result = self._make_mock_pose_result()
        obs = generate_technique_observations(pose_result)
        for o in obs:
            assert o["severity"] in {"good", "warning", "info"}


class TestClassifier:
    def test_classifier_graceful_when_no_model(self):
        """Classifier should gracefully report unavailability when no model file exists."""
        classifier = get_classifier()
        # Model may or may not be available in test env
        result = classifier.predict(np.zeros(18))
        # Result must always have the required keys
        assert "model_available" in result
        assert "shot_label" in result
        if not result["model_available"]:
            assert result["shot_label"] is None
            assert result["confidence"] is None

    def test_supported_shots_always_returned(self):
        classifier = get_classifier()
        shots = classifier.supported_shots
        assert len(shots) >= 7
        assert "cover_drive" in shots
