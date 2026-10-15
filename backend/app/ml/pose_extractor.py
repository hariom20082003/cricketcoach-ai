"""
Pose extraction using MediaPipe Pose.
Extracts landmarks, computes joint angles, and produces movement features.
"""
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

POSE_MAX_FRAMES = 30
FEATURE_SCHEMA_NAME = "cricketcoach_pose_features"
FEATURE_SCHEMA_VERSION = 1
FEATURE_NAMES = [
    "elbow_primary_mean", "elbow_primary_std", "elbow_secondary_mean", "elbow_secondary_std",
    "knee_primary_mean", "knee_secondary_mean", "shoulder_primary_mean", "shoulder_secondary_mean",
    "hip_primary_mean", "hip_secondary_mean", "spine_angle_mean", "spine_angle_std",
    "wrist_velocity_mean", "wrist_velocity_max", "head_stability_x", "head_stability_y",
    "stance_width_mean", "stance_width_std",
]

# MediaPipe landmark indices
LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """Calculate angle at point b formed by a-b-c. Returns degrees."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angle)


def landmark_to_list(lm) -> List[float]:
    return [lm.x, lm.y, lm.z, lm.visibility]


def extract_pose_from_video(video_path: str, max_frames: int = POSE_MAX_FRAMES) -> Dict[str, Any]:
    """
    Extract pose landmarks and compute features from a video file.
    Returns a dict with pose data, joint angles, movement features, and frame snapshots.
    """
    result = {
        "frame_count": 0,
        "sampled_frames": 0,
        "landmarks_detected": False,
        "detection_confidence": None,
        "pose_landmarks": [],   # list of per-frame landmark dicts
        "joint_angles": {},
        "movement_features": {},
        "error": None,
    }

    try:
        import cv2
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            result["error"] = "Cannot open video file"
            return result

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        result["frame_count"] = total_frames

        # Sample frames evenly
        sample_indices = np.linspace(0, max(total_frames - 1, 0), min(max_frames, total_frames), dtype=int)
        result["sampled_frames"] = len(sample_indices)

        all_frames_landmarks = []
        confidence_scores = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb)

                if pose_result.pose_landmarks:
                    lms = pose_result.pose_landmarks.landmark
                    frame_lm = {}
                    for name, idx in LANDMARKS.items():
                        lm = lms[idx]
                        frame_lm[name] = [lm.x, lm.y, lm.z, lm.visibility]
                    all_frames_landmarks.append({
                        "frame_idx": int(frame_idx),
                        "timestamp": float(frame_idx) / fps,
                        "landmarks": frame_lm,
                    })
                    # track confidence
                    visibility_scores = [lms[idx].visibility for idx in LANDMARKS.values()]
                    confidence_scores.append(float(np.mean(visibility_scores)))

        cap.release()

        if not all_frames_landmarks:
            result["error"] = "No pose landmarks detected in any sampled frame"
            return result

        result["landmarks_detected"] = True
        result["pose_landmarks"] = all_frames_landmarks
        result["detection_confidence"] = float(np.mean(confidence_scores))

        # Compute per-frame joint angles
        frame_angles = []
        for frame_data in all_frames_landmarks:
            lm = frame_data["landmarks"]
            angles = _compute_joint_angles(lm)
            frame_angles.append(angles)

        # Aggregate angles (mean across frames)
        result["joint_angles"] = _aggregate_angles(frame_angles)
        result["movement_features"] = _compute_movement_features(all_frames_landmarks, fps)

    except ImportError as e:
        result["error"] = f"Required library not installed: {e}"
    except Exception as e:
        logger.exception(f"Pose extraction failed: {e}")
        result["error"] = str(e)

    return result


def _compute_joint_angles(lm: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    """Compute key joint angles from a single frame's landmarks."""
    angles = {}

    def get_pts(*names):
        pts = [lm.get(n) for n in names]
        if all(p is not None and p[3] > 0.3 for p in pts):  # visibility check
            return [p[:3] for p in pts]
        return None

    # Elbow angles
    pts = get_pts("left_shoulder", "left_elbow", "left_wrist")
    angles["left_elbow"] = calculate_angle(*pts) if pts else None

    pts = get_pts("right_shoulder", "right_elbow", "right_wrist")
    angles["right_elbow"] = calculate_angle(*pts) if pts else None

    # Knee angles
    pts = get_pts("left_hip", "left_knee", "left_ankle")
    angles["left_knee"] = calculate_angle(*pts) if pts else None

    pts = get_pts("right_hip", "right_knee", "right_ankle")
    angles["right_knee"] = calculate_angle(*pts) if pts else None

    # Shoulder angles (arm raise)
    pts = get_pts("left_elbow", "left_shoulder", "left_hip")
    angles["left_shoulder_raise"] = calculate_angle(*pts) if pts else None

    pts = get_pts("right_elbow", "right_shoulder", "right_hip")
    angles["right_shoulder_raise"] = calculate_angle(*pts) if pts else None

    # Hip angle (torso lean)
    pts = get_pts("left_shoulder", "left_hip", "left_knee")
    angles["left_hip"] = calculate_angle(*pts) if pts else None

    pts = get_pts("right_shoulder", "right_hip", "right_knee")
    angles["right_hip"] = calculate_angle(*pts) if pts else None

    # Spine / torso angle (vertical tilt)
    left_sh = lm.get("left_shoulder")
    right_sh = lm.get("right_shoulder")
    left_hip = lm.get("left_hip")
    right_hip = lm.get("right_hip")
    if all(p is not None and p[3] > 0.3 for p in [left_sh, right_sh, left_hip, right_hip]):
        shoulder_mid = [(left_sh[0] + right_sh[0]) / 2, (left_sh[1] + right_sh[1]) / 2]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        dy = shoulder_mid[1] - hip_mid[1]
        dx = shoulder_mid[0] - hip_mid[0]
        angles["spine_angle"] = float(math.degrees(math.atan2(abs(dx), abs(dy))))
    else:
        angles["spine_angle"] = None

    return angles


def _aggregate_angles(frame_angles: List[Dict]) -> Dict[str, Any]:
    """Average angles across frames, report mean and std."""
    if not frame_angles:
        return {}

    keys = frame_angles[0].keys()
    aggregated = {}
    for key in keys:
        vals = [f[key] for f in frame_angles if f.get(key) is not None]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_frames": len(vals),
            }
        else:
            aggregated[key] = None
    return aggregated


def _compute_movement_features(frames_data: List[Dict], fps: float) -> Dict[str, Any]:
    """Compute higher-level movement features across frames."""
    features = {}
    if len(frames_data) < 2:
        return features

    # Wrist velocity (proxy for bat speed)
    wrist_positions = []
    for fd in frames_data:
        lm = fd["landmarks"]
        lw = lm.get("left_wrist")
        rw = lm.get("right_wrist")
        if lw and rw and lw[3] > 0.3 and rw[3] > 0.3:
            wrist_positions.append((fd["timestamp"], lw[:2], rw[:2]))

    if len(wrist_positions) >= 2:
        velocities = []
        for i in range(1, len(wrist_positions)):
            dt = wrist_positions[i][0] - wrist_positions[i - 1][0]
            if dt > 0:
                dl = np.linalg.norm(np.array(wrist_positions[i][1]) - np.array(wrist_positions[i - 1][1]))
                dr = np.linalg.norm(np.array(wrist_positions[i][2]) - np.array(wrist_positions[i - 1][2]))
                velocities.append(float((dl + dr) / (2 * dt)))
        if velocities:
            features["wrist_velocity_mean"] = float(np.mean(velocities))
            features["wrist_velocity_max"] = float(np.max(velocities))

    # Head stability (nose position variance)
    nose_positions = [fd["landmarks"]["nose"][:2] for fd in frames_data if "nose" in fd["landmarks"] and fd["landmarks"]["nose"][3] > 0.3]
    if nose_positions:
        np_arr = np.array(nose_positions)
        features["head_stability_x"] = float(np.std(np_arr[:, 0]))
        features["head_stability_y"] = float(np.std(np_arr[:, 1]))

    # Foot stance width (normalized by hip width)
    stance_widths = []
    for fd in frames_data:
        lm = fd["landmarks"]
        la = lm.get("left_ankle")
        ra = lm.get("right_ankle")
        lh = lm.get("left_hip")
        rh = lm.get("right_hip")
        if la and ra and lh and rh and la[3] > 0.3 and ra[3] > 0.3:
            ankle_width = abs(la[0] - ra[0])
            hip_width = abs(lh[0] - rh[0]) + 1e-6
            stance_widths.append(ankle_width / hip_width)
    if stance_widths:
        features["normalized_stance_width_mean"] = float(np.mean(stance_widths))
        features["normalized_stance_width_std"] = float(np.std(stance_widths))

    # Weight transfer: relative left/right ankle y changes
    lank = [fd["landmarks"].get("left_ankle") for fd in frames_data if fd["landmarks"].get("left_ankle", [0,0,0,0])[3] > 0.3]
    rank = [fd["landmarks"].get("right_ankle") for fd in frames_data if fd["landmarks"].get("right_ankle", [0,0,0,0])[3] > 0.3]
    if lank and rank:
        features["left_ankle_y_mean"] = float(np.mean([a[1] for a in lank]))
        features["right_ankle_y_mean"] = float(np.mean([a[1] for a in rank]))

    return features


def extract_feature_vector(pose_result: Dict[str, Any], batting_hand: str = "right") -> Optional[np.ndarray]:
    """
    Convert pose analysis result into a flat feature vector suitable for ML classification.
    Returns None if insufficient landmarks were detected.
    """
    if not pose_result.get("landmarks_detected"):
        return None

    angles = pose_result.get("joint_angles", {})
    movement = pose_result.get("movement_features", {})

    def angle_mean(key):
        v = angles.get(key)
        return v["mean"] if v else 0.0

    def angle_std(key):
        v = angles.get(key)
        return v["std"] if v else 0.0

    # Mirror features for left-handed batsmen
    if batting_hand == "left":
        features = [
            angle_mean("right_elbow"),
            angle_std("right_elbow"),
            angle_mean("left_elbow"),
            angle_std("left_elbow"),
            angle_mean("right_knee"),
            angle_mean("left_knee"),
            angle_mean("right_shoulder_raise"),
            angle_mean("left_shoulder_raise"),
            angle_mean("right_hip"),
            angle_mean("left_hip"),
        ]
    else:
        features = [
            angle_mean("left_elbow"),
            angle_std("left_elbow"),
            angle_mean("right_elbow"),
            angle_std("right_elbow"),
            angle_mean("left_knee"),
            angle_mean("right_knee"),
            angle_mean("left_shoulder_raise"),
            angle_mean("right_shoulder_raise"),
            angle_mean("left_hip"),
            angle_mean("right_hip"),
        ]

    features += [
        angle_mean("spine_angle"),
        angle_std("spine_angle"),
        movement.get("wrist_velocity_mean", 0.0),
        movement.get("wrist_velocity_max", 0.0),
        movement.get("head_stability_x", 0.0),
        movement.get("head_stability_y", 0.0),
        movement.get("normalized_stance_width_mean", 0.0),
        movement.get("normalized_stance_width_std", 0.0),
    ]

    vector = np.array(features, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Feature schema mismatch: expected {len(FEATURE_NAMES)} values, got {vector.shape}")
    return vector
