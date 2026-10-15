"""
Technique coaching module.
Generates evidence-based observations from pose measurements and biomechanical rules.
Each observation includes: source type, reliability, measurement, explanation, and drill recommendation.
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Biomechanical reference ranges (documented from cricket coaching literature)
# These are heuristic rules, not validated against expert-annotated data.
REFERENCE_RANGES = {
    "left_elbow_front_arm": {"ideal_min": 110, "ideal_max": 160, "description": "Front arm elbow angle at impact"},
    "back_arm_elbow": {"ideal_min": 90, "ideal_max": 140, "description": "Back arm elbow angle"},
    "front_knee": {"ideal_min": 120, "ideal_max": 170, "description": "Front knee angle at impact"},
    "back_knee": {"ideal_min": 100, "ideal_max": 150, "description": "Back knee at stance"},
    "spine_angle": {"ideal_min": 0, "ideal_max": 20, "description": "Spine tilt from vertical"},
    "stance_width_ratio": {"ideal_min": 1.2, "ideal_max": 2.2, "description": "Stance width / hip width ratio"},
    "head_stability_x": {"threshold": 0.05, "description": "Head lateral movement std"},
}

DRILL_MAP = {
    "head_position": 1,        # Mapped to PracticeDrill IDs (seeded in DB)
    "front_arm": 2,
    "back_arm": 3,
    "stance": 4,
    "balance": 5,
    "follow_through": 6,
    "footwork": 7,
}


def generate_technique_observations(
    pose_result: Dict[str, Any],
    batting_hand: str = "right",
    shot_label: Optional[str] = None,
) -> List[Dict]:
    """
    Generate coaching observations from pose analysis.
    Returns a list of observation dicts.
    All observations are labelled with their source_type and reliability.
    """
    observations = []

    if not pose_result.get("landmarks_detected"):
        observations.append({
            "category": "pose_detection",
            "observation": "Pose landmarks could not be reliably detected in this video.",
            "measurement": None,
            "frame_timestamp": None,
            "reliability": "low",
            "explanation": (
                "The pose estimation model could not detect body landmarks with sufficient confidence. "
                "This may be due to camera angle, lighting, clothing, or partial occlusion. "
                "Try recording from a clear side-on angle with good lighting."
            ),
            "source_type": "measured_feature",
            "severity": "warning",
            "drill_id": None,
        })
        return observations

    angles = pose_result.get("joint_angles", {})
    movement = pose_result.get("movement_features", {})
    detection_confidence = pose_result.get("detection_confidence", 0)

    # Determine which side is "front" (leading) arm based on batting hand
    # Right-handed: left arm is front arm
    front_elbow_key = "left_elbow" if batting_hand == "right" else "right_elbow"
    back_elbow_key = "right_elbow" if batting_hand == "right" else "left_elbow"
    front_knee_key = "left_knee" if batting_hand == "right" else "right_knee"
    back_knee_key = "right_knee" if batting_hand == "right" else "left_knee"

    # ── 1. Head Position & Stability ─────────────────────────────────────────
    head_x_std = movement.get("head_stability_x")
    head_y_std = movement.get("head_stability_y")

    if head_x_std is not None:
        threshold = REFERENCE_RANGES["head_stability_x"]["threshold"]
        if head_x_std > threshold * 2:
            obs = {
                "category": "head_position",
                "observation": "Head shows significant lateral movement across the shot.",
                "measurement": f"Head lateral std: {head_x_std:.3f} (normalized units)",
                "reliability": "medium",
                "explanation": (
                    f"Your head moved laterally by a normalized standard deviation of {head_x_std:.3f}, "
                    "which is above the guideline threshold. In cricket batting, keeping your head still "
                    "and over the ball is fundamental to picking up the ball early and playing the line. "
                    "Excessive head movement can cause misjudgement of length and line."
                ),
                "source_type": "measured_feature",
                "severity": "warning",
                "drill_id": DRILL_MAP["head_position"],
                "frame_timestamp": None,
            }
        elif head_x_std > threshold:
            obs = {
                "category": "head_position",
                "observation": "Head shows some lateral movement. Minor improvement possible.",
                "measurement": f"Head lateral std: {head_x_std:.3f} (normalized units)",
                "reliability": "medium",
                "explanation": (
                    "Your head position is reasonably stable. Aim to minimise lateral movement further — "
                    "try the 'Still Head Drill' for reinforcement."
                ),
                "source_type": "measured_feature",
                "severity": "info",
                "drill_id": DRILL_MAP["head_position"],
                "frame_timestamp": None,
            }
        else:
            obs = {
                "category": "head_position",
                "observation": "Good head stability detected throughout the shot.",
                "measurement": f"Head lateral std: {head_x_std:.3f} (normalized units)",
                "reliability": "medium",
                "explanation": (
                    "Your head appears well-positioned with minimal lateral movement, "
                    "which is a strong indicator of good batting fundamentals."
                ),
                "source_type": "measured_feature",
                "severity": "good",
                "drill_id": None,
                "frame_timestamp": None,
            }
        observations.append(obs)
    else:
        observations.append({
            "category": "head_position",
            "observation": "Head position could not be assessed — nose landmark not reliably visible.",
            "measurement": None,
            "reliability": "low",
            "explanation": "Insufficient visibility of head landmarks. Try recording from a clearer angle.",
            "source_type": "measured_feature",
            "severity": "info",
            "drill_id": None,
            "frame_timestamp": None,
        })

    # ── 2. Front Arm Elbow Angle ──────────────────────────────────────────────
    front_elbow = angles.get(front_elbow_key)
    if front_elbow and isinstance(front_elbow, dict):
        mean_angle = front_elbow["mean"]
        n_frames = front_elbow["n_frames"]
        ref = REFERENCE_RANGES["left_elbow_front_arm"]
        reliability = "high" if n_frames >= 5 else "medium"

        if mean_angle < ref["ideal_min"]:
            observations.append({
                "category": "front_arm",
                "observation": f"Front arm elbow appears to be too bent (mean: {mean_angle:.1f}°).",
                "measurement": f"Front elbow mean: {mean_angle:.1f}° (ideal: {ref['ideal_min']}–{ref['ideal_max']}°, n={n_frames} frames)",
                "reliability": reliability,
                "explanation": (
                    f"Your front elbow angle averaged {mean_angle:.1f}°, below the coaching guideline of "
                    f"{ref['ideal_min']}–{ref['ideal_max']}°. An overly bent front arm can restrict reach "
                    "and reduce bat control in front-foot drives. Try to keep the front elbow higher and "
                    "the arm straighter through impact."
                ),
                "source_type": "heuristic",
                "severity": "warning",
                "drill_id": DRILL_MAP["front_arm"],
                "frame_timestamp": None,
            })
        elif mean_angle > ref["ideal_max"]:
            observations.append({
                "category": "front_arm",
                "observation": f"Front arm elbow is very straight (mean: {mean_angle:.1f}°).",
                "measurement": f"Front elbow mean: {mean_angle:.1f}° (ideal: {ref['ideal_min']}–{ref['ideal_max']}°, n={n_frames} frames)",
                "reliability": reliability,
                "explanation": (
                    f"Your front elbow angle averaged {mean_angle:.1f}°. While a straighter arm provides "
                    "power, an over-extended elbow (above the guideline maximum) can reduce wrist flexibility "
                    "through the stroke. A small bend helps maintain bat control at impact."
                ),
                "source_type": "heuristic",
                "severity": "info",
                "drill_id": DRILL_MAP["front_arm"],
                "frame_timestamp": None,
            })
        else:
            observations.append({
                "category": "front_arm",
                "observation": f"Front arm elbow angle looks good (mean: {mean_angle:.1f}°).",
                "measurement": f"Front elbow mean: {mean_angle:.1f}° (ideal: {ref['ideal_min']}–{ref['ideal_max']}°, n={n_frames} frames)",
                "reliability": reliability,
                "explanation": (
                    "Your front arm elbow angle is within the recommended range, "
                    "supporting good bat control and reach through the stroke."
                ),
                "source_type": "heuristic",
                "severity": "good",
                "drill_id": None,
                "frame_timestamp": None,
            })
    else:
        observations.append({
            "category": "front_arm",
            "observation": "Front arm angle could not be reliably measured.",
            "measurement": None,
            "reliability": "low",
            "explanation": "Front arm landmarks (shoulder, elbow, wrist) were not clearly visible. Ensure full body is in frame.",
            "source_type": "measured_feature",
            "severity": "info",
            "drill_id": DRILL_MAP["front_arm"],
            "frame_timestamp": None,
        })

    # ── 3. Knee Bend / Stance ─────────────────────────────────────────────────
    front_knee = angles.get(front_knee_key)
    if front_knee and isinstance(front_knee, dict):
        mean_angle = front_knee["mean"]
        n_frames = front_knee["n_frames"]
        ref = REFERENCE_RANGES["front_knee"]

        if mean_angle < ref["ideal_min"]:
            observations.append({
                "category": "stance",
                "observation": f"Front knee appears heavily bent (mean: {mean_angle:.1f}°) — may affect balance.",
                "measurement": f"Front knee mean: {mean_angle:.1f}° (guideline: {ref['ideal_min']}–{ref['ideal_max']}°, n={n_frames} frames)",
                "reliability": "medium",
                "explanation": (
                    f"Your front knee averaged {mean_angle:.1f}°. Excessive knee bend can lower your centre "
                    "of gravity too much for some shots, restricting weight transfer and follow-through. "
                    "Aim to keep flexion appropriate to the shot being played."
                ),
                "source_type": "heuristic",
                "severity": "info",
                "drill_id": DRILL_MAP["stance"],
                "frame_timestamp": None,
            })
        else:
            observations.append({
                "category": "stance",
                "observation": f"Front knee angle within expected range (mean: {mean_angle:.1f}°).",
                "measurement": f"Front knee mean: {mean_angle:.1f}° (guideline: {ref['ideal_min']}–{ref['ideal_max']}°, n={n_frames} frames)",
                "reliability": "medium",
                "explanation": "Front knee flexion appears appropriate, supporting balance during the shot.",
                "source_type": "heuristic",
                "severity": "good",
                "drill_id": None,
                "frame_timestamp": None,
            })

    # ── 4. Spine Angle / Upright Posture ─────────────────────────────────────
    spine = angles.get("spine_angle")
    if spine and isinstance(spine, dict):
        mean_spine = spine["mean"]
        ref = REFERENCE_RANGES["spine_angle"]
        if mean_spine > 30:
            observations.append({
                "category": "balance",
                "observation": f"Significant torso tilt detected (spine angle: {mean_spine:.1f}° from vertical).",
                "measurement": f"Spine angle mean: {mean_spine:.1f}°",
                "reliability": "medium",
                "explanation": (
                    f"Your torso averaged {mean_spine:.1f}° from vertical. Excessive forward lean "
                    "can throw off balance and make it harder to play tall and straight. "
                    "Focus on keeping your head upright and your spine vertical, especially for "
                    "straight drives and defensive shots."
                ),
                "source_type": "measured_feature",
                "severity": "warning",
                "drill_id": DRILL_MAP["balance"],
                "frame_timestamp": None,
            })
        else:
            observations.append({
                "category": "balance",
                "observation": f"Torso appears reasonably upright (spine angle: {mean_spine:.1f}°).",
                "measurement": f"Spine angle mean: {mean_spine:.1f}°",
                "reliability": "medium",
                "explanation": "Your posture looks balanced. Maintaining an upright spine helps with line and length judgement.",
                "source_type": "measured_feature",
                "severity": "good",
                "drill_id": None,
                "frame_timestamp": None,
            })

    # ── 5. Stance Width ───────────────────────────────────────────────────────
    stance_w = movement.get("normalized_stance_width_mean")
    if stance_w is not None:
        ref = REFERENCE_RANGES["stance_width_ratio"]
        if stance_w < ref["ideal_min"]:
            observations.append({
                "category": "stance",
                "observation": f"Stance width appears narrow (ratio: {stance_w:.2f}× hip width).",
                "measurement": f"Normalized stance width: {stance_w:.2f}",
                "reliability": "medium",
                "explanation": (
                    f"Your ankle separation was approximately {stance_w:.2f}× your hip width. "
                    "A narrower stance reduces stability, making it harder to withstand pace and generate power. "
                    "Aim for a stance width of around shoulder-width or slightly wider."
                ),
                "source_type": "measured_feature",
                "severity": "warning",
                "drill_id": DRILL_MAP["stance"],
                "frame_timestamp": None,
            })
        elif stance_w > ref["ideal_max"]:
            observations.append({
                "category": "stance",
                "observation": f"Stance width appears wide (ratio: {stance_w:.2f}× hip width).",
                "measurement": f"Normalized stance width: {stance_w:.2f}",
                "reliability": "medium",
                "explanation": (
                    f"Your stance width was {stance_w:.2f}× your hip width, which is on the wider side. "
                    "Overly wide stances can restrict footwork and movement to the pitch of the ball."
                ),
                "source_type": "measured_feature",
                "severity": "info",
                "drill_id": DRILL_MAP["footwork"],
                "frame_timestamp": None,
            })
        else:
            observations.append({
                "category": "stance",
                "observation": f"Stance width looks good (ratio: {stance_w:.2f}× hip width).",
                "measurement": f"Normalized stance width: {stance_w:.2f}",
                "reliability": "medium",
                "explanation": "Your stance width is within the recommended range for stable batting posture.",
                "source_type": "measured_feature",
                "severity": "good",
                "drill_id": None,
                "frame_timestamp": None,
            })

    # ── 6. Wrist Velocity (Proxy for Bat Speed) ────────────────────────────────
    wrist_vel = movement.get("wrist_velocity_max")
    if wrist_vel is not None:
        # Normalized velocity; we can only compare relative, not absolute
        observations.append({
            "category": "follow_through",
            "observation": f"Peak wrist speed recorded (normalized: {wrist_vel:.3f} units/s).",
            "measurement": f"Max wrist velocity: {wrist_vel:.3f} normalized units/s",
            "reliability": "low",
            "explanation": (
                "Wrist velocity is a rough proxy for bat speed. Note: this is computed in normalized "
                "(frame-relative) coordinates and cannot be converted to absolute mph/kmh without camera "
                "calibration. Use this as a relative comparison across your own sessions only."
            ),
            "source_type": "measured_feature",
            "severity": "info",
            "drill_id": None,
            "frame_timestamp": None,
        })

    # ── 7. Confidence Caveat ──────────────────────────────────────────────────
    if detection_confidence is not None and detection_confidence < 0.6:
        observations.append({
            "category": "data_quality",
            "observation": f"Pose detection confidence was low ({detection_confidence:.0%}).",
            "measurement": f"Mean landmark visibility: {detection_confidence:.2f}",
            "reliability": "low",
            "explanation": (
                "The pose estimation model detected landmarks with lower confidence than ideal. "
                "Some measurements above may be less reliable. For best results: "
                "record in bright, natural light; wear close-fitting clothing; "
                "use a side-on camera angle; ensure the full body is visible."
            ),
            "source_type": "measured_feature",
            "severity": "warning",
            "drill_id": None,
            "frame_timestamp": None,
        })

    # ── 8. Limitations Disclosure ─────────────────────────────────────────────
    observations.append({
        "category": "methodology",
        "observation": "These observations are computer-vision measurements, not expert coaching validation.",
        "measurement": None,
        "reliability": "high",
        "explanation": (
            "All technique observations above are generated from automated pose measurements "
            "and documented biomechanical heuristics. They should be treated as training aids, "
            "not as a replacement for guidance from a qualified cricket coach. "
            "Bat-ball contact, timing, and shot selection cannot be assessed from pose alone."
        ),
        "source_type": "heuristic",
        "severity": "info",
        "drill_id": None,
        "frame_timestamp": None,
    })

    return observations
