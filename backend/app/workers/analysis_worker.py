"""
Background analysis worker.
Processes a video: runs pose extraction, shot classification, and technique coaching.
Updates the AnalysisJob record throughout.
"""
import logging
import threading
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import AnalysisJob, ShotPrediction, PoseResult, TechniqueObservation, AnalysisStatus
from app.ml.pose_extractor import POSE_MAX_FRAMES, extract_pose_from_video
from app.ml.visual_classifier import get_visual_classifier
from app.ml.technique_coach import generate_technique_observations
from app.services.object_storage import materialize_video

logger = logging.getLogger(__name__)


def process_video_analysis(job_id: int):
    """
    Entry point for background video analysis.
    Runs in a separate thread. Uses its own DB session.
    """
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error("Job %s not found", job_id)
            return

        _run_analysis(db, job)
    except Exception as e:
        logger.exception("Unexpected error in analysis worker for job %s: %s", job_id, e)
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = AnalysisStatus.failed
                job.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _update_progress(db: Session, job: AnalysisJob, progress: int, status: Optional[AnalysisStatus] = None):
    job.progress = progress
    if status:
        job.status = status
    db.commit()


def _run_analysis(db: Session, job: AnalysisJob):
    job.status = AnalysisStatus.processing
    job.started_at = datetime.utcnow()
    job.progress = 5
    db.commit()

    video = job.video
    if not video:
        job.status = AnalysisStatus.failed
        job.error_message = "Video record not found"
        db.commit()
        return

    # Remote objects are private in Supabase Storage. Materialize once to a
    # temporary local file because OpenCV and the classifier consume file paths.
    with materialize_video(video.file_path) as local_video_path:
        _run_analysis_for_local_video(db, job, video, local_video_path)


def _run_analysis_for_local_video(db: Session, job: AnalysisJob, video, local_video_path: str):
    # ── Step 1: Pose extraction ───────────────────────────────────────────────
    logger.info("Job %s: Starting pose extraction", job.id)
    _update_progress(db, job, 10)

    try:
        pose_result_data = extract_pose_from_video(local_video_path, max_frames=POSE_MAX_FRAMES)
    except Exception as e:
        pose_result_data = {
            "frame_count": 0,
            "sampled_frames": 0,
            "landmarks_detected": False,
            "detection_confidence": None,
            "pose_landmarks": [],
            "joint_angles": {},
            "movement_features": {},
            "error": str(e),
        }
        logger.warning("Job %s: Pose extraction error: %s", job.id, e)

    _update_progress(db, job, 50)

    # Save pose result
    pose_record = PoseResult(
        job_id=job.id,
        frame_count=pose_result_data.get("frame_count"),
        sampled_frames=pose_result_data.get("sampled_frames"),
        landmarks_detected=pose_result_data.get("landmarks_detected", False),
        pose_landmarks_json=_truncate_pose_data(pose_result_data.get("pose_landmarks", [])),
        joint_angles_json=pose_result_data.get("joint_angles", {}),
        movement_features_json=pose_result_data.get("movement_features", {}),
        detection_confidence=pose_result_data.get("detection_confidence"),
    )
    db.add(pose_record)
    db.commit()

    _update_progress(db, job, 60)

    # ── Step 2: Visual-temporal shot classification ────────────────────────────
    # This model consumes sampled RGB frames from the original video. Pose
    # landmarks remain a separate input to technique coaching below.
    logger.info("Job %s: Running visual-temporal shot classification", job.id)
    classifier = get_visual_classifier()
    prediction_data = classifier.predict_video(local_video_path) if classifier.is_available else {
        "shot_label": None,
        "confidence": None,
        "all_scores": {},
        "model_available": False,
    }
    if prediction_data.get("error"):
        logger.warning("Job %s: Visual classification failed: %s", job.id, prediction_data["error"])

    prediction_record = ShotPrediction(
        job_id=job.id,
        shot_label=prediction_data.get("shot_label"),
        confidence=prediction_data.get("confidence"),
        all_scores=prediction_data.get("all_scores"),
        model_available=prediction_data.get("model_available", False),
    )
    db.add(prediction_record)
    db.commit()

    _update_progress(db, job, 75)

    # ── Step 3: Technique observations ───────────────────────────────────────
    logger.info("Job %s: Generating technique observations", job.id)
    observations = generate_technique_observations(
        pose_result_data,
        batting_hand=video.batting_hand,
        shot_label=prediction_data.get("shot_label"),
    )

    for obs in observations:
        record = TechniqueObservation(
            job_id=job.id,
            category=obs["category"],
            observation=obs["observation"],
            measurement=obs.get("measurement"),
            frame_timestamp=obs.get("frame_timestamp"),
            reliability=obs["reliability"],
            explanation=obs["explanation"],
            source_type=obs["source_type"],
            severity=obs.get("severity", "info"),
            drill_id=obs.get("drill_id"),
        )
        db.add(record)
    db.commit()

    _update_progress(db, job, 95)

    # ── Done ──────────────────────────────────────────────────────────────────
    job.status = AnalysisStatus.completed
    job.completed_at = datetime.utcnow()
    job.progress = 100
    db.commit()
    logger.info("Job %s: Analysis completed successfully", job.id)


def _truncate_pose_data(pose_landmarks: list, max_frames: int = 10) -> list:
    """Keep only a subset of frames to avoid huge DB entries."""
    if not pose_landmarks:
        return []
    step = max(1, len(pose_landmarks) // max_frames)
    return pose_landmarks[::step][:max_frames]


def start_analysis_background(job_id: int):
    """Launch analysis in a daemon thread."""
    t = threading.Thread(target=process_video_analysis, args=(job_id,), daemon=True)
    t.start()
    return t
