"""Tests for worker failure-state handling without writing to the app database."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import AnalysisJob, AnalysisStatus, User, Video
from app.workers import analysis_worker


def test_unhandled_analysis_error_marks_job_failed(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autoflush=False, autocommit=False, bind=engine)

    db = TestSession()
    user = User(
        email="worker-failure@cricket.ai",
        username="workerfailure",
        hashed_password="test-hash",
    )
    db.add(user)
    db.flush()
    video = Video(
        user_id=user.id,
        filename="failure.avi",
        original_filename="failure.avi",
        file_path=str(tmp_path / "failure.avi"),
        file_size_bytes=1,
    )
    # The worker now checks local-file existence before analysis; this test
    # replaces the analysis steps, so a placeholder file is sufficient.
    (tmp_path / "failure.avi").write_bytes(b"test")
    db.add(video)
    db.flush()
    job = AnalysisJob(user_id=user.id, video_id=video.id, status=AnalysisStatus.pending)
    db.add(job)
    db.commit()
    job_id = job.id
    db.close()

    monkeypatch.setattr(analysis_worker, "SessionLocal", TestSession)
    monkeypatch.setattr(analysis_worker, "extract_pose_from_video", lambda *args, **kwargs: {
        "frame_count": 0,
        "sampled_frames": 0,
        "landmarks_detected": False,
        "detection_confidence": None,
        "pose_landmarks": [],
        "joint_angles": {},
        "movement_features": {},
    })

    class AvailableClassifier:
        is_available = True

        def predict_video(self, _path):
            return {
                "shot_label": "cover",
                "confidence": 0.9,
                "all_scores": {"cover": 1.0},
                "model_available": True,
            }

    monkeypatch.setattr(analysis_worker, "get_visual_classifier", lambda: AvailableClassifier())

    def fail_observation_generation(*args, **kwargs):
        raise RuntimeError("controlled analysis failure")

    monkeypatch.setattr(analysis_worker, "generate_technique_observations", fail_observation_generation)

    analysis_worker.process_video_analysis(job_id)

    db = TestSession()
    try:
        failed_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).one()
        assert failed_job.status == AnalysisStatus.failed
        assert failed_job.error_message == "controlled analysis failure"
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_missing_classifier_fails_job_without_creating_pose_only_result(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autoflush=False, autocommit=False, bind=engine)

    db = TestSession()
    user = User(
        email="worker-no-model@cricket.ai",
        username="workernomodel",
        hashed_password="test-hash",
    )
    db.add(user)
    db.flush()
    video = Video(
        user_id=user.id,
        filename="no-model.avi",
        original_filename="no-model.avi",
        file_path=str(tmp_path / "no-model.avi"),
        file_size_bytes=1,
    )
    (tmp_path / "no-model.avi").write_bytes(b"test")
    db.add(video)
    db.flush()
    job = AnalysisJob(user_id=user.id, video_id=video.id, status=AnalysisStatus.pending)
    db.add(job)
    db.commit()
    job_id = job.id
    db.close()

    class MissingClassifier:
        is_available = False

    monkeypatch.setattr(analysis_worker, "SessionLocal", TestSession)
    monkeypatch.setattr(analysis_worker, "get_visual_classifier", lambda: MissingClassifier())
    monkeypatch.setattr(
        analysis_worker,
        "extract_pose_from_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pose extraction must not run without the visual classifier")
        ),
    )

    try:
        analysis_worker.process_video_analysis(job_id)
        db = TestSession()
        try:
            failed_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).one()
            assert failed_job.status == AnalysisStatus.failed
            assert "visual classifier is not loaded" in failed_job.error_message
            assert db.query(analysis_worker.PoseResult).filter_by(job_id=job_id).count() == 0
            assert db.query(analysis_worker.ShotPrediction).filter_by(job_id=job_id).count() == 0
        finally:
            db.close()
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
