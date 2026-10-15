"""SQLAlchemy ORM models for CricketCoach AI."""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, ForeignKeyConstraint, JSON, Enum as SAEnum,
    CheckConstraint, Index, Table, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum


class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.user, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    batting_hand = Column(String(10), default="right", nullable=False)  # right / left
    experience_level = Column(String(20), default="beginner", nullable=False)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisJob", back_populates="user", cascade="all, delete-orphan")
    completed_drills = relationship("CompletedDrill", back_populates="user", cascade="all, delete-orphan")
    session_notes = relationship("SessionNote", back_populates="user", cascade="all, delete-orphan")
    auth_sessions = relationship("AuthSession", back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    """Persisted JWT sessions so logout/revocation works across restarts."""
    __tablename__ = "auth_sessions"

    jti = Column(String(128), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="auth_sessions")

    __table_args__ = (
        Index("ix_auth_sessions_user_expires_at", "user_id", "expires_at"),
    )


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    batting_hand = Column(String(10), default="right", nullable=False)
    recording_angle = Column(String(50), default="side-on", nullable=False)  # side-on, front-on, overhead
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="videos")
    analyses = relationship(
        "AnalysisJob", back_populates="video", cascade="all, delete-orphan",
        foreign_keys="AnalysisJob.video_id",
    )

    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_videos_id_user_id"),
        Index("ix_videos_user_deleted_created", "user_id", "is_deleted", "created_at"),
        CheckConstraint("file_size_bytes > 0", name="ck_videos_file_size_positive"),
    )


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    video_id = Column(Integer, nullable=False)
    status = Column(SAEnum(AnalysisStatus), default=AnalysisStatus.pending, nullable=False)
    progress = Column(Integer, default=0, nullable=False)  # 0-100
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="analyses", foreign_keys=[user_id])
    video = relationship("Video", back_populates="analyses", foreign_keys=[video_id])
    prediction = relationship("ShotPrediction", back_populates="job", uselist=False, cascade="all, delete-orphan")
    pose_result = relationship("PoseResult", back_populates="job", uselist=False, cascade="all, delete-orphan")
    observations = relationship("TechniqueObservation", back_populates="job", cascade="all, delete-orphan")
    model_version = relationship("ModelVersion", secondary="job_model_version", uselist=False)

    __table_args__ = (
        ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE", name="fk_analysis_jobs_user_id_users"),
        ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE", name="fk_analysis_jobs_video_id_videos"),
        ForeignKeyConstraint(
            ["video_id", "user_id"], ["videos.id", "videos.user_id"],
            ondelete="CASCADE", name="fk_analysis_jobs_video_owner",
        ),
        Index("ix_analysis_jobs_video_status", "video_id", "status"),
        Index("ix_analysis_jobs_user_status_completed", "user_id", "status", "completed_at"),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_analysis_jobs_progress_range"),
    )


class ShotPrediction(Base):
    __tablename__ = "shot_predictions"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False)
    shot_label = Column(String(100), nullable=True)   # None if model not available
    confidence = Column(Float, nullable=True)
    all_scores = Column(JSON, nullable=True)  # {label: score, ...}
    model_available = Column(Boolean, default=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("AnalysisJob", back_populates="prediction")

    __table_args__ = (
        UniqueConstraint("job_id", name="uq_shot_predictions_job_id"),
    )


class PoseResult(Base):
    __tablename__ = "pose_results"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False)
    frame_count = Column(Integer, nullable=True)
    sampled_frames = Column(Integer, nullable=True)
    landmarks_detected = Column(Boolean, default=False, nullable=False)
    pose_landmarks_json = Column(JSON, nullable=True)  # key frames pose data
    joint_angles_json = Column(JSON, nullable=True)
    movement_features_json = Column(JSON, nullable=True)
    annotated_frames_paths = Column(JSON, nullable=True)  # list of paths
    detection_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("AnalysisJob", back_populates="pose_result")

    __table_args__ = (
        UniqueConstraint("job_id", name="uq_pose_results_job_id"),
    )


class TechniqueObservation(Base):
    __tablename__ = "technique_observations"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(100), nullable=False)  # e.g., "head_position", "balance"
    observation = Column(Text, nullable=False)
    measurement = Column(String(500), nullable=True)
    frame_timestamp = Column(Float, nullable=True)
    reliability = Column(String(20), nullable=False)  # high / medium / low
    explanation = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False)  # measured_feature / heuristic / validated_reference
    severity = Column(String(20), default="info")  # info / warning / good
    drill_id = Column(Integer, ForeignKey("practice_drills.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("AnalysisJob", back_populates="observations")
    drill = relationship("PracticeDrill")

    __table_args__ = (
        Index("ix_technique_observations_job_created", "job_id", "created_at"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)  # sklearn_rf, cnn, etc.
    file_path = Column(String(1000), nullable=True)
    accuracy = Column(Float, nullable=True)
    precision_macro = Column(Float, nullable=True)
    recall_macro = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    supported_shots = Column(JSON, nullable=True)
    training_samples = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    evaluation_report = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
    )


# Association table for job → model version
job_model_version = Table(
    "job_model_version",
    Base.metadata,
    Column("job_id", Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("model_version_id", Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False),
    Index("ix_job_model_version_model_version_id", "model_version_id"),
)


class PracticeDrill(Base):
    __tablename__ = "practice_drills"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    shot_category = Column(String(100), nullable=True)  # None = general
    technique_focus = Column(String(200), nullable=False)
    difficulty = Column(String(20), default="beginner")  # beginner/intermediate/advanced
    duration_minutes = Column(Integer, default=15)
    description = Column(Text, nullable=False)
    steps = Column(JSON, nullable=False)  # list of step strings
    equipment = Column(JSON, nullable=True)  # list of equipment needed
    video_url = Column(String(500), nullable=True)
    tips = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    completions = relationship("CompletedDrill", back_populates="drill")

    __table_args__ = (
        Index("ix_practice_drills_category_difficulty", "shot_category", "difficulty"),
    )


class CompletedDrill(Base):
    __tablename__ = "completed_drills"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    drill_id = Column(Integer, ForeignKey("practice_drills.id", ondelete="CASCADE"), nullable=False)
    notes = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5
    duration_minutes = Column(Integer, nullable=True)
    completed_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="completed_drills")
    drill = relationship("PracticeDrill", back_populates="completions")

    __table_args__ = (
        Index("ix_completed_drills_user_completed_at", "user_id", "completed_at"),
        CheckConstraint("rating IS NULL OR (rating >= 1 AND rating <= 5)", name="ck_completed_drills_rating_range"),
    )


class SessionNote(Base):
    __tablename__ = "session_notes"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(300), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="session_notes")

    __table_args__ = (
        Index("ix_session_notes_user_created_at", "user_id", "created_at"),
    )
