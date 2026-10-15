"""Pydantic schemas for CricketCoach AI API."""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    batting_hand: str = "right"
    experience_level: str = "beginner"

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        v = v.strip()
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be 3-30 characters")
        return v

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be no more than 72 UTF-8 bytes")
        return v

    @field_validator("batting_hand")
    @classmethod
    def valid_batting_hand(cls, v):
        if v not in {"right", "left"}:
            raise ValueError("Batting hand must be 'right' or 'left'")
        return v

    @field_validator("experience_level")
    @classmethod
    def valid_experience_level(cls, v):
        if v not in {"beginner", "intermediate", "advanced"}:
            raise ValueError("Experience level must be beginner, intermediate, or advanced")
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    batting_hand: Optional[str] = None
    experience_level: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    batting_hand: str
    experience_level: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


class LoginForm(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def valid_login_password_size(cls, v):
        if not v:
            raise ValueError("Password is required")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be no more than 72 UTF-8 bytes")
        return v


# ── Video ─────────────────────────────────────────────────────────────────────

class VideoOut(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_size_bytes: int
    duration_seconds: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    batting_hand: str
    recording_angle: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisJobOut(BaseModel):
    id: int
    video_id: int
    status: str
    progress: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class ShotPredictionOut(BaseModel):
    shot_label: Optional[str]
    confidence: Optional[float]
    all_scores: Optional[Dict[str, float]]
    model_available: bool

    model_config = {"from_attributes": True}


class JointAngles(BaseModel):
    left_elbow: Optional[float] = None
    right_elbow: Optional[float] = None
    left_knee: Optional[float] = None
    right_knee: Optional[float] = None
    left_shoulder: Optional[float] = None
    right_shoulder: Optional[float] = None
    left_hip: Optional[float] = None
    right_hip: Optional[float] = None
    spine_angle: Optional[float] = None


class PoseResultOut(BaseModel):
    frame_count: Optional[int]
    sampled_frames: Optional[int]
    landmarks_detected: bool
    joint_angles_json: Optional[Dict[str, Any]]
    movement_features_json: Optional[Dict[str, Any]]
    annotated_frames_paths: Optional[List[str]]
    detection_confidence: Optional[float]

    model_config = {"from_attributes": True}


class TechniqueObservationOut(BaseModel):
    id: int
    category: str
    observation: str
    measurement: Optional[str]
    frame_timestamp: Optional[float]
    reliability: str
    explanation: str
    source_type: str
    severity: str
    drill_id: Optional[int]

    model_config = {"from_attributes": True}


class AnalysisResultOut(BaseModel):
    job: AnalysisJobOut
    video: VideoOut
    prediction: Optional[ShotPredictionOut]
    pose_result: Optional[PoseResultOut]
    observations: List[TechniqueObservationOut]

    model_config = {"from_attributes": True}


# ── Drills ────────────────────────────────────────────────────────────────────

class DrillOut(BaseModel):
    id: int
    title: str
    shot_category: Optional[str]
    technique_focus: str
    difficulty: str
    duration_minutes: int
    description: str
    steps: List[str]
    equipment: Optional[List[str]]
    tips: Optional[List[str]]

    model_config = {"from_attributes": True}


class CompleteDrillIn(BaseModel):
    notes: Optional[str] = None
    rating: Optional[int] = None
    duration_minutes: Optional[int] = None

    @field_validator("rating")
    @classmethod
    def rating_range(cls, v):
        if v is not None and not (1 <= v <= 5):
            raise ValueError("Rating must be 1-5")
        return v


class CompletedDrillOut(BaseModel):
    id: int
    drill_id: int
    notes: Optional[str]
    rating: Optional[int]
    duration_minutes: Optional[int]
    completed_at: datetime

    model_config = {"from_attributes": True}


# ── Progress ──────────────────────────────────────────────────────────────────

class ProgressSummary(BaseModel):
    total_analyses: int
    this_week_analyses: int
    total_drills_completed: int
    this_week_drills: int
    most_common_shot: Optional[str]
    average_confidence: Optional[float]
    observations_by_severity: Dict[str, int]
    shot_distribution: Dict[str, int]
    weekly_activity: List[Dict[str, Any]]


# ── Model Status ──────────────────────────────────────────────────────────────

class ModelStatusOut(BaseModel):
    model_available: bool
    model_name: Optional[str]
    version: Optional[str]
    accuracy: Optional[float]
    macro_f1: Optional[float] = None
    architecture: Optional[str] = None
    test_samples: Optional[int] = None
    inference_latency_ms: Optional[float] = None
    dataset: Optional[str] = None
    supported_shots: Optional[List[str]]
    training_samples: Optional[int]
    message: str


# ── Session Notes ─────────────────────────────────────────────────────────────

class SessionNoteIn(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = None


class SessionNoteOut(BaseModel):
    id: int
    title: str
    content: str
    tags: Optional[List[str]]
    created_at: datetime

    model_config = {"from_attributes": True}
