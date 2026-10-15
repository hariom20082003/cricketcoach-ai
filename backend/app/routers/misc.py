"""Drills, progress, model status, and shots guide routers."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
from app.database import get_db
from app.models import (
    User, PracticeDrill, CompletedDrill, AnalysisJob, ShotPrediction,
    TechniqueObservation, ModelVersion, AnalysisStatus, Video
)
from app.schemas import (
    DrillOut, CompleteDrillIn, CompletedDrillOut, ProgressSummary, ModelStatusOut, SessionNoteIn, SessionNoteOut
)
from app.services.auth import get_current_user
from app.ml.visual_classifier import get_visual_classifier

# ── Drills Router ─────────────────────────────────────────────────────────────
drills_router = APIRouter(prefix="/api/drills", tags=["drills"])


@drills_router.get("/", response_model=List[DrillOut])
def list_drills(
    shot_category: Optional[str] = None,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(PracticeDrill)
    if shot_category:
        q = q.filter(PracticeDrill.shot_category == shot_category)
    if difficulty:
        q = q.filter(PracticeDrill.difficulty == difficulty)
    return q.order_by(PracticeDrill.id).all()


@drills_router.get("/{drill_id}", response_model=DrillOut)
def get_drill(drill_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    drill = db.query(PracticeDrill).filter(PracticeDrill.id == drill_id).first()
    if not drill:
        raise HTTPException(status_code=404, detail="Drill not found")
    return drill


@drills_router.post("/{drill_id}/complete", response_model=CompletedDrillOut, status_code=status.HTTP_201_CREATED)
def complete_drill(
    drill_id: int,
    data: CompleteDrillIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    drill = db.query(PracticeDrill).filter(PracticeDrill.id == drill_id).first()
    if not drill:
        raise HTTPException(status_code=404, detail="Drill not found")
    record = CompletedDrill(
        user_id=current_user.id,
        drill_id=drill_id,
        notes=data.notes,
        rating=data.rating,
        duration_minutes=data.duration_minutes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@drills_router.get("/completed/list", response_model=List[CompletedDrillOut])
def list_completed(
    skip: int = 0, limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(CompletedDrill)
        .filter(CompletedDrill.user_id == current_user.id)
        .order_by(CompletedDrill.completed_at.desc())
        .offset(skip).limit(limit).all()
    )


# ── Progress Router ───────────────────────────────────────────────────────────
progress_router = APIRouter(prefix="/api/progress", tags=["progress"])


@progress_router.get("/summary", response_model=ProgressSummary)
def get_progress_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    # Total analyses
    total_analyses = db.query(func.count(AnalysisJob.id)).filter(
        AnalysisJob.user_id == current_user.id,
        AnalysisJob.status == AnalysisStatus.completed,
        AnalysisJob.video.has(Video.is_deleted == False),
    ).scalar() or 0

    week_analyses = db.query(func.count(AnalysisJob.id)).filter(
        AnalysisJob.user_id == current_user.id,
        AnalysisJob.status == AnalysisStatus.completed,
        AnalysisJob.completed_at >= week_ago,
        AnalysisJob.video.has(Video.is_deleted == False),
    ).scalar() or 0

    # Drills
    total_drills = db.query(func.count(CompletedDrill.id)).filter(
        CompletedDrill.user_id == current_user.id
    ).scalar() or 0

    week_drills = db.query(func.count(CompletedDrill.id)).filter(
        CompletedDrill.user_id == current_user.id,
        CompletedDrill.completed_at >= week_ago
    ).scalar() or 0

    # Most common shot
    shot_rows = (
        db.query(ShotPrediction.shot_label, func.count(ShotPrediction.id).label("cnt"))
        .join(AnalysisJob, AnalysisJob.id == ShotPrediction.job_id)
        .filter(
            AnalysisJob.user_id == current_user.id,
            AnalysisJob.status == AnalysisStatus.completed,
            AnalysisJob.video.has(Video.is_deleted == False),
            ShotPrediction.shot_label != None,
        )
        .group_by(ShotPrediction.shot_label)
        .order_by(func.count(ShotPrediction.id).desc())
        .all()
    )
    most_common_shot = shot_rows[0][0] if shot_rows else None

    shot_distribution = {r[0]: r[1] for r in shot_rows if r[0]}

    # Average confidence
    avg_conf = db.query(func.avg(ShotPrediction.confidence)).join(
        AnalysisJob, AnalysisJob.id == ShotPrediction.job_id
    ).filter(
        AnalysisJob.user_id == current_user.id,
        AnalysisJob.status == AnalysisStatus.completed,
        ShotPrediction.confidence != None,
        AnalysisJob.video.has(Video.is_deleted == False),
    ).scalar()

    # Observations by severity
    sev_rows = (
        db.query(TechniqueObservation.severity, func.count(TechniqueObservation.id).label("cnt"))
        .join(AnalysisJob, AnalysisJob.id == TechniqueObservation.job_id)
        .filter(
            AnalysisJob.user_id == current_user.id,
            AnalysisJob.status == AnalysisStatus.completed,
            AnalysisJob.video.has(Video.is_deleted == False),
        )
        .group_by(TechniqueObservation.severity)
        .all()
    )
    observations_by_severity = {r[0]: r[1] for r in sev_rows}

    # Weekly activity (last 7 days)
    weekly = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0)
        day_end = day.replace(hour=23, minute=59, second=59)
        count = db.query(func.count(AnalysisJob.id)).filter(
            AnalysisJob.user_id == current_user.id,
            AnalysisJob.status == AnalysisStatus.completed,
            AnalysisJob.completed_at.between(day_start, day_end),
            AnalysisJob.video.has(Video.is_deleted == False),
        ).scalar() or 0
        weekly.append({"date": day.strftime("%Y-%m-%d"), "analyses": count})

    return ProgressSummary(
        total_analyses=total_analyses,
        this_week_analyses=week_analyses,
        total_drills_completed=total_drills,
        this_week_drills=week_drills,
        most_common_shot=most_common_shot,
        average_confidence=float(avg_conf) if avg_conf else None,
        observations_by_severity=observations_by_severity,
        shot_distribution=shot_distribution,
        weekly_activity=weekly,
    )


# ── Model Status Router ───────────────────────────────────────────────────────
model_router = APIRouter(prefix="/api/model", tags=["model"])


@model_router.get("/status", response_model=ModelStatusOut)
def get_model_status(db: Session = Depends(get_db)):
    classifier = get_visual_classifier()
    meta = classifier.model_meta

    if classifier.is_available:
        return ModelStatusOut(
            model_available=True,
            model_name=meta.get("name", "Shot Classifier") if meta else "Shot Classifier",
            version=meta.get("version", "1.0") if meta else "1.0",
            accuracy=(meta.get("test_metrics") or {}).get("accuracy") if meta else None,
            macro_f1=(meta.get("test_metrics") or {}).get("f1_macro") if meta else None,
            architecture=meta.get("architecture") if meta else None,
            test_samples=meta.get("test_samples") if meta else None,
            inference_latency_ms=(meta.get("test_inference_latency_ms") or {}).get("median") if meta else None,
            dataset=meta.get("dataset", {}).get("id") if meta else None,
            supported_shots=classifier.supported_shots,
            training_samples=meta.get("training_samples") if meta else None,
            message=f"Visual-temporal model is loaded on {classifier.device_name} and ready for video inference.",
        )
    else:
        return ModelStatusOut(
            model_available=False,
            model_name=None,
            version=None,
            accuracy=None,
            macro_f1=None,
            architecture=None,
            test_samples=None,
            inference_latency_ms=None,
            dataset=None,
            supported_shots=classifier.supported_shots,
            training_samples=None,
            message=(
                "No compatible visual-temporal model is loaded. Pose analysis and technique coaching "
                "remain separate; train with ml_pipeline/train_visual_model.py after preparing the dataset."
            ),
        )


# ── Shots Guide Router ────────────────────────────────────────────────────────
shots_router = APIRouter(prefix="/api/shots", tags=["shots"])

SHOTS_GUIDE = [
    {
        "id": "cover_drive",
        "name": "Cover Drive",
        "category": "front_foot",
        "difficulty": "intermediate",
        "description": (
            "The cover drive is one of cricket's most elegant shots, played to a full-pitched delivery "
            "outside off stump. The batter steps forward, drives through the covers with a straight bat, "
            "and follows through high."
        ),
        "key_points": [
            "Step forward to the pitch of the ball",
            "Head and eyes level, over the ball",
            "Elbow leads, bat face straight at impact",
            "Follow through high and toward cover",
            "Weight transfers onto the front foot",
        ],
        "common_errors": [
            "Playing away from the body — increases edge risk",
            "Closed bat face — ball goes to mid-on",
            "Head falling away — loss of control",
        ],
        "suitable_deliveries": ["Full length", "Good length", "Outside off stump"],
    },
    {
        "id": "straight_drive",
        "name": "Straight Drive",
        "category": "front_foot",
        "difficulty": "beginner",
        "description": (
            "Played to a full delivery on or just outside off stump. The bat comes straight through "
            "and the ball is driven back past the bowler."
        ),
        "key_points": [
            "Step forward and get close to the pitch of the ball",
            "Bat swing straight and vertical",
            "High elbow on the downswing",
            "Full follow-through down the ground",
        ],
        "common_errors": [
            "Not getting to the pitch — aerial shot risk",
            "Angling the bat — ball goes off line",
        ],
        "suitable_deliveries": ["Full length on off or middle stump"],
    },
    {
        "id": "pull",
        "name": "Pull Shot",
        "category": "back_foot",
        "difficulty": "intermediate",
        "description": (
            "Played to a short-pitched delivery on leg or middle stump. The batter rocks back and "
            "hits the ball powerfully to the leg side at or above shoulder height."
        ),
        "key_points": [
            "Move back and across to the ball",
            "Eyes level with the ball as it passes",
            "Roll the wrists through impact",
            "Hit with the top hand dominant",
        ],
        "common_errors": [
            "Playing too early — top edge to fine leg",
            "Not getting into line — mis-hit to cow corner",
        ],
        "suitable_deliveries": ["Short-pitched on middle/leg", "Below shoulder height"],
    },
    {
        "id": "cut",
        "name": "Cut Shot",
        "category": "back_foot",
        "difficulty": "intermediate",
        "description": (
            "Played to a short, wide delivery outside off stump. The batter rocks back and "
            "cuts the ball square or backward of square on the off side."
        ),
        "key_points": [
            "Move back and across",
            "Contact the ball beside or in front of the body",
            "Keep the head still and level",
            "Short, punchy swing through point",
        ],
        "common_errors": [
            "Playing too close to the body — cramped, edge risk",
            "Getting under the ball — top edge",
        ],
        "suitable_deliveries": ["Short and wide outside off stump"],
    },
    {
        "id": "sweep",
        "name": "Sweep",
        "category": "front_foot",
        "difficulty": "intermediate",
        "description": (
            "An attacking shot played to spin, with the batter going down on the front knee "
            "and sweeping the ball to the leg side."
        ),
        "key_points": [
            "Get forward to the pitch",
            "Front knee down to the ground",
            "Keep bat angle flat at impact",
            "Aim toward mid-wicket or fine leg",
        ],
        "common_errors": [
            "Trying to sweep balls too full",
            "Opening the bat face too much — top edge",
            "Head coming up too early",
        ],
        "suitable_deliveries": ["Pitched on middle or leg, spinning delivery"],
    },
    {
        "id": "flick",
        "name": "Flick / Leg Glance",
        "category": "back_foot",
        "difficulty": "advanced",
        "description": (
            "A wristy shot played to a straight or leg-side delivery, deflecting the ball "
            "fine or square on the leg side with minimal foot movement."
        ),
        "key_points": [
            "Stand tall on the back foot",
            "Turn the wrists at impact to direct the ball fine",
            "Minimal back-lift required",
            "Keep bat close to the body",
        ],
        "common_errors": [
            "Playing across the line — LBW risk",
            "Too much wrist — ball goes in the air",
        ],
        "suitable_deliveries": ["Full or good length on the stumps / leg stump"],
    },
    {
        "id": "defensive",
        "name": "Defensive Shot",
        "category": "general",
        "difficulty": "beginner",
        "description": (
            "A blocking shot used to defend good-length deliveries and protect the wicket. "
            "The primary aim is survival and to play out good balls."
        ),
        "key_points": [
            "Get behind the line of the ball",
            "Play with a straight bat",
            "Soft hands — absorb the ball, don't push",
            "Head over the ball",
        ],
        "common_errors": [
            "Stiff hands — ball pops up for a catch",
            "Playing away from the body — edge risk",
        ],
        "suitable_deliveries": ["Good length, on a good line"],
    },
]


@shots_router.get("/guide")
def get_shots_guide():
    return SHOTS_GUIDE


@shots_router.get("/guide/{shot_id}")
def get_shot_detail(shot_id: str):
    shot = next((s for s in SHOTS_GUIDE if s["id"] == shot_id), None)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    return shot


# ── Session Notes Router ──────────────────────────────────────────────────────
notes_router = APIRouter(prefix="/api/notes", tags=["notes"])

from app.models import SessionNote


@notes_router.get("/", response_model=List[SessionNoteOut])
def list_notes(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(SessionNote).filter(SessionNote.user_id == current_user.id).order_by(SessionNote.created_at.desc()).all()


@notes_router.post("/", response_model=SessionNoteOut, status_code=status.HTTP_201_CREATED)
def create_note(data: SessionNoteIn, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    note = SessionNote(user_id=current_user.id, title=data.title, content=data.content, tags=data.tags)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@notes_router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    note = db.query(SessionNote).filter(SessionNote.id == note_id, SessionNote.user_id == current_user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
