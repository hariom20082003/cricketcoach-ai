"""Analysis router: start analysis, poll status, get results."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Video, AnalysisJob, AnalysisStatus
from app.schemas import AnalysisJobOut, AnalysisResultOut, ShotPredictionOut, PoseResultOut, TechniqueObservationOut, VideoOut
from app.services.auth import get_current_user
from app.workers.analysis_worker import start_analysis_background
from app.ml.visual_classifier import get_visual_classifier

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/start/{video_id}", response_model=AnalysisJobOut, status_code=status.HTTP_201_CREATED)
def start_analysis(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = db.query(Video).filter(
        Video.id == video_id, Video.user_id == current_user.id, Video.is_deleted == False
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not get_visual_classifier().is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Cricket-shot analysis is unavailable because the visual classifier is not loaded. "
                "No analysis job was created."
            ),
        )

    # Check no active job already running for this video
    existing = db.query(AnalysisJob).filter(
        AnalysisJob.video_id == video_id,
        AnalysisJob.status.in_([AnalysisStatus.pending, AnalysisStatus.processing])
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Analysis already running for this video")

    job = AnalysisJob(
        user_id=current_user.id,
        video_id=video_id,
        status=AnalysisStatus.pending,
        progress=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background processing
    start_analysis_background(job.id)

    return job


@router.get("/{job_id}/status", response_model=AnalysisJobOut)
def get_status(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(AnalysisJob).filter(
        AnalysisJob.id == job_id,
        AnalysisJob.user_id == current_user.id,
        AnalysisJob.video.has(Video.is_deleted == False),
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job


@router.get("/{job_id}/results", response_model=AnalysisResultOut)
def get_results(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = db.query(AnalysisJob).filter(
        AnalysisJob.id == job_id,
        AnalysisJob.user_id == current_user.id,
        AnalysisJob.video.has(Video.is_deleted == False),
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if job.status != AnalysisStatus.completed:
        raise HTTPException(status_code=425, detail=f"Analysis not yet complete. Status: {job.status}")

    return AnalysisResultOut(
        job=AnalysisJobOut.model_validate(job),
        video=VideoOut.model_validate(job.video),
        prediction=ShotPredictionOut.model_validate(job.prediction) if job.prediction else None,
        pose_result=PoseResultOut.model_validate(job.pose_result) if job.pose_result else None,
        observations=[TechniqueObservationOut.model_validate(o) for o in job.observations],
    )


@router.get("/history/list", response_model=List[AnalysisResultOut])
def list_history(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all completed analysis jobs for the current user."""
    jobs = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.user_id == current_user.id,
            AnalysisJob.status == AnalysisStatus.completed,
            AnalysisJob.video.has(Video.is_deleted == False),
        )
        .order_by(AnalysisJob.completed_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    results = []
    for job in jobs:
        if job.video:
            results.append(AnalysisResultOut(
                job=AnalysisJobOut.model_validate(job),
                video=VideoOut.model_validate(job.video),
                prediction=ShotPredictionOut.model_validate(job.prediction) if job.prediction else None,
                pose_result=PoseResultOut.model_validate(job.pose_result) if job.pose_result else None,
                observations=[TechniqueObservationOut.model_validate(o) for o in job.observations],
            ))
    return results
