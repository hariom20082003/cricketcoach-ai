"""Video upload and management router."""
import os
import uuid
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models import User, Video, AnalysisJob
from app.schemas import VideoOut
from app.services.auth import get_current_user
from app.config import settings
from app.services.object_storage import delete_video as delete_stored_video, store_video, stream_video as stream_private_video
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/videos", tags=["videos"])


def validate_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions


@router.post("/upload", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    batting_hand: str = Form("right"),
    recording_angle: str = Form("side-on"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Validate file extension
    if not file.filename or not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    # Read file content to check size
    content = await file.read()
    if len(content) > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )

    # Validate the bytes before writing them to durable storage.
    ext = file.filename.rsplit(".", 1)[-1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    validation_path = None
    duration, fps, width, height = None, None, None, None
    try:
        import cv2
        with tempfile.NamedTemporaryFile(prefix="cricketcoach-upload-check-", suffix=f".{ext}", delete=False) as check_file:
            validation_path = check_file.name
            check_file.write(content)
        cap = cv2.VideoCapture(validation_path)
        try:
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Uploaded file is not a readable video")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            readable, _ = cap.read()
            if not readable or width <= 0 or height <= 0 or frame_count <= 0:
                raise HTTPException(status_code=400, detail="Uploaded file is not a readable video")
            duration = frame_count / fps if fps and fps > 0 else None
        finally:
            cap.release()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.warning(f"Could not extract video metadata: {e}")
        raise HTTPException(status_code=400, detail="Uploaded file could not be validated as a video")
    finally:
        if validation_path and os.path.exists(validation_path):
            os.remove(validation_path)

    try:
        file_path = store_video(current_user.id, unique_filename, content, file.content_type)
    except Exception as e:
        logger.exception("Could not store uploaded video: %s", e)
        raise HTTPException(status_code=503, detail="Video storage is temporarily unavailable")

    # Save to database
    video = Video(
        user_id=current_user.id,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size_bytes=len(content),
        duration_seconds=duration,
        fps=fps,
        width=width,
        height=height,
        batting_hand=batting_hand,
        recording_angle=recording_angle,
    )
    db.add(video)
    try:
        db.commit()
        db.refresh(video)
    except Exception:
        db.rollback()
        try:
            delete_stored_video(file_path)
        except Exception:
            logger.exception("Could not clean up video after database write failure")
        raise
    return video


@router.get("/", response_model=List[VideoOut])
def list_videos(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    videos = (
        db.query(Video)
        .filter(Video.user_id == current_user.id, Video.is_deleted == False)
        .order_by(Video.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return videos


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    video = db.query(Video).filter(
        Video.id == video_id, Video.user_id == current_user.id, Video.is_deleted == False
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    video = db.query(Video).filter(
        Video.id == video_id, Video.user_id == current_user.id, Video.is_deleted == False
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Delete the private object before purging the associated database rows.
        delete_stored_video(video.file_path)
        db.delete(video)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Could not delete video and its saved analysis: %s", e)
        raise HTTPException(status_code=503, detail="The video and analysis could not be fully deleted")


@router.get("/{video_id}/stream")
def stream_video(video_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    video = db.query(Video).filter(
        Video.id == video_id, Video.user_id == current_user.id, Video.is_deleted == False
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.file_path.startswith("s3://") and not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    if video.file_path.startswith("s3://"):
        return StreamingResponse(stream_private_video(video.file_path), media_type="video/mp4")
    return FileResponse(video.file_path, media_type="video/mp4")
