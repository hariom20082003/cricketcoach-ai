"""FastAPI application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends

from app.config import settings
from app.database import SessionLocal, get_db
from app.models import AnalysisJob, AnalysisStatus
from app.routers.auth import router as auth_router
from app.routers.videos import router as videos_router
from app.routers.analysis import router as analysis_router
from app.routers.misc import (
    drills_router, progress_router, model_router, shots_router, notes_router
)
from app.services.seed_drills import seed_drills
from app.ml.visual_classifier import get_visual_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CricketCoach AI API",
    description="AI-powered cricket shot classification and batting technique improvement platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploaded videos are private. They are served only through authenticated
# owner-scoped endpoints; never mount the upload directory as a public static path.

# Register routers
app.include_router(auth_router)
app.include_router(videos_router)
app.include_router(analysis_router)
app.include_router(drills_router)
app.include_router(progress_router)
app.include_router(model_router)
app.include_router(shots_router)
app.include_router(notes_router)


@app.on_event("startup")
def startup_event():
    # Alembic migrations must be applied before the API starts. Production startup
    # intentionally never calls metadata.create_all().
    logger.info("Checking database and seeding practice drills...")
    db = SessionLocal()
    try:
        interrupted_jobs = db.query(AnalysisJob).filter(
            AnalysisJob.status.in_([AnalysisStatus.pending, AnalysisStatus.processing])
        ).update(
            {
                AnalysisJob.status: AnalysisStatus.failed,
                AnalysisJob.error_message: "Analysis was interrupted by a backend restart. Start a new analysis from the saved video.",
            },
            synchronize_session=False,
        )
        if interrupted_jobs:
            db.commit()
            logger.warning("Marked %s interrupted analysis job(s) failed after restart", interrupted_jobs)
        seed_drills(db)
    finally:
        db.close()
    classifier = get_visual_classifier()
    if settings.is_production and not classifier.is_available:
        raise RuntimeError("The visual model could not be loaded; refusing an unhealthy production start")
    logger.info("CricketCoach AI API started successfully.")


@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    classifier = get_visual_classifier()
    return {
        "status": "ok",
        "service": "CricketCoach AI",
        "version": "1.0.0",
        "database": "ok",
        "model_available": classifier.is_available,
    }
