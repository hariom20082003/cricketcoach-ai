"""Schema and migration regression checks for the persistent database."""
import os
from pathlib import Path
import subprocess
import sys

from sqlalchemy.orm import configure_mappers

from app.models import Base


def test_orm_relationships_and_owner_integrity_constraints_are_configured():
    configure_mappers()
    tables = Base.metadata.tables
    assert {
        "users", "auth_sessions", "videos", "analysis_jobs", "shot_predictions",
        "pose_results", "technique_observations", "session_notes", "completed_drills",
        "practice_drills", "model_versions", "job_model_version",
    }.issubset(tables)

    owner_fk = next(
        constraint for constraint in tables["analysis_jobs"].foreign_key_constraints
        if constraint.name == "fk_analysis_jobs_video_owner"
    )
    assert [element.parent.name for element in owner_fk.elements] == ["video_id", "user_id"]
    assert [element.target_fullname for element in owner_fk.elements] == ["videos.id", "videos.user_id"]
    assert owner_fk.ondelete == "CASCADE"
    assert any(index.name == "ix_analysis_jobs_user_status_completed" for index in tables["analysis_jobs"].indexes)
    assert any(index.name == "ix_videos_user_deleted_created" for index in tables["videos"].indexes)
    assert any(constraint.name == "uq_shot_predictions_job_id" for constraint in tables["shot_predictions"].constraints)
    assert any(constraint.name == "uq_pose_results_job_id" for constraint in tables["pose_results"].constraints)


def test_fresh_database_migrates_and_matches_orm_metadata(tmp_path):
    backend_dir = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "fresh-schema.sqlite3"
    env = os.environ.copy()
    env.update({
        "APP_ENV": "development",
        "DATABASE_URL": f"sqlite:///{database_path}",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(backend_dir), env.get("PYTHONPATH", "")])),
    })

    def run_alembic(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "alembic", *arguments],
            cwd=backend_dir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    run_alembic("upgrade", "head")
    run_alembic("check")

    from sqlalchemy import create_engine, inspect

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        migrated = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert migrated == set(Base.metadata.tables)
    finally:
        engine.dispose()
