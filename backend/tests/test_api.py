"""Backend API tests for CricketCoach AI."""
import pytest
import sys
import os
import tempfile
import cv2
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.config import settings


def create_test_video_bytes():
    handle = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    path = handle.name
    handle.close()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    try:
        assert writer.isOpened(), "OpenCV could not create a test video"
        for frame in range(8):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, frame % 3] = frame * 20
            writer.write(image)
    finally:
        writer.release()
    try:
        with open(path, "rb") as stream:
            return stream.read()
    finally:
        os.remove(path)

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test_cricketcoach.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module", autouse=True)
def setup_db(tmp_path_factory):
    from app import models
    Base.metadata.create_all(bind=engine)
    upload_patch = pytest.MonkeyPatch()
    upload_patch.setattr(settings, "UPLOAD_DIR", str(tmp_path_factory.mktemp("cricketcoach-test-uploads")))
    db = TestingSessionLocal()
    from app.services.seed_drills import seed_drills
    seed_drills(db)
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("./test_cricketcoach.db")
    except:
        pass
    upload_patch.undo()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers(client):
    """Register a user and return auth headers."""
    resp = client.post("/api/auth/register", json={
        "email": "test@cricket.ai",
        "username": "testbatter",
        "password": "test1234",
        "full_name": "Test Batter",
        "batting_hand": "right",
        "experience_level": "beginner"
    })
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "CricketCoach AI" in data["service"]


class TestAuth:
    def test_register_success(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "newuser@cricket.ai",
            "username": "newbatter",
            "password": "secure123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "newuser@cricket.ai"

    def test_register_duplicate_email(self, client):
        client.post("/api/auth/register", json={
            "email": "dup@cricket.ai", "username": "dup1", "password": "pass1234"
        })
        resp = client.post("/api/auth/register", json={
            "email": "dup@cricket.ai", "username": "dup2", "password": "pass1234"
        })
        assert resp.status_code == 409
        assert "Email" in resp.json()["detail"]

    def test_register_short_password(self, client):
        resp = client.post("/api/auth/register", json={
            "email": "short@cricket.ai", "username": "shortpass", "password": "abc"
        })
        assert resp.status_code == 422

    def test_login_success(self, client):
        client.post("/api/auth/register", json={
            "email": "logintest@cricket.ai", "username": "loginbatter", "password": "pass1234"
        })
        resp = client.post("/api/auth/login", json={
            "email": "logintest@cricket.ai", "password": "pass1234"
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password(self, client):
        resp = client.post("/api/auth/login", json={
            "email": "logintest@cricket.ai", "password": "wrongpass"
        })
        assert resp.status_code == 401

    def test_login_rejects_bcrypt_oversized_password_without_server_error(self, client):
        response = client.post("/api/auth/login", json={
            "email": "logintest@cricket.ai", "password": "é" * 37
        })
        assert response.status_code == 422

    def test_registration_hashes_password_and_logout_revokes_session(self, client):
        response = client.post("/api/auth/register", json={
            "email": "logout@cricket.ai",
            "username": "logoutuser",
            "password": "private123",
        })
        assert response.status_code == 201
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/auth/me", headers=headers).status_code == 200

        from app.models import AuthSession, User
        db = TestingSessionLocal()
        try:
            user = db.query(User).filter(User.email == "logout@cricket.ai").one()
            assert user.hashed_password != "private123"
            assert user.hashed_password.startswith(("$2a$", "$2b$", "$2y$"))
            session_count = db.query(AuthSession).filter(AuthSession.user_id == user.id).count()
            assert session_count == 1
        finally:
            db.close()

        assert client.post("/api/auth/logout", headers=headers).status_code == 204
        rejected = client.get("/api/auth/me", headers=headers)
        assert rejected.status_code == 401
        assert "session" in rejected.json()["detail"].lower()

    def test_auth_validation_errors_are_clear(self, client):
        invalid_email = client.post("/api/auth/register", json={
            "email": "not-an-email", "username": "invalidemail", "password": "valid-pass-1"
        })
        assert invalid_email.status_code == 422

        short_password = client.post("/api/auth/register", json={
            "email": "short-auth@cricket.ai", "username": "shortauth", "password": "short"
        })
        assert short_password.status_code == 422

        long_password = client.post("/api/auth/register", json={
            "email": "long-auth@cricket.ai", "username": "longauth", "password": "é" * 37
        })
        assert long_password.status_code == 422

    def test_legacy_or_malformed_token_is_rejected(self, client):
        from app.services.auth import create_access_token
        legacy_token = create_access_token({"sub": "1"})
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {legacy_token}"})
        assert response.status_code == 401

    def test_get_me(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["email"] == "test@cricket.ai"

    def test_unauthorized_access(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


class TestVideos:
    def test_list_videos_empty(self, client, auth_headers):
        resp = client.get("/api/videos/", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_upload_invalid_extension(self, client, auth_headers):
        resp = client.post(
            "/api/videos/upload",
            headers=auth_headers,
            files={"file": ("test.txt", b"not a video", "text/plain")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_valid_video(self, client, auth_headers):
        video_bytes = create_test_video_bytes()
        resp = client.post(
            "/api/videos/upload",
            headers=auth_headers,
            files={"file": ("test.avi", video_bytes, "video/x-msvideo")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        assert resp.status_code == 201
        video_data = resp.json()
        assert video_data["original_filename"] == "test.avi"
        assert video_data["batting_hand"] == "right"
        return video_data

    def test_upload_rejects_corrupt_video_content(self, client, auth_headers):
        resp = client.post(
            "/api/videos/upload",
            headers=auth_headers,
            files={"file": ("broken.avi", b"not a video", "video/x-msvideo")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        assert resp.status_code == 400
        assert "readable video" in resp.json()["detail"]

    def test_get_video(self, client, auth_headers):
        # Upload a decodable video.
        upload_resp = client.post(
            "/api/videos/upload",
            headers=auth_headers,
            files={"file": ("gettest.avi", create_test_video_bytes(), "video/x-msvideo")},
            data={"batting_hand": "left", "recording_angle": "front-on"},
        )
        video_id = upload_resp.json()["id"]
        resp = client.get(f"/api/videos/{video_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == video_id

    def test_cannot_access_other_user_video(self, client):
        # Register a second user
        resp2 = client.post("/api/auth/register", json={
            "email": "other@cricket.ai", "username": "otherbatter", "password": "pass1234"
        })
        other_token = resp2.json()["access_token"]
        other_headers = {"Authorization": f"Bearer {other_token}"}

        # Upload a video as the first user
        test_headers = {"Authorization": ""}
        auth_resp = client.post("/api/auth/login", json={"email": "test@cricket.ai", "password": "test1234"})
        test_headers = {"Authorization": f"Bearer {auth_resp.json()['access_token']}"}
        upload = client.post(
            "/api/videos/upload",
            headers=test_headers,
            files={"file": ("private.avi", create_test_video_bytes(), "video/x-msvideo")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        video_id = upload.json()["id"]

        # Try to access with other user
        resp = client.get(f"/api/videos/{video_id}", headers=other_headers)
        assert resp.status_code == 404  # Should not find it


class TestAnalysisOwnershipAndDeletion:
    def _register(self, client, email, username):
        response = client.post("/api/auth/register", json={
            "email": email,
            "username": username,
            "password": "releasecheck123",
        })
        assert response.status_code == 201
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def _completed_analysis(self, client, headers, filename):
        upload = client.post(
            "/api/videos/upload",
            headers=headers,
            files={"file": (filename, create_test_video_bytes(), "video/x-msvideo")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        assert upload.status_code == 201
        video_id = upload.json()["id"]

        from app.models import Video, AnalysisJob, ShotPrediction, AnalysisStatus
        db = TestingSessionLocal()
        try:
            video = db.query(Video).filter(Video.id == video_id).one()
            job = AnalysisJob(
                user_id=video.user_id,
                video_id=video.id,
                status=AnalysisStatus.completed,
                progress=100,
                completed_at=datetime.utcnow(),
            )
            db.add(job)
            db.flush()
            db.add(ShotPrediction(
                job_id=job.id,
                shot_label="cover",
                confidence=0.9,
                all_scores={"cover": 0.9},
                model_available=True,
            ))
            db.commit()
            db.refresh(job)
            return video.id, job.id, video.file_path
        finally:
            db.close()

    def test_video_and_analysis_results_are_owner_only(self, client):
        owner_headers = self._register(client, "private-owner@cricket.ai", "privateowner")
        other_headers = self._register(client, "private-other@cricket.ai", "privateother")
        video_id, job_id, _ = self._completed_analysis(client, owner_headers, "private.avi")

        assert client.get(f"/api/videos/{video_id}", headers=owner_headers).status_code == 200
        assert client.get(f"/api/videos/{video_id}/stream", headers=owner_headers).status_code == 200
        assert client.get(f"/api/analysis/{job_id}/status", headers=owner_headers).status_code == 200
        assert client.get(f"/api/analysis/{job_id}/results", headers=owner_headers).status_code == 200

        assert client.get(f"/api/videos/{video_id}", headers=other_headers).status_code == 404
        assert client.get(f"/api/videos/{video_id}/stream", headers=other_headers).status_code == 404
        assert client.get(f"/api/analysis/{job_id}/status", headers=other_headers).status_code == 404
        assert client.get(f"/api/analysis/{job_id}/results", headers=other_headers).status_code == 404
        assert client.delete(f"/api/videos/{video_id}", headers=other_headers).status_code == 404
        assert client.get("/api/analysis/history/list", headers=other_headers).json() == []
        # The former unauthenticated /uploads static mount must not expose private files.
        from app.models import Video
        db = TestingSessionLocal()
        try:
            stored_path = db.query(Video).filter(Video.id == video_id).one().file_path
            upload_relative_path = "/" + os.path.relpath(stored_path, settings.UPLOAD_DIR).replace(os.sep, "/")
        finally:
            db.close()
        assert client.get(f"/uploads{upload_relative_path}").status_code == 404

    def test_delete_video_purges_upload_and_saved_analysis(self, client):
        owner_headers = self._register(client, "delete-owner@cricket.ai", "deleteowner")
        video_id, job_id, file_path = self._completed_analysis(client, owner_headers, "delete-me.avi")
        assert os.path.isfile(file_path)

        response = client.delete(f"/api/videos/{video_id}", headers=owner_headers)
        assert response.status_code == 204
        assert not os.path.exists(file_path)
        assert client.get(f"/api/videos/{video_id}", headers=owner_headers).status_code == 404
        assert client.get(f"/api/analysis/{job_id}/status", headers=owner_headers).status_code == 404
        assert client.get(f"/api/analysis/{job_id}/results", headers=owner_headers).status_code == 404
        assert client.get("/api/analysis/history/list", headers=owner_headers).json() == []

        progress = client.get("/api/progress/summary", headers=owner_headers)
        assert progress.status_code == 200
        assert progress.json()["total_analyses"] == 0
        assert progress.json()["shot_distribution"] == {}

        from app.models import Video, AnalysisJob, ShotPrediction, PoseResult, TechniqueObservation
        db = TestingSessionLocal()
        try:
            assert db.query(Video).filter(Video.id == video_id).count() == 0
            assert db.query(AnalysisJob).filter(AnalysisJob.id == job_id).count() == 0
            assert db.query(ShotPrediction).join(AnalysisJob, ShotPrediction.job_id == AnalysisJob.id).filter(AnalysisJob.id == job_id).count() == 0
            assert db.query(PoseResult).filter(PoseResult.job_id == job_id).count() == 0
            assert db.query(TechniqueObservation).filter(TechniqueObservation.job_id == job_id).count() == 0
        finally:
            db.close()

    def test_delete_account_purges_user_data_and_revokes_sessions(self, client):
        owner_headers = self._register(client, "delete-account@cricket.ai", "deleteaccount")
        video_id, job_id, file_path = self._completed_analysis(client, owner_headers, "account-video.avi")
        assert os.path.isfile(file_path)
        from app.models import User
        db = TestingSessionLocal()
        try:
            user_id = db.query(User).filter(User.email == "delete-account@cricket.ai").one().id
            from app.models import PracticeDrill, SessionNote, CompletedDrill
            drill_id = db.query(PracticeDrill).first().id
            db.add(SessionNote(user_id=user_id, title="Private note", content="Remove with account"))
            db.add(CompletedDrill(user_id=user_id, drill_id=drill_id, notes="Private progress"))
            db.commit()
        finally:
            db.close()

        response = client.delete("/api/auth/me", headers=owner_headers)
        assert response.status_code == 204
        assert not os.path.exists(file_path)
        assert client.get("/api/auth/me", headers=owner_headers).status_code == 401

        from app.models import Video, AnalysisJob, AuthSession, SessionNote, CompletedDrill
        db = TestingSessionLocal()
        try:
            assert db.query(User).filter(User.email == "delete-account@cricket.ai").count() == 0
            assert db.query(Video).filter(Video.id == video_id).count() == 0
            assert db.query(AnalysisJob).filter(AnalysisJob.id == job_id).count() == 0
            assert db.query(AuthSession).filter(AuthSession.user_id == user_id).count() == 0
            assert db.query(SessionNote).filter(SessionNote.user_id == user_id).count() == 0
            assert db.query(CompletedDrill).filter(CompletedDrill.user_id == user_id).count() == 0
        finally:
            db.close()


class TestDrills:
    def test_list_drills(self, client, auth_headers):
        resp = client.get("/api/drills/", headers=auth_headers)
        assert resp.status_code == 200
        drills = resp.json()
        assert len(drills) >= 7  # We seeded 7 drills

    def test_get_drill(self, client, auth_headers):
        resp = client.get("/api/drills/1", headers=auth_headers)
        assert resp.status_code == 200
        drill = resp.json()
        assert drill["id"] == 1
        assert "steps" in drill
        assert len(drill["steps"]) > 0

    def test_complete_drill(self, client, auth_headers):
        resp = client.post(
            "/api/drills/1/complete",
            headers=auth_headers,
            json={"notes": "Good session", "rating": 4, "duration_minutes": 20}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["drill_id"] == 1
        assert data["rating"] == 4


class TestModelStatus:
    def test_model_status(self, client):
        resp = client.get("/api/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_available" in data
        assert "message" in data

    def test_analysis_start_returns_503_without_visual_model(self, client, auth_headers, monkeypatch):
        class MissingClassifier:
            is_available = False
            model_meta = None
            supported_shots = [
                "cover", "defense", "flick", "hook", "late_cut",
                "lofted", "pull", "square_cut", "straight", "sweep",
            ]

        monkeypatch.setattr(
            "app.routers.analysis.get_visual_classifier", lambda: MissingClassifier()
        )
        monkeypatch.setattr(
            "app.routers.misc.get_visual_classifier", lambda: MissingClassifier()
        )
        model_status = client.get("/api/model/status")
        assert model_status.status_code == 200
        assert model_status.json()["model_available"] is False
        assert "returns HTTP 503" in model_status.json()["message"]

        upload = client.post(
            "/api/videos/upload",
            headers=auth_headers,
            files={"file": ("no-model.avi", create_test_video_bytes(), "video/x-msvideo")},
            data={"batting_hand": "right", "recording_angle": "side-on"},
        )
        assert upload.status_code == 201
        video_id = upload.json()["id"]

        response = client.post(f"/api/analysis/start/{video_id}", headers=auth_headers)
        assert response.status_code == 503
        assert "visual classifier is not loaded" in response.json()["detail"]
        assert "No analysis job was created" in response.json()["detail"]

        from app.models import AnalysisJob
        db = TestingSessionLocal()
        try:
            assert db.query(AnalysisJob).filter(AnalysisJob.video_id == video_id).count() == 0
        finally:
            db.close()


class TestShotsGuide:
    def test_get_guide(self, client):
        resp = client.get("/api/shots/guide")
        assert resp.status_code == 200
        shots = resp.json()
        assert len(shots) >= 7
        shot_ids = [s["id"] for s in shots]
        assert "cover_drive" in shot_ids

    def test_get_shot_detail(self, client):
        resp = client.get("/api/shots/guide/cover_drive")
        assert resp.status_code == 200
        shot = resp.json()
        assert shot["name"] == "Cover Drive"
        assert "key_points" in shot


class TestProgress:
    def test_progress_summary(self, client, auth_headers):
        resp = client.get("/api/progress/summary", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_analyses" in data
        assert "weekly_activity" in data
        assert len(data["weekly_activity"]) == 7
