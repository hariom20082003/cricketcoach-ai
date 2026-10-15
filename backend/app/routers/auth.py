"""Auth router: registration, login, revocable sessions, and profile."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database import get_db
from app.models import AuthSession, User
from app.schemas import UserCreate, UserOut, Token, LoginForm, UserUpdate
from app.services.auth import hash_password, verify_password, issue_access_token, get_current_user
from app.services.object_storage import delete_video as delete_stored_video

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(data: UserCreate, db: Session = Depends(get_db)):
    email = str(data.email).strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=email,
        username=data.username,
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
        batting_hand=data.batting_hand,
        experience_level=data.experience_level,
    )
    db.add(user)
    try:
        db.flush()
        token = issue_access_token(db, user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email or username is already registered")
    return Token(access_token=token, token_type="bearer", user=UserOut.model_validate(user))


@router.post("/login", response_model=Token)
def login(data: LoginForm, db: Session = Depends(get_db)):
    email = str(data.email).strip().lower()
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = issue_access_token(db, user)
    db.commit()
    return Token(access_token=token, token_type="bearer", user=UserOut.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Revoke the presented session in the persistent database."""
    jti = getattr(current_user, "_access_token_jti", None)
    if jti:
        session = db.query(AuthSession).filter(
            AuthSession.jti == jti,
            AuthSession.user_id == current_user.id,
            AuthSession.revoked_at.is_(None),
        ).first()
        if session:
            session.revoked_at = datetime.now(timezone.utc)
            db.commit()
    return None


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Permanently remove this account, its records, and its private uploads."""
    videos = list(current_user.videos)
    try:
        # Object storage is outside the database transaction. Delete first; the
        # S3 DeleteObject operation is idempotent so a retry is safe if the DB
        # transaction subsequently fails.
        for video in videos:
            delete_stored_video(video.file_path)
        db.delete(current_user)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Account data could not be fully deleted. Please retry or contact the demonstration operator.",
        ) from exc
    return None


@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserOut)
def update_me(data: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if data.full_name is not None:
        current_user.full_name = data.full_name
    if data.batting_hand is not None:
        current_user.batting_hand = data.batting_hand
    if data.experience_level is not None:
        current_user.experience_level = data.experience_level
    db.commit()
    db.refresh(current_user)
    return current_user
