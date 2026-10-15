"""Authentication utilities: bcrypt password hashing and revocable JWT sessions."""
from datetime import datetime, timedelta, timezone
import secrets
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.models import AuthSession, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "jti": to_encode.get("jti") or secrets.token_urlsafe(32)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def issue_access_token(db: Session, user: User) -> str:
    """Create a signed token and persist its session so it can be revoked on logout."""
    jti = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        {"sub": str(user.id), "jti": jti},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    db.add(AuthSession(jti=jti, user_id=user.id, expires_at=expires_at))
    return token


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return {}


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    subject = payload.get("sub")
    jti = payload.get("jti")
    if not subject or not jti:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    session = db.query(AuthSession).filter(
        AuthSession.jti == jti,
        AuthSession.user_id == user_id,
        AuthSession.revoked_at.is_(None),
        AuthSession.expires_at > datetime.now(timezone.utc),
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is invalid or expired")
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    # An unmapped attribute is request-local and lets /logout revoke this exact token.
    user._access_token_jti = jti
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    subject = payload.get("sub")
    jti = payload.get("jti")
    if not subject or not jti:
        return None
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        return None
    session = db.query(AuthSession).filter(
        AuthSession.jti == jti,
        AuthSession.user_id == user_id,
        AuthSession.revoked_at.is_(None),
        AuthSession.expires_at > datetime.now(timezone.utc),
    ).first()
    if not session:
        return None
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if user:
        user._access_token_jti = jti
    return user
