from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "dev-secret-key-cricketcoach-ai-2024-change-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720  # 12 hours

    DATABASE_URL: str = "sqlite:///./cricketcoach.db"
    DB_POOL_SIZE: int = 2
    DB_MAX_OVERFLOW: int = 0
    DB_POOL_RECYCLE_SECONDS: int = 1800
    UPLOAD_DIR: str = "./uploads"
    VIDEO_STORAGE_BACKEND: str = "local"
    SUPABASE_S3_ENDPOINT: str = ""
    SUPABASE_S3_REGION: str = ""
    SUPABASE_S3_BUCKET: str = ""
    SUPABASE_S3_ACCESS_KEY_ID: str = ""
    SUPABASE_S3_SECRET_ACCESS_KEY: str = ""
    MODEL_ARTIFACT_PREFIX: str = "models"
    VISUAL_MODEL_DIR: Optional[str] = None
    CRICKETCOACH_DEVICE: str = "auto"
    MAX_VIDEO_SIZE_MB: int = 100
    ALLOWED_VIDEO_EXTENSIONS: str = "mp4,mov,avi,webm"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def allowed_extensions(self) -> List[str]:
        return [e.strip() for e in self.ALLOWED_VIDEO_EXTENSIONS.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.strip().lower() == "production"

    def validate_production(self) -> None:
        if not self.is_production:
            return
        if not self.SECRET_KEY or self.SECRET_KEY in {
            "dev-secret-key-cricketcoach-ai-2024-change-in-prod",
            "your-super-secret-key-change-in-production-min-32-chars",
        }:
            raise ValueError("SECRET_KEY must be set to a unique secret in production")
        if len(self.SECRET_KEY.encode("utf-8")) < 32:
            raise ValueError("Production SECRET_KEY must contain at least 32 UTF-8 bytes")
        if not self.DATABASE_URL.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("Production requires a persistent PostgreSQL DATABASE_URL")
        database_url = make_url(self.DATABASE_URL)
        sslmode = database_url.query.get("sslmode")
        if sslmode and sslmode not in {"require", "verify-ca", "verify-full"}:
            raise ValueError("Production PostgreSQL DATABASE_URL must require SSL")
        if self.DB_POOL_SIZE < 1 or self.DB_MAX_OVERFLOW < 0 or self.DB_POOL_RECYCLE_SECONDS < 60:
            raise ValueError("Production database pool settings are invalid")
        if self.VIDEO_STORAGE_BACKEND.strip().lower() != "s3":
            raise ValueError("Production requires private S3-compatible video storage")
        required = {
            "SUPABASE_S3_ENDPOINT": self.SUPABASE_S3_ENDPOINT,
            "SUPABASE_S3_REGION": self.SUPABASE_S3_REGION,
            "SUPABASE_S3_BUCKET": self.SUPABASE_S3_BUCKET,
            "SUPABASE_S3_ACCESS_KEY_ID": self.SUPABASE_S3_ACCESS_KEY_ID,
            "SUPABASE_S3_SECRET_ACCESS_KEY": self.SUPABASE_S3_SECRET_ACCESS_KEY,
        }
        missing = [key for key, value in required.items() if not value.strip()]
        if missing:
            raise ValueError("Production object storage is missing: " + ", ".join(missing))
        origins = self.cors_origins_list
        if not origins or "*" in origins:
            raise ValueError("Production CORS_ORIGINS must list explicit frontend origins")
        if any(not origin.startswith("https://") for origin in origins):
            raise ValueError("Production CORS_ORIGINS must contain HTTPS origins only")
        if not self.SUPABASE_S3_ENDPOINT.startswith("https://"):
            raise ValueError("Production Supabase S3 endpoint must use HTTPS")
        if not self.SUPABASE_S3_ENDPOINT.rstrip("/").endswith("/storage/v1/s3"):
            raise ValueError("Production Supabase S3 endpoint must end in /storage/v1/s3")
        if self.CRICKETCOACH_DEVICE.strip().lower() not in {"auto", "cpu"}:
            raise ValueError("Production CRICKETCOACH_DEVICE must be auto or cpu")

    @property
    def max_video_size_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
settings.validate_production()

# Only prepare local storage when that backend is selected.
if settings.VIDEO_STORAGE_BACKEND.strip().lower() == "local":
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
