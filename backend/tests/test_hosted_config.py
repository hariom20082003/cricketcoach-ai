import pytest

from app.config import Settings
from app.database import normalize_database_url


def hosted_settings(**overrides):
    values = {
        "APP_ENV": "production",
        "SECRET_KEY": "s" * 40,
        "DATABASE_URL": "postgresql://demo@db.example.test:5432/app",
        "VIDEO_STORAGE_BACKEND": "s3",
        "SUPABASE_S3_ENDPOINT": "https://project.storage.supabase.co/storage/v1/s3",
        "SUPABASE_S3_REGION": "ap-south-1",
        "SUPABASE_S3_BUCKET": "private-demo",
        "SUPABASE_S3_ACCESS_KEY_ID": "server-key",
        "SUPABASE_S3_SECRET_ACCESS_KEY": "server-secret",
        "CORS_ORIGINS": "https://college.github.io",
        "CRICKETCOACH_DEVICE": "cpu",
    }
    values.update(overrides)
    return Settings(**values)


def test_valid_hosted_configuration_is_accepted():
    hosted_settings().validate_production()


def test_remote_space_requires_a_token():
    with pytest.raises(ValueError, match="HF_TOKEN"):
        hosted_settings(HF_ZERO_GPU_SPACE_ID="demo/private-space").validate_production()

    hosted_settings(
        HF_ZERO_GPU_SPACE_ID="demo/private-space",
        HF_TOKEN="private-read-token",
    ).validate_production()


def test_missing_model_is_strict_in_production_unless_preview_is_enabled():
    strict = hosted_settings()
    assert strict.ALLOW_MODEL_UNAVAILABLE_PREVIEW is False
    with pytest.raises(RuntimeError, match="refusing an unhealthy production start"):
        strict.validate_model_startup(model_available=False)

    preview = hosted_settings(ALLOW_MODEL_UNAVAILABLE_PREVIEW=True)
    preview.validate_model_startup(model_available=False)


def test_loaded_model_and_nonproduction_startup_do_not_require_preview_flag():
    strict = hosted_settings()
    strict.validate_model_startup(model_available=True)

    development = Settings(APP_ENV="development")
    development.validate_model_startup(model_available=False)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"SECRET_KEY": "too-short"}, "at least 32"),
        ({"DATABASE_URL": "sqlite:///./cricketcoach.db"}, "persistent PostgreSQL"),
        ({"CORS_ORIGINS": "*"}, "explicit frontend origins"),
        ({"CORS_ORIGINS": "http://college.github.io"}, "HTTPS origins"),
        ({"VIDEO_STORAGE_BACKEND": "local"}, "private S3-compatible"),
        ({"CRICKETCOACH_DEVICE": "mps"}, "auto or cpu"),
    ],
)
def test_invalid_hosted_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        hosted_settings(**overrides).validate_production()


def test_render_postgres_url_schemes_use_psycopg3():
    assert normalize_database_url("postgres://demo@db.example.test/app") == (
        "postgresql+psycopg://demo@db.example.test/app"
    )
    assert normalize_database_url("postgresql://demo@db.example.test/app") == (
        "postgresql+psycopg://demo@db.example.test/app"
    )


def test_production_postgres_url_requires_tls_and_preserves_verified_tls():
    assert normalize_database_url(
        "postgres://demo@db.example.test/app", require_ssl=True
    ) == "postgresql+psycopg://demo@db.example.test/app?sslmode=require"
    assert normalize_database_url(
        "postgresql://demo@db.example.test/app", require_ssl=True
    ) == "postgresql+psycopg://demo@db.example.test/app?sslmode=require"
    assert normalize_database_url(
        "postgresql://demo@db.example.test/app?sslmode=verify-full&sslrootcert=%2Ftmp%2Froot.crt",
        require_ssl=True,
    ).endswith("sslmode=verify-full&sslrootcert=%2Ftmp%2Froot.crt")
    with pytest.raises(ValueError, match="require SSL"):
        normalize_database_url(
            "postgresql://demo@db.example.test/app?sslmode=disable",
            require_ssl=True,
        )


def test_production_rejects_insecure_ssl_and_invalid_s3_endpoint():
    with pytest.raises(ValueError, match="require SSL"):
        hosted_settings(DATABASE_URL="postgresql://demo@db.example.test/app?sslmode=disable").validate_production()
    with pytest.raises(ValueError, match="/storage/v1/s3"):
        hosted_settings(SUPABASE_S3_ENDPOINT="https://project.supabase.co").validate_production()
