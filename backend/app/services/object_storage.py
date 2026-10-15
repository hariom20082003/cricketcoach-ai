"""Private local or Supabase S3-compatible video/model artifact storage."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator, Optional
from urllib.parse import urlparse

from app.config import settings


def _is_s3_ref(file_ref: str) -> bool:
    return file_ref.startswith("s3://")


def _parse_s3_ref(file_ref: str) -> tuple[str, str]:
    parsed = urlparse(file_ref)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError("Invalid private object reference")
    if parsed.netloc != settings.SUPABASE_S3_BUCKET:
        raise ValueError("Object reference does not match the configured private bucket")
    return parsed.netloc, key


def _s3_client():
    if settings.VIDEO_STORAGE_BACKEND.strip().lower() != "s3":
        raise RuntimeError("S3 object storage is not configured")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.SUPABASE_S3_ENDPOINT,
        region_name=settings.SUPABASE_S3_REGION,
        aws_access_key_id=settings.SUPABASE_S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.SUPABASE_S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def store_video(user_id: int, filename: str, content: bytes, content_type: Optional[str] = None) -> str:
    """Persist a video and return an internal reference, never a public URL."""
    if settings.VIDEO_STORAGE_BACKEND.strip().lower() == "s3":
        key = f"videos/{user_id}/{filename}"
        _s3_client().put_object(
            Bucket=settings.SUPABASE_S3_BUCKET,
            Key=key,
            Body=content,
            ContentType=content_type or "application/octet-stream",
        )
        return f"s3://{settings.SUPABASE_S3_BUCKET}/{key}"

    path = Path(settings.UPLOAD_DIR) / str(user_id) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path.resolve())


def delete_video(file_ref: str) -> None:
    if _is_s3_ref(file_ref):
        bucket, key = _parse_s3_ref(file_ref)
        _s3_client().delete_object(Bucket=bucket, Key=key)
        return
    try:
        os.remove(file_ref)
    except FileNotFoundError:
        pass


@contextmanager
def materialize_video(file_ref: str) -> Iterator[str]:
    """Yield a local path for OpenCV/PyTorch, downloading private objects temporarily."""
    if not _is_s3_ref(file_ref):
        if not os.path.isfile(file_ref):
            raise FileNotFoundError("Video file is missing")
        yield file_ref
        return

    bucket, key = _parse_s3_ref(file_ref)
    suffix = Path(key).suffix or ".video"
    handle = tempfile.NamedTemporaryFile(prefix="cricketcoach-video-", suffix=suffix, delete=False)
    local_path = handle.name
    handle.close()
    try:
        _s3_client().download_file(bucket, key, local_path)
        yield local_path
    finally:
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass


def stream_video(file_ref: str) -> Iterator[bytes]:
    """Stream an object through the owner-authorized API route."""
    if not _is_s3_ref(file_ref):
        with open(file_ref, "rb") as stream:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        return

    bucket, key = _parse_s3_ref(file_ref)
    body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
    try:
        while True:
            chunk = body.read(64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        body.close()


def download_private_object(key: str, destination: str) -> None:
    """Download a model artifact from the same private bucket at process startup."""
    if key.startswith("/") or ".." in Path(key).parts:
        raise ValueError("Invalid model artifact key")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(settings.SUPABASE_S3_BUCKET, key, destination)
