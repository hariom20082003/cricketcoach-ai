from io import BytesIO
from pathlib import Path
import sys
from types import ModuleType

import pytest

from app.config import settings
from app.services import object_storage


class FakeS3:
    def __init__(self):
        self.puts = []
        self.deletes = []
        self.downloads = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)

    def download_file(self, bucket, key, destination):
        self.downloads.append((bucket, key))
        Path(destination).write_bytes(b"private video")

    def get_object(self, **kwargs):
        return {"Body": BytesIO(b"private stream")}


def test_private_s3_video_storage_is_user_scoped_and_streamed(monkeypatch):
    client = FakeS3()
    monkeypatch.setattr(settings, "VIDEO_STORAGE_BACKEND", "s3")
    monkeypatch.setattr(settings, "SUPABASE_S3_BUCKET", "private-demo")
    monkeypatch.setattr(object_storage, "_s3_client", lambda: client)

    reference = object_storage.store_video(42, "clip.avi", b"video", "video/x-msvideo")
    assert reference == "s3://private-demo/videos/42/clip.avi"
    assert client.puts[0]["Bucket"] == "private-demo"
    assert client.puts[0]["Key"].startswith("videos/42/")
    assert client.puts[0]["Body"] == b"video"

    with object_storage.materialize_video(reference) as path:
        assert Path(path).read_bytes() == b"private video"
        assert Path(path).exists()
    assert not Path(path).exists()
    assert b"".join(object_storage.stream_video(reference)) == b"private stream"

    object_storage.delete_video(reference)
    assert client.deletes == [{"Bucket": "private-demo", "Key": "videos/42/clip.avi"}]


def test_s3_reference_cannot_escape_configured_bucket(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_S3_BUCKET", "private-demo")
    with pytest.raises(ValueError, match="configured private bucket"):
        object_storage._parse_s3_ref("s3://other-bucket/videos/1/clip.mp4")


def test_supabase_s3_client_uses_sigv4_path_style_and_private_server_credentials(monkeypatch):
    captured = {}

    class StubConfig:
        def __init__(self, signature_version, s3):
            self.signature_version = signature_version
            self.s3 = s3

    boto3_module = ModuleType("boto3")
    boto3_module.client = lambda service, **kwargs: captured.update(service=service, **kwargs) or object()
    botocore_module = ModuleType("botocore")
    botocore_config_module = ModuleType("botocore.config")
    botocore_config_module.Config = StubConfig
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config_module)
    monkeypatch.setattr(settings, "VIDEO_STORAGE_BACKEND", "s3")
    monkeypatch.setattr(settings, "SUPABASE_S3_ENDPOINT", "https://project.storage.supabase.co/storage/v1/s3")
    monkeypatch.setattr(settings, "SUPABASE_S3_REGION", "ap-south-1")
    monkeypatch.setattr(settings, "SUPABASE_S3_BUCKET", "private-demo")
    monkeypatch.setattr(settings, "SUPABASE_S3_ACCESS_KEY_ID", "server-key")
    monkeypatch.setattr(settings, "SUPABASE_S3_SECRET_ACCESS_KEY", "server-secret")

    object_storage._s3_client()

    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://project.storage.supabase.co/storage/v1/s3"
    assert captured["region_name"] == "ap-south-1"
    assert captured["aws_access_key_id"] == "server-key"
    assert captured["aws_secret_access_key"] == "server-secret"
    assert captured["config"].signature_version == "s3v4"
    assert captured["config"].s3 == {"addressing_style": "path"}
