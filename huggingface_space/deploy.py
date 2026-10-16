import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    space_id = os.environ.get("HF_SPACE_ID", "").strip()
    token = os.environ.get("HF_SPACE_DEPLOY_TOKEN", "").strip()
    if not space_id or not token:
        raise RuntimeError(
            "Set the HF_SPACE_ID Actions variable and HF_SPACE_DEPLOY_TOKEN secret."
        )

    api = HfApi(token=token)
    info = api.repo_info(repo_id=space_id, repo_type="space")
    if not getattr(info, "private", False):
        raise RuntimeError("The Hugging Face inference Space must be private.")

    with tempfile.TemporaryDirectory(prefix="cricketcoach-hf-space-") as temporary_dir:
        staging = Path(temporary_dir)
        shutil.copy2(ROOT / "huggingface_space/README.md", staging / "README.md")
        shutil.copy2(ROOT / "huggingface_space/requirements.txt", staging / "requirements.txt")
        shutil.copy2(ROOT / "huggingface_space/app.py", staging / "app.py")

        app_root = staging / "backend/app"
        for directory in ("ml", "ml/models", "services"):
            (app_root / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "backend/app/config.py", app_root / "config.py")
        shutil.copy2(
            ROOT / "backend/app/ml/visual_classifier.py",
            app_root / "ml/visual_classifier.py",
        )
        shutil.copy2(
            ROOT / "backend/app/ml/models/visual_model_meta.json",
            app_root / "ml/models/visual_model_meta.json",
        )
        shutil.copy2(
            ROOT / "backend/app/services/object_storage.py",
            app_root / "services/object_storage.py",
        )

        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy CricketCoach ZeroGPU inference service",
        )

    print(
        "Uploaded private Space application code. Model weights are fetched at runtime "
        "from the private Supabase bucket; no model binaries were uploaded to Hugging Face."
    )


if __name__ == "__main__":
    main()
