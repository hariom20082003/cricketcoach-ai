# Hosted CricketCoach AI setup

This guide describes the existing GitHub Pages, Render, and Supabase setup and the optional Hugging Face ZeroGPU inference service. It does not create accounts, upload video/model files, or select a paid plan. Review provider pricing and limits before making changes.

## Application and data architecture

- Registration, login, logout, and bearer-session validation remain in the FastAPI backend. Passwords are bcrypt-hashed; a signed 12-hour token references an `auth_sessions` row so logout revocation survives backend restarts. Supabase Auth is not used. Email verification, password reset, and data export are not implemented.
- SQLAlchemy stores users, revocable auth sessions, videos and metadata, analysis jobs, shot predictions/scores, pose results, technique observations, model versions, drills, drill completions, and session notes in PostgreSQL. Progress is computed from completed drills and completed analysis jobs; there is no separate progress table. Local SQLite data is not migrated automatically.
- Alembic is the schema source of truth. Revision `b7fd8d64cd78` creates the application schema and enables RLS on its 12 business tables. Revision `f6a1c3d9e742` adds RLS to `alembic_version`, revokes existing `anon`/`authenticated` table and sequence privileges, and removes their default grants for future application objects. The live `cricguru` database was read-only verified at `f6a1c3d9e742`. The Render source must include that migration file so Alembic recognizes the live revision. Production startup does not call `create_all()`; the Render start command runs `alembic upgrade head` before Uvicorn.
- The API uses the existing FastAPI identity and checks ownership on private video, analysis, progress, note, and profile routes. A database composite foreign key also prevents an analysis job from claiming a video owned by another user. User deletion cascades through relational records and the API deletes their private video objects first.
- Uploaded videos and, if rights are confirmed, model files use one private Supabase Storage bucket over its S3-compatible API. The backend stores `s3://` object references, never public URLs. It streams videos through owner-checked endpoints and downloads working copies to temporary storage. S3 access keys are server-only and bypass Storage RLS, so keep them only in Render environment variables.
- Render's local filesystem is ephemeral and is not used as durable storage. PostgreSQL and the private bucket are the durable stores. Analysis runs in an in-process background thread; a restart can stop an in-progress job, which startup marks failed. Completed history remains in PostgreSQL. This is a small demonstration architecture, not a durable job queue.

The Privacy Notice describes collected data, intended storage, and deletion. Users can delete individual videos/results from History or delete the account from Profile.

## Repository and model artifacts

The root `.gitignore` excludes environment files, local databases, uploads, all dataset media/archive/cache content, the full `dataset/prepared/` metadata directory, virtual environments, build/cache output, and PyTorch checkpoints. The model loader fetches `models/cricketshot_temporal.pt` and `models/visual_model_meta.json` from the configured private bucket prefix when local files are absent. It loads the weights with `map_location="cpu"` and strict state-dict matching. Production defaults to requiring the classifier. `ALLOW_MODEL_UNAVAILABLE_PREVIEW=true` is an explicit non-ML preview override: authentication and other non-ML routes can start, `/api/model/status` reports the model unavailable, and `POST /api/analysis/start/{video_id}` returns HTTP 503 before creating a job. The optional Hugging Face integration instead selects a remote classifier when `HF_ZERO_GPU_SPACE_ID` is configured; Render does not load the weights in that mode. The model status endpoint reports that the Space is configured, not that its queue, quota, storage access, or model has been live-probed. Keep the preview override false only after model artifacts and inference capacity are configured.

The model artifacts still need to be uploaded to private storage. The user reports written rights clearance, but the repository does not contain or independently verify that permission; ensure it covers both source-footage use and derived-weight hosting. The [Hugging Face dataset card](https://huggingface.co/datasets/rokmr/cricket-shot/tree/main) lists Apache-2.0, while the upstream [CricShot10 README](https://github.com/ascuet/CricShot10/blob/main/README.md) limits use of the source footage to research purposes and notes the original rights holders. Do not upload the footage or the 2.89 GB archive. The local model is version `1.0.0`, architecture `efficientnet_b0_frozen_gru` v1. After confirming that the clearance permits hosting, verify the local SHA-256 values and upload exactly these private bucket object keys: `models/cricketshot_temporal.pt` and `models/visual_model_meta.json`. The checkpoint SHA-256 is `da8466e42ad09e036ab739592135c88062e284b8c958c53bf746e9e19eed4d20`; the metadata SHA-256 is `bd6cd349b60042c4938e73cb306bd94f4334c03de5eb4b674bdc328107a8a369`.

## 1. Create the Supabase project and private bucket

1. Use the existing Supabase organization `HAS` and project `cricguru`. Do not create another project or select paid compute, IPv4, or other add-ons. Review current plan limits in the dashboard.
2. In **Database → Connect**, choose **Session pooler** and copy its PostgreSQL URI exactly. Render does not support IPv6, and Supabase Free direct database connections are IPv6-only, so the direct endpoint will not work from Render without a paid IPv4 add-on. The shared Session pooler is available over IPv4 on Free at port `5432`; it supports the long-lived sessions used by this Render API. Do not purchase an IPv4 add-on. Supabase documents connection modes, ports, IPv4 behavior, and SSL options in its [Postgres connection guide](https://supabase.com/docs/guides/database/connecting-to-postgres), and Render documents its IPv6 limitation in its [deployment troubleshooting guide](https://render.com/docs/troubleshooting-deploys).
3. Keep the URI server-side. The application normalizes `postgresql://` to psycopg 3 and adds `sslmode=require` in production when no SSL mode is specified. It rejects explicit insecure SSL modes. Use `sslmode=verify-full` with Supabase's root certificate if you configure certificate verification; `require` encrypts the connection but does not verify the server certificate.
4. Use the existing `cricketcoach-private` bucket. Verify **Public bucket** is disabled; do not create a duplicate bucket, public read policies, or public object URLs. Set the bucket's file-size limit to at least the configured `MAX_VIDEO_SIZE_MB` (default 100 MiB) and permit the supported video MIME types.
5. In **Storage → S3 Connection**, copy the exact endpoint, region, access key ID, and secret access key. The endpoint format is `https://<project_ref>.storage.supabase.co/storage/v1/s3`; configure the exact region displayed by Supabase. The app's boto3 client uses SigV4 and path-style addressing, matching Supabase's [S3 authentication documentation](https://supabase.com/docs/guides/storage/s3/authentication).
6. S3 access keys have broad bucket access and bypass Storage RLS. Do not put them in the frontend, repository, GitHub variables, or local files that can be committed. Store them only as Render backend environment variables.
7. Do not upload video or model files during setup. The API creates video objects under `videos/<user_id>/...` when a user uploads. After written rights clearance for the derived model, upload only the two verified artifacts to `models/cricketshot_temporal.pt` and `models/visual_model_meta.json`; keep the bucket private. Never upload the dataset archive or source cricket footage.

### Database API access and RLS

The app uses FastAPI, not Supabase's Data API, for user access. The applied `f6a1c3d9e742` migration enables RLS on the 12 business tables and Alembic bookkeeping table, revokes direct `anon`/`authenticated` table and sequence grants, and sets default privileges for objects created by `postgres`. It creates no Supabase Auth policies. Keep direct client access revoked; do not add generic `auth.uid()` policies because app users are authenticated by FastAPI, not Supabase Auth. If Data API access is deliberately added later, design explicit grants and ownership policies first. See the [Data API security guide](https://supabase.com/docs/guides/api/securing-your-api).

## 2. Configure Render

Use the existing GitHub repository and root `render.yaml` Blueprint after the reviewed migration and preview-mode changes are committed and pushed. The Blueprint explicitly selects Render's Free plan; this does not create a service. It installs the CPU PyTorch requirements, binds Uvicorn to `0.0.0.0:$PORT`, applies migrations, uses one worker, and checks `/api/health`. Render documents plan IDs in the [Blueprint reference](https://render.com/docs/blueprint-spec) and the [Free plan limits](https://render.com/docs/free). Its start command is:

```bash
alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

The Render service root directory is `backend`, so `alembic.ini` and `migrations/` resolve relative to that directory. With migration `f6a1c3d9e742` committed and the live database already at that revision, the startup migration command should be a no-op. Keep a single service instance for this demo so two processes do not try to apply migrations simultaneously.

Set these backend environment variables in Render. `render.yaml` generates `SECRET_KEY` and prompts for the values marked `sync: false`.

| Variable | Value |
|---|---|
| `APP_ENV` | `production` |
| `SECRET_KEY` | Render-generated unique value (at least 32 UTF-8 bytes). |
| `DATABASE_URL` | Supabase PostgreSQL URI from the Connect panel; never a frontend variable. |
| `DB_POOL_SIZE` | `2` |
| `DB_MAX_OVERFLOW` | `0` |
| `DB_POOL_RECYCLE_SECONDS` | `1800` |
| `CORS_ORIGINS` | `https://hariom20082003.github.io` (the Pages origin, without `/cricketcoach-ai`). |
| `VIDEO_STORAGE_BACKEND` | `s3` |
| `SUPABASE_S3_ENDPOINT` | Exact S3 endpoint ending `/storage/v1/s3`. |
| `SUPABASE_S3_REGION` | Exact Supabase S3 region. |
| `SUPABASE_S3_BUCKET` | `cricketcoach-private` (or your chosen private bucket name). |
| `SUPABASE_S3_ACCESS_KEY_ID` | Server-side Supabase S3 access key ID. |
| `SUPABASE_S3_SECRET_ACCESS_KEY` | Server-side Supabase S3 secret. |
| `MODEL_ARTIFACT_PREFIX` | `models` |
| `VISUAL_MODEL_DIR` | `/tmp/cricketcoach-model` |
| `CRICKETCOACH_DEVICE` | `cpu` |
| `ALLOW_MODEL_UNAVAILABLE_PREVIEW` | `true` in this demo Blueprint for the explicitly limited non-ML preview; set `false` only after approved model artifacts are available and inference capacity is provisioned. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `720` |
| `MAX_VIDEO_SIZE_MB` | `100` |

`DATABASE_URL` passwords containing reserved URI characters must be percent-encoded. Production refuses SQLite, local upload storage, insecure PostgreSQL SSL modes, a non-HTTPS S3 endpoint, wildcard/non-HTTPS CORS, missing object-storage settings, or unsupported device values. SQLAlchemy uses a small application pool with `pool_pre_ping`; psycopg automatic prepared statements are disabled for transaction-pooler compatibility. Use the same database URI for the Render migration and application process.

After deployment, verify `https://<render-service>/api/health` returns HTTP 200 and reports database and model status accurately. With preview mode enabled and remote inference unset, `model_available` is false and analysis-start requests return 503. A local CPU measurement that imports the API, loads the model, and runs a synthetic forward pass peaked at about 577 MiB RSS; it exceeded Render Free's 512 MB RAM limit before real video decoding. The optional Hugging Face ZeroGPU integration keeps model weights off Render, but free inference is queue-based and quota-limited (currently 5 GPU minutes/day for eligible free accounts). It is not a guaranteed production service. Do not choose a paid plan without approval.

## 3. Configure the optional Hugging Face ZeroGPU inference service

This path is optional and does not upload the model weights to GitHub or Hugging Face. It keeps the model in the existing private Supabase bucket, and the private Gradio Space downloads it when its GPU prediction function first runs. Hugging Face ZeroGPU is shared, queue-based, and subject to account eligibility, cold starts, outages, and daily quota. A free account currently receives 5 GPU minutes per day; confirm current terms in Hugging Face before relying on it.

1. Create a **private Gradio Space** in your Hugging Face account and select **ZeroGPU** hardware. The account must be eligible for ZeroGPU. Do not make the Space public: private video uploads pass through it during analysis.
2. In that Space's **Settings → Variables and secrets**, add `VIDEO_STORAGE_BACKEND=s3`, `SUPABASE_S3_ENDPOINT`, `SUPABASE_S3_REGION`, `SUPABASE_S3_BUCKET`, `SUPABASE_S3_ACCESS_KEY_ID`, `SUPABASE_S3_SECRET_ACCESS_KEY`, `MODEL_ARTIFACT_PREFIX=models`, `VISUAL_MODEL_DIR=/tmp/cricketcoach-model`, and `CRICKETCOACH_DEVICE=cuda`. Reuse the existing private-bucket connection values, but create a separate S3 credential if the provider supports a restricted key. Never put these values in GitHub or chat.
3. After confirming the rights clearance, upload only `cricketshot_temporal.pt` and `visual_model_meta.json` to the private bucket at `models/cricketshot_temporal.pt` and `models/visual_model_meta.json`. Verify both checksums shown above. Do not upload source videos or weights to the Space repository.
4. In Hugging Face **Settings → Access Tokens**, create a read-only token that can read this private Space. Add `HF_ZERO_GPU_SPACE_ID` to Render as `owner/space-name` and `HF_TOKEN` as that read token. Do not share either token in chat. These settings are opt-in; leaving the Space ID empty preserves the existing limited-preview behavior.
5. In GitHub **Settings → Secrets and variables → Actions**, add the repository variable `HF_SPACE_ID` (`owner/space-name`) and secret `HF_SPACE_DEPLOY_TOKEN` (a separate Hugging Face token with write access to the Space). Then run **Actions → Deploy CricketCoach model to Hugging Face ZeroGPU → Run workflow**. The workflow uploads only Space code and metadata, never the model checkpoint.
6. Wait for the Space to build and become healthy, then redeploy the Render service after setting its two Hugging Face variables. Check `/api/model/status`, then upload a non-sensitive test clip while observing both providers' logs. The first request may be slow or fail if the Space has not loaded the private weights, is queued, or has exhausted its quota.

When enabled, the Render API sends each analyzed video to the private Space for inference. This transfer is disclosed in the frontend Privacy Notice. Hugging Face provider-side processing/log retention may apply; do not send sensitive footage.

## 4. Configure GitHub Pages

1. Use the existing repository `hariom20082003/cricketcoach-ai`.
2. In **Settings → Pages**, set the source to **GitHub Actions**.
3. In **Settings → Secrets and variables → Actions → Variables**, add `VITE_API_URL` with the Render HTTPS origin, e.g. `https://cricketcoach-api.onrender.com`, without `/api` or a trailing slash. This is a public API address, not a secret.
4. The existing `.github/workflows/deploy-pages.yml` derives the repository base path and runs the Vite production build. Push to `main` or manually run **Actions → Build and deploy CricketCoach AI frontend → Run workflow** after the repository is ready.
5. Copy the final Pages site origin into Render's `CORS_ORIGINS` and redeploy the backend. Do not include `/REPOSITORY/` in the CORS origin.

## 5. Apply and check migrations

From the repository root, initialize a fresh local development database with the committed migration:

```bash
cd backend
DATABASE_URL=sqlite:///./migration-check.sqlite3 ../.venv/bin/python -m alembic upgrade head
DATABASE_URL=sqlite:///./migration-check.sqlite3 ../.venv/bin/python -m alembic check
```

On Render the same operation runs automatically before Uvicorn. To apply it manually against a configured Supabase database before deployment, set `DATABASE_URL` in the backend shell and run:

```bash
cd backend
alembic upgrade head
alembic current
alembic check
```

The initial migration creates all tables from an empty database. Local SQLite data is not copied. Keep future schema changes in new Alembic revisions and deploy migrations before serving code that depends on them. Do not use `create_all()` to initialize a hosted schema.

## 6. End-to-end deployment checks

After configuring services and permitting the model artifact, test manually:

1. Confirm `/api/health` returns HTTP 200, `database: "ok"`, and `model_available: true`.
2. Register a test account, sign out, sign in, upload a small supported video, wait for analysis, refresh History, and confirm the saved result remains.
3. Create a second test account. Confirm it cannot view, stream, or delete the first account's video or analysis. Delete the video and then the account to confirm cleanup.
4. Restart/redeploy Render. Confirm login and completed history remain, and the private video streams through the API. An in-progress job may be marked failed after restart.
5. Check browser CORS/network errors and Render logs. Local checks do not count as hosted verification.

The last recorded deployment check found the GitHub Pages site and Render API live, with the database healthy and the model unavailable. The optional Hugging Face Space, model artifacts in Supabase Storage, remote tokens, and an end-to-end inference request have not been configured or verified.

## Local checks

```bash
cd backend
../.venv/bin/python -m pytest tests -q
DATABASE_URL=sqlite:///./migration-check.sqlite3 ../.venv/bin/python -m alembic upgrade head
DATABASE_URL=sqlite:///./migration-check.sqlite3 ../.venv/bin/python -m alembic check
CRICKETCOACH_DEVICE=cpu ../.venv/bin/python -c 'from app.ml.visual_classifier import VisualShotClassifier; c=VisualShotClassifier(); print(c.is_available, c.device_name, c.load_error)'
cd ../frontend
npm ci
npm run build
```

For the backend's private object storage, only local mocked tests have been run; no Supabase credentials or bucket are configured here.
