# Hosted college demonstration setup

This guide prepares the existing app for GitHub Pages, Render, and Supabase. It does not create accounts, provision services, publish the site, upload video/model files, or select a paid plan. Review provider pricing and limits before creating anything.

## Application and data architecture

- Registration, login, logout, and bearer-session validation remain in the FastAPI backend. Passwords are bcrypt-hashed; a signed 12-hour token references an `auth_sessions` row so logout revocation survives backend restarts. Supabase Auth is not used. Email verification, password reset, and data export are not implemented.
- SQLAlchemy stores users, revocable auth sessions, videos and metadata, analysis jobs, shot predictions/scores, pose results, technique observations, model versions, drills, drill completions, and session notes in PostgreSQL. Progress is computed from completed drills and completed analysis jobs; there is no separate progress table. Local SQLite data is not migrated automatically.
- Alembic is the schema source of truth. The initial revision is `backend/migrations/versions/b7fd8d64cd78_initial_application_schema.py`. It creates every application table, foreign key, index, check/unique constraint, enum, and enables PostgreSQL RLS on the public tables. No grants or `anon`/`authenticated` policies are added. Production startup does not call `create_all()`; the Render start command runs `alembic upgrade head` before Uvicorn.
- The API uses the existing FastAPI identity and checks ownership on private video, analysis, progress, note, and profile routes. A database composite foreign key also prevents an analysis job from claiming a video owned by another user. User deletion cascades through relational records and the API deletes their private video objects first.
- Uploaded videos and, if rights are confirmed, model files use one private Supabase Storage bucket over its S3-compatible API. The backend stores `s3://` object references, never public URLs. It streams videos through owner-checked endpoints and downloads working copies to temporary storage. S3 access keys are server-only and bypass Storage RLS, so keep them only in Render environment variables.
- Render's local filesystem is ephemeral and is not used as durable storage. PostgreSQL and the private bucket are the durable stores. Analysis runs in an in-process background thread; a restart can stop an in-progress job, which startup marks failed. Completed history remains in PostgreSQL. This is a small demonstration architecture, not a durable job queue.

The Privacy Notice describes collected data, intended storage, and deletion. Users can delete individual videos/results from History or delete the account from Profile.

## Repository and model artifacts

The root `.gitignore` excludes environment files, local databases, uploads, all dataset media/archive/cache content, the full `dataset/prepared/` metadata directory, virtual environments, build/cache output, and PyTorch checkpoints. The model loader fetches `models/cricketshot_temporal.pt` and `models/visual_model_meta.json` from the configured private bucket prefix when local files are absent. It loads the weights with `map_location="cpu"` and strict state-dict matching. Production is configured for CPU inference and refuses to start if the model is unavailable.

The model and original footage have **not** been uploaded. The licensing question remains unresolved: the [Hugging Face dataset card](https://huggingface.co/datasets/rokmr/cricket-shot/tree/main) lists Apache-2.0, while the upstream [CricShot10 README](https://github.com/ascuet/CricShot10/blob/main/README.md) limits use of the source footage to research purposes and notes the original rights holders. Obtain permission for the derived weights before uploading them to private storage or enabling a hosted model deployment. Do not upload the footage or the 2.89 GB archive.

## 1. Create the Supabase project and private bucket

1. Create/select a Supabase project manually and review its current pricing, region, and database/storage limits. No Supabase account or project has been created by this task.
2. In **Database → Connect**, copy the PostgreSQL URI. For a long-running Render web service, prefer the direct connection when its IPv6 route is available, or the shared **Session pooler** when IPv4 is required. Avoid assuming that every Render region/network can reach the direct IPv6 endpoint. Transaction pool mode also works with this code because psycopg prepared statements are disabled, but session mode is the simpler fit for a persistent API service. Supabase documents the connection modes, ports, IPv4 behavior, and SSL options in its [Postgres connection guide](https://supabase.com/docs/guides/database/connecting-to-postgres).
3. Keep the URI server-side. The application normalizes `postgresql://` to psycopg 3 and adds `sslmode=require` in production when no SSL mode is specified. It rejects explicit insecure SSL modes. Use `sslmode=verify-full` with Supabase's root certificate if you configure certificate verification; `require` encrypts the connection but does not verify the server certificate.
4. Create a Storage bucket named `cricketcoach-private` (or another name you will configure). Leave **Public bucket** disabled; do not create public read policies or public object URLs. Set the bucket's file-size limit to at least the configured `MAX_VIDEO_SIZE_MB` (default 100 MiB) and permit the supported video MIME types.
5. In **Storage → S3 Connection**, copy the exact endpoint, region, access key ID, and secret access key. The endpoint format is `https://<project_ref>.storage.supabase.co/storage/v1/s3`; configure the exact region displayed by Supabase. The app's boto3 client uses SigV4 and path-style addressing, matching Supabase's [S3 authentication documentation](https://supabase.com/docs/guides/storage/s3/authentication).
6. S3 access keys have broad bucket access and bypass Storage RLS. Do not put them in the frontend, repository, GitHub variables, or local files that can be committed. Store them only as Render backend environment variables.
7. Do not upload video or model files as part of setup. The video prefix (`videos/<user_id>/...`) is created by the API when a user uploads. Upload the two model files to the `models/` prefix only after the college confirms rights to host the derived weights.

### Database API access and RLS

The app uses FastAPI, not Supabase's Data API, for user access. The Alembic migration enables RLS on all 12 application tables and creates no Data API grants or policies. Keep these tables unavailable to `anon` and `authenticated` roles; do not add generic `auth.uid()` policies because app users are authenticated by FastAPI, not Supabase Auth. If Data API access is deliberately added later, design explicit grants and ownership policies first. Supabase's current defaults for newly created Data API tables have changed, so verify the project's Data API exposure settings rather than relying on a default; see the [Data API security guide](https://supabase.com/docs/guides/api/securing-your-api).

## 2. Configure Render

Create a Render web service from the repository and select the root `render.yaml` Blueprint. It installs the CPU PyTorch requirements, binds Uvicorn to `0.0.0.0:$PORT`, applies migrations, uses one worker, and checks `/api/health`. Its start command is:

```bash
alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

The Render service root directory is `backend`, so `alembic.ini` and `migrations/` resolve relative to that directory. The migration command is safe to repeat after the first successful upgrade. Keep a single service instance for this demo so two processes do not try to apply migrations simultaneously.

Set these backend environment variables in Render. `render.yaml` generates `SECRET_KEY` and prompts for the values marked `sync: false`.

| Variable | Value |
|---|---|
| `APP_ENV` | `production` |
| `SECRET_KEY` | Render-generated unique value (at least 32 UTF-8 bytes). |
| `DATABASE_URL` | Supabase PostgreSQL URI from the Connect panel; never a frontend variable. |
| `DB_POOL_SIZE` | `2` |
| `DB_MAX_OVERFLOW` | `0` |
| `DB_POOL_RECYCLE_SECONDS` | `1800` |
| `CORS_ORIGINS` | Exact HTTPS origin of the Pages site, e.g. `https://OWNER.github.io`, without its repository path. |
| `VIDEO_STORAGE_BACKEND` | `s3` |
| `SUPABASE_S3_ENDPOINT` | Exact S3 endpoint ending `/storage/v1/s3`. |
| `SUPABASE_S3_REGION` | Exact Supabase S3 region. |
| `SUPABASE_S3_BUCKET` | `cricketcoach-private` (or your chosen private bucket name). |
| `SUPABASE_S3_ACCESS_KEY_ID` | Server-side Supabase S3 access key ID. |
| `SUPABASE_S3_SECRET_ACCESS_KEY` | Server-side Supabase S3 secret. |
| `MODEL_ARTIFACT_PREFIX` | `models` |
| `VISUAL_MODEL_DIR` | `/tmp/cricketcoach-model` |
| `CRICKETCOACH_DEVICE` | `cpu` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `720` |
| `MAX_VIDEO_SIZE_MB` | `100` |

`DATABASE_URL` passwords containing reserved URI characters must be percent-encoded. Production refuses SQLite, local upload storage, insecure PostgreSQL SSL modes, a non-HTTPS S3 endpoint, wildcard/non-HTTPS CORS, missing object-storage settings, or unsupported device values. SQLAlchemy uses a small application pool with `pool_pre_ping`; psycopg automatic prepared statements are disabled for transaction-pooler compatibility. Use the same database URI for the Render migration and application process.

After deployment, verify `https://<render-service>/api/health` returns HTTP 200 with `database: "ok"` and `model_available: true`. The model flag stays false until the two permitted model artifacts exist in the private bucket; production startup fails if it cannot load the compatible model. A successful local CPU load does not prove the chosen Render instance has enough memory or startup time.

## 3. Configure GitHub Pages

1. Create an empty GitHub repository manually. This task has not created a repository or pushed any commit.
2. In **Settings → Pages**, set the source to **GitHub Actions**.
3. In **Settings → Secrets and variables → Actions → Variables**, add `VITE_API_URL` with the Render HTTPS origin, e.g. `https://cricketcoach-api.onrender.com`, without `/api` or a trailing slash. This is a public API address, not a secret.
4. The existing `.github/workflows/deploy-pages.yml` derives the repository base path and runs the Vite production build. Push to `main` or manually run **Actions → Build and deploy CricketCoach AI frontend → Run workflow** after the repository is ready.
5. Copy the final Pages site origin into Render's `CORS_ORIGINS` and redeploy the backend. Do not include `/REPOSITORY/` in the CORS origin.

## 4. Apply and check migrations

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

## 5. End-to-end deployment checks

After configuring services and permitting the model artifact, test manually:

1. Confirm `/api/health` returns HTTP 200, `database: "ok"`, and `model_available: true`.
2. Register a test account, sign out, sign in, upload a small supported video, wait for analysis, refresh History, and confirm the saved result remains.
3. Create a second test account. Confirm it cannot view, stream, or delete the first account's video or analysis. Delete the video and then the account to confirm cleanup.
4. Restart/redeploy Render. Confirm login and completed history remain, and the private video streams through the API. An in-progress job may be marked failed after restart.
5. Check browser CORS/network errors and Render logs. Local checks do not count as hosted verification.

No hosted database, bucket, Render service, or Pages deployment has been created or tested by this task.

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
