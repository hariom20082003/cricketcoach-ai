# CricketCoach AI 🏏

An AI-powered cricket shot classification and batting technique improvement platform built as an engineering college ML project.

---

## Features

| Feature | Status |
|---|---|
| User registration & login (JWT) | ✅ |
| Video upload (drag-drop) | ✅ |
| Webcam recording | ✅ |
| MediaPipe pose extraction | ✅ |
| Joint angle measurement | ✅ |
| Technique observations (labelled by source) | ✅ |
| Practice drills library (7 seeded) | ✅ |
| Drill completion tracking with ratings | ✅ |
| Shot guide (7 shots) | ✅ |
| Progress analytics with charts | ✅ |
| Shot classification (EfficientNet-B0 + bidirectional GRU) | ✅ Trained 10-class model |
| Per-user video authorization | ✅ |
| Background analysis worker | ✅ |
| Model status page with training instructions | ✅ |

The visual classifier reads sampled RGB video frames. Pose measurements remain a separate input to technique coaching.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, TypeScript, Tailwind CSS, React Router v6, Recharts, Zustand |
| API | Python, FastAPI 0.104, Pydantic v2, SQLAlchemy 2, SQLite locally / PostgreSQL when hosted |
| Video/Pose | OpenCV 4.8, MediaPipe 0.10 |
| ML | PyTorch 2.8, TorchVision 0.23, EfficientNet-B0, bidirectional GRU |
| Auth | Backend JWT sessions (python-jose), bcrypt (passlib); revocable sessions stored in the configured database |

---

## Requirements

- Python 3.9+
- Node.js 18+
- macOS / Linux (Windows via WSL)

---

## Installation

### 1. Clone / enter the project

```bash
cd "cricket guru"
```

### 2. Backend

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

Create the environment file:

```bash
cp backend/.env.example backend/.env
# Edit .env if needed (SECRET_KEY etc.)
```

Initialize the local database from the committed Alembic migration:

```bash
cd backend
../.venv/bin/python -m alembic upgrade head
cd ..
```

### 3. Frontend

```bash
cd ../frontend
npm install
```

---

## Running the App

### Start the backend (from the repository root)

```bash
cd backend
../.venv/bin/python -m alembic upgrade head
../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

The backend starts at **http://localhost:8000**. The migration command creates the database schema; the backend seeds 7 practice drills at startup.

Local SQLite and upload files are ignored by Git. Hosted operation uses PostgreSQL and private object storage; see [docs/HOSTED_DEMO.md](docs/HOSTED_DEMO.md).

### Start the frontend dev server (from `frontend/`)

```bash
npm run dev
```

Open **http://localhost:5173** in your browser.

The Vite proxy forwards `/api` requests to `localhost:8000`.

### Production build

```bash
cd frontend && npm run build
# Serve dist/ with any static file server or let FastAPI serve it
```

---

## Shot Model and Dataset

The model report describes the training data and held-out evaluation. Raw footage, archives, and generated dataset manifests are excluded from this repository; they must be acquired and prepared locally. Dataset licensing for the underlying footage and derived weights is unresolved. Review [docs/VISUAL_MODEL.md](docs/VISUAL_MODEL.md) before using or distributing dataset-derived artifacts. The trained weights are not committed; the hosted backend loads permitted weights from private storage as described in [docs/HOSTED_DEMO.md](docs/HOSTED_DEMO.md).

To rebuild the manifest from extracted videos and check split readiness:

```bash
.venv/bin/python ml_pipeline/prepare_dataset.py --input-dir dataset/external/cricket-shot --output-dir dataset/prepared/cricket-shot
.venv/bin/python ml_pipeline/train_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --check-only
```

To train and evaluate the video model:

```bash
.venv/bin/python ml_pipeline/train_visual_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --device auto
```

The checkpoint is generated locally at `backend/app/ml/models/cricketshot_temporal.pt` and is ignored by Git; only aggregate evaluation metadata is included at `backend/app/ml/models/visual_model_meta.json`. See [docs/VISUAL_MODEL.md](docs/VISUAL_MODEL.md) for split counts, held-out results, reproduction commands, and dataset limitations. The dataset card license and source-footage rights are distinct; review those terms before redistribution or commercial use.

## Running Tests

### Backend tests (from `backend/`)

```bash
../.venv/bin/python -m pytest tests/ -v
```

The backend suite covers API auth, video validation, analysis and model status, plus ML and dataset-preparation behavior. Run it in the project virtual environment so the trained visual model dependencies are available.

## Hosted college demo

The deployment configuration and step-by-step GitHub Pages, Render, and Supabase setup are in [docs/HOSTED_DEMO.md](docs/HOSTED_DEMO.md). The local app uses SQLite/files by default; durable hosted operation requires a PostgreSQL database and private S3-compatible object storage. No hosted service is configured by this repository alone.

### ML unit tests only

```bash
../.venv/bin/python -m pytest tests/test_ml.py -v
```

---

## Project Structure

```
cricket guru/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── database.py          # SQLAlchemy engine + session
│   │   ├── models/              # ORM models (User, Video, AnalysisJob, …)
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── routers/             # API routes (auth, videos, analysis, misc)
│   │   ├── services/            # Auth utilities, drill seeding
│   │   ├── ml/
│   │   │   ├── pose_extractor.py    # MediaPipe extraction + feature vector
│   │   │   ├── visual_classifier.py # EfficientNet-B0 + temporal GRU inference
│   │   │   ├── technique_coach.py   # Observation generation
│   │   │   └── models/              # Trained model files go here
│   │   └── workers/
│   │       └── analysis_worker.py   # Background analysis thread
│   ├── tests/
│   │   ├── test_api.py          # Full API test suite
│   │   └── test_ml.py           # ML unit tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Router with protected routes
│   │   ├── pages/               # 15 page components
│   │   ├── components/layout/   # Sidebar layout
│   │   ├── services/            # Axios API layer
│   │   ├── store/               # Zustand auth store
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── ml_pipeline/
│   ├── train_model.py           # Dataset readiness and legacy pipeline
│   ├── train_visual_model.py    # Visual-temporal training and evaluation
│   └── prepare_dataset.py       # Dataset preparation helper
└── README.md
```

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/health` | No | Health check |
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Login, get JWT |
| GET | `/api/auth/me` | Yes | Current user |
| PUT | `/api/auth/me` | Yes | Update profile |
| POST | `/api/videos/upload` | Yes | Upload video |
| GET | `/api/videos/` | Yes | List my videos |
| GET | `/api/videos/{id}` | Yes | Get video info |
| GET | `/api/videos/{id}/stream` | Yes | Stream video |
| DELETE | `/api/videos/{id}` | Yes | Delete video |
| POST | `/api/analysis/start/{video_id}` | Yes | Start analysis job |
| GET | `/api/analysis/{job_id}/status` | Yes | Poll job status |
| GET | `/api/analysis/{job_id}/results` | Yes | Get full results |
| GET | `/api/analysis/history/list` | Yes | My analysis history |
| GET | `/api/drills/` | Yes | List drills |
| GET | `/api/drills/{id}` | Yes | Get drill |
| POST | `/api/drills/{id}/complete` | Yes | Mark drill done |
| GET | `/api/drills/completed/list` | Yes | My completions |
| GET | `/api/progress/summary` | Yes | Progress analytics |
| GET | `/api/model/status` | No | ML model status |
| GET | `/api/shots/guide` | No | All shot guides |
| GET | `/api/shots/guide/{id}` | No | Single shot guide |
| GET | `/api/notes/` | Yes | Session notes |
| POST | `/api/notes/` | Yes | Create note |

---

## Transparency

- **Pose estimation** measures body landmarks — it cannot confirm bat face, ball contact, or timing.
- **Technique observations** are labelled by source: `measured_feature`, `heuristic`, or `validated_reference`.
- **Shot classification** uses the trained visual-temporal model; its held-out scores are dataset-specific and uncalibrated.
- **Reliability** is reported as `high`, `medium`, or `low` for each observation.
- **Class scores** are the model's raw softmax outputs and are uncalibrated; do not interpret them as calibrated probabilities.

---

## Known Limitations

- Background worker uses Python threads (not Celery). Works fine for course use; not production-scaled.
- No real-time video analysis — only uploaded/recorded videos.
- The available test set is small, and source match IDs are unavailable, so cross-split match leakage cannot be ruled out.
- Pose estimation works best with side-on, well-lit, full-body visible recordings.
