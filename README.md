# Multimodal Facial Behavior Analytics

A web application that analyzes a video frame by frame. It predicts facial
expressions, estimates head pose, summarizes the results over time, and lets
the user ask an AI assistant questions about the report.

- **Backend:** FastAPI serving a TensorFlow / Keras facial-expression classifier
  (MobileNetV2, 7 classes) and MediaPipe face landmarks and head pose.
- **Frontend:** React + Vite dashboard with charts (Recharts).
- **AI assistant:** Google Gemini, answering questions grounded in the analysis report.

> **Scope:** The expression metrics below come from a static test split of
> FER2013 images. They do not measure accuracy on videos or real emotional
> states. The engagement score in `src/engagement.py` is an experimental
> heuristic and is not used by the application.

## Results

Evaluated on the FER2013 balanced test split (6,286 images). Source:
`outputs/verified_evaluation_summary.json`.

| Metric | Value |
|---|---|
| Accuracy | 70.78% |
| Macro F1 | 0.703 |
| Best class (F1) | disgust, 0.953 |
| Hardest class (F1) | sad, 0.512 |

## Project structure

```
backend/          FastAPI app, analytics, AI assistant
src/              Model wrapper, MediaPipe features, video sampling, pipeline
frontend/         React + Vite dashboard
models/           Saved classifier and MediaPipe face-landmarker model
outputs/          Generated analysis output (verified evaluation summary is kept)
scripts/          Utility scripts
tests/            pytest suite (unit, pipeline, and API tests)
```

## Requirements

- **Python 3.11** (TensorFlow 2.20 and the pinned packages are tested on 3.11)
- **Node.js 18 or newer** and npm
- A **Gemini API key** for the AI assistant (the analysis works without it)

## Setup

From the project root:

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements-dev.txt
```

Create your settings file from the example:

```bash
# Windows (Command Prompt)
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` in a text editor and replace `paste-your-key-here` with your key from
https://aistudio.google.com/apikey. Save the file. The backend loads `.env` automatically
when it starts. Keep this file private: it is listed in `.gitignore` and must never be committed.

Values set in your terminal take priority over `.env`. If the assistant ignores your
`.env` values, check for an old `set GEMINI_API_KEY=...` line in that terminal window.

## Run

Terminal 1, backend (from the project root, with the virtual environment active):

```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2, frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The API documentation is at http://127.0.0.1:8000/docs.
The Vite dev server proxies `/api` requests to the backend on port 8000.

## Tests

```bash
pytest
```

The suite has 20 tests. The API tests and the pipeline tests load the real model,
so the first run takes a little longer.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service status and whether the model is loaded |
| GET | `/api/evaluation` | Verified evaluation metrics |
| POST | `/api/analyze` | Upload a video (`file`) with an optional sampling `interval` (0.25 to 10 s) |
| POST | `/api/chat` | Ask the AI assistant a question about a report |

Accepted video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.mpeg`, `.mpg`.
The default upload limit is 500 MB (set `MAX_UPLOAD_SIZE_MB` to change it).

## Configuration

| Variable | Required | Default |
|---|---|---|
| `GEMINI_API_KEY` | For the AI assistant | none |
| `GEMINI_MODEL` | No | `gemini-3.7-flash` |
| `GEMINI_FALLBACK_MODEL` | No | `gemini-2.5-flash-lite` |
| `MAX_UPLOAD_SIZE_MB` | No | `500` |

Put these values in `.env` in the project root (copy `.env.example`). The backend loads it on startup.

### Choosing a model

Google retires model names over time. If the assistant returns a 404 error about a model,
list the models your key can use (run this with the virtual environment active):

```bash
python -c "from google import genai; import os; from dotenv import load_dotenv; load_dotenv(); c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); [print(m.name) for m in c.models.list() if 'flash' in m.name]"
```

Copy a text model name (without the `models/` prefix) into `GEMINI_MODEL` in `.env`,
then restart the backend. A 503 "high demand" error is temporary, so try again later.

## Known limitations

- The MediaPipe and TensorFlow versions are pinned because the saved model
  requires Keras 3.13 and the MediaPipe Tasks API. Upgrading them requires
  retesting.
- The engagement heuristic (`src/engagement.py`) is experimental and is not validated.
- Head pose is an approximation from MediaPipe's facial transformation matrix.
- Expression predictions on real video are not benchmarked. Treat them as indicative.

## Security

Never commit API keys. `.gitignore` excludes `.env` and `.streamlit/secrets.toml`.
If a key was ever shared or committed, revoke it in the Google AI console and create a new one.
