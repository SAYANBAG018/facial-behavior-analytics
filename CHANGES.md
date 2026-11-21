# Changes from the original upload

## Security
- Removed `.streamlit/secrets.toml`, which contained a live Gemini API key.
  **The key must be revoked and replaced** (see README, Security).
- The Gemini key is now read only from environment variables.
  The Streamlit secrets fallback was removed.

## Structure
- Removed the Streamlit app (`app.py`) and its module (`src/ai_assistant.py`).
  The React + FastAPI app is the single supported stack.
- Removed build and generated files: `frontend/node_modules/`, all `__pycache__/`,
  a personal video file, `test_video_browser.mp4`, the empty `test` file,
  and stale Streamlit-era outputs (`analysis_results.csv`, `test_results.csv`).
- Tests now generate a synthetic video, so no personal footage is stored.
  Removed the sample outputs (they came from personal footage). `check_model_metadata.py`
  moved to `scripts/`.
- Added `src/__init__.py` and `.python-version` (3.11).
- Normalized all text files to LF line endings.

## Code fixes
- `src/features.py`: removed the deprecated `mp.solutions.face_mesh` call, which
  current MediaPipe no longer provides (the app crashed on startup). Landmarks and
  head pose now come from one MediaPipe Tasks detection per frame, which also
  halves model work per frame. **Note:** landmark values can differ slightly from
  the old FaceMesh output, so sample outputs may not match exactly.
- `backend/assistant.py`: environment-only configuration; removed the TOML fallback.
- `scripts/check_model_metadata.py`: uses paths relative to the project, so it works
  from any directory.

## Dependencies
- Model fix: the saved model was written with Keras 3.13.2. The old pin
  (TensorFlow 2.15.1, which ships Keras 2) could not load it. Now pinned to
  TensorFlow 2.20.0 and Keras 3.13.2.
- Pinned every runtime dependency. Removed unused packages (`streamlit`, `scipy`,
  `scikit-learn`, `Pillow`).
- Added `requirements-dev.txt` (pytest, httpx) and `pytest.ini`.

## Tests
- Converted the print-based scripts to a pytest suite (`tests/`, 14 tests).
- Failures now fail the run. The old `test_model.py` caught every exception and exited 0.
- Added API tests for health, evaluation, upload validation, chat validation,
  and an end-to-end video analysis. Generated output goes to a temporary folder.

## .env support
- Added `python-dotenv`. The backend loads `<project>/.env` on startup.
  Variables set in the terminal take priority over the file.
- Added `tests/test_config.py` (4 tests) for configuration loading and the `.env` behavior.
- Default model names changed to `gemini-3.7-flash` and `gemini-2.5-flash-lite`.
  `gemini-2.5-flash` returns 404 for new keys.

## Assistant speed
- Gemini requests now have a 45-second timeout (`REQUEST_TIMEOUT_MS`). Previously a stalled
  request could wait for many minutes.
- The Google library's hidden retries are disabled. Retries are controlled only by
  `backend/assistant.py`, with 2 attempts per model.
- Added `tests/test_assistant_timeout.py`.

## Documentation
- Wrote a full README (setup, run, tests, API, configuration, limitations).
- Added `.gitignore`, `.env.example`, and this CHANGES.md.

## Still to do (recommended)
- Consider code-splitting the frontend bundle (build warns it is over 500 kB).
