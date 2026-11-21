
"""
FastAPI wrapper around the existing facial behavior analytics pipeline.

Run from the project root:

    uvicorn backend.main:app --reload --port 8000

Frontend:
    http://localhost:5173

API documentation:
    http://127.0.0.1:8000/docs
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load <project>/.env if it exists. Variables already set in the shell
# take priority over values in the file (override=False).
load_dotenv(ROOT / ".env", override=False)

import src.pipeline as sp  # noqa: E402
from src.fer import FERModel  # noqa: E402
from backend import analytics as an  # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
}

# Default upload limit: 500 MB.
# Override with MAX_UPLOAD_SIZE_MB in the environment if required.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

OUTPUT_DIR = ROOT / "outputs"


# ---------------------------------------------------------------------
# Initialize the FER model once
# ---------------------------------------------------------------------

# Avoid loading the classifier separately for every analysis request.
_fer = FERModel()

# Reuse the already-loaded model inside the existing pipeline.
sp.FERModel = lambda: _fer


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Facial Behavior Analytics API",
    description=(
        "API for video-based facial expression predictions, "
        "head-pose analytics, temporal summaries, and AI-assisted reporting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe records.

    Pandas converts NaN and infinity values to JSON null through
    to_json(), preventing invalid JSON values from reaching React.
    """

    if df is None or df.empty:
        return []

    try:
        return json.loads(
            df.to_json(
                orient="records",
                date_format="iso",
            )
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not convert analysis results into JSON.",
        ) from exc


def validate_video_filename(filename: str | None) -> str:
    """
    Validate the uploaded file extension.

    Returns the validated suffix.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file with a valid filename.",
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))

        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format. Allowed formats: {allowed}",
        )

    return suffix


async def save_upload_to_temp_file(
    file: UploadFile,
    suffix: str,
) -> Path:
    """
    Save an uploaded video to a temporary file in chunks.

    Chunked reading avoids loading the entire video into memory at once.
    The temporary file is removed after analysis.
    """

    temp_path = None
    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:

            temp_path = Path(tmp.name)

            while True:
                chunk = await file.read(chunk_size)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Video exceeds the maximum upload size "
                            f"of {MAX_UPLOAD_SIZE_MB} MB."
                        ),
                    )

                tmp.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        return temp_path

    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def save_analysis_outputs(
    df: pd.DataFrame,
    report: dict,
) -> None:
    """
    Save the latest analysis results.

    Existing output filenames are retained for compatibility with
    the current project.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "facial_behavior_results.csv"
    json_path = OUTPUT_DIR / "facial_behavior_report.json"

    temp_csv = OUTPUT_DIR / "facial_behavior_results.tmp.csv"
    temp_json = OUTPUT_DIR / "facial_behavior_report.tmp.json"

    try:
        df.to_csv(temp_csv, index=False)

        temp_json.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # Replace the previous output only after both temporary
        # files have been written successfully.
        temp_csv.replace(csv_path)
        temp_json.replace(json_path)

    finally:
        temp_csv.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Facial Behavior Analytics API",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Facial Behavior Analytics API",
        "fer_model_loaded": _fer is not None,
    }


# ---------------------------------------------------------------------
# Verified evaluation metrics
# ---------------------------------------------------------------------

@app.get("/api/evaluation")
def evaluation():
    """
    Return verified evaluation metrics, if available.

    These metrics are separate from the predictions generated
    for an individual uploaded video.
    """

    try:
        return an.load_verified_evaluation_metrics()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load evaluation metrics.",
        ) from exc


# ---------------------------------------------------------------------
# Video analysis endpoint
# ---------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    interval: float = Form(1.0),
):
    """
    Analyze an uploaded video.

    Input:
        file: Video file.
        interval: Sampling interval in seconds.

    Output:
        report
        frames
        pose_series
        pose_summary
        movement_summary
        emotion_counts
        emotion_runs
    """

    # Validate sampling interval.
    if not 0.25 <= interval <= 10:
        raise HTTPException(
            status_code=400,
            detail="Sampling interval must be between 0.25 and 10 seconds.",
        )

    # Validate file extension.
    suffix = validate_video_filename(file.filename)

    temp_video_path = None

    try:
        # -------------------------------------------------------------
        # 1. Save uploaded video
        # -------------------------------------------------------------

        temp_video_path = await save_upload_to_temp_file(
            file=file,
            suffix=suffix,
        )

        # -------------------------------------------------------------
        # 2. Run the existing facial behavior pipeline
        # -------------------------------------------------------------

        try:
            pipeline = sp.FacialBehaviorPipeline(
                frame_interval_seconds=interval
            )

            results = pipeline.analyze_video(
                str(temp_video_path)
            )

        except Exception as exc:
            # Keep detailed exception information in the server logs.
            print(f"[Analysis Error] {type(exc).__name__}: {exc}")

            raise HTTPException(
                status_code=422,
                detail=(
                    "Video analysis failed. Check that the video is "
                    "valid and that the required models are available."
                ),
            ) from exc

        if not results:
            raise HTTPException(
                status_code=422,
                detail="No frames were analyzed from the uploaded video.",
            )

        # -------------------------------------------------------------
        # 3. Convert pipeline results into a DataFrame
        # -------------------------------------------------------------

        df = pd.DataFrame(results)

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="The analysis pipeline returned no frame data.",
            )

        if "timestamp_seconds" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Pipeline output is missing the "
                    "timestamp_seconds column."
                ),
            )

        df["timestamp_seconds"] = pd.to_numeric(
            df["timestamp_seconds"],
            errors="coerce",
        )

        df = (
            df.dropna(subset=["timestamp_seconds"])
            .sort_values("timestamp_seconds")
            .reset_index(drop=True)
        )

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="No frames with valid timestamps were produced.",
            )

        # -------------------------------------------------------------
        # 4. Temporal expression analytics
        # -------------------------------------------------------------

        counts, runs, changes = an.build_temporal_summary(df)

        # -------------------------------------------------------------
        # 5. Head-pose and movement analytics
        # -------------------------------------------------------------

        pose_df, pose_summary, movement = (
            an.build_head_pose_analytics(df)
        )

        # -------------------------------------------------------------
        # 6. Data-quality analytics
        # -------------------------------------------------------------

        quality, _ = an.build_quality_summary(df)

        # -------------------------------------------------------------
        # 7. Build the complete session report
        # -------------------------------------------------------------

        report = an.build_session_report(
            df=df,
            filename=file.filename,
            frame_interval=interval,
            emotion_counts=counts,
            emotion_runs=runs,
            prediction_changes=changes,
            pose_summary=pose_summary,
            movement_summary=movement,
            quality=quality,
        )

        # -------------------------------------------------------------
        # 8. Save analysis outputs
        # -------------------------------------------------------------

        try:
            save_analysis_outputs(
                df=df,
                report=report,
            )

        except Exception as exc:
            print(f"[Output Save Error] {type(exc).__name__}: {exc}")

            raise HTTPException(
                status_code=500,
                detail="Analysis completed, but saving the output files failed.",
            ) from exc

        # -------------------------------------------------------------
        # 9. Prepare JSON response for React
        # -------------------------------------------------------------

        pose_summary_records = []

        if pose_summary is not None and not pose_summary.empty:
            pose_summary_records = records(
                pose_summary.reset_index().rename(
                    columns={"index": "Angle"}
                )
            )

        movement_records = (
            records(movement)
            if movement is not None and not movement.empty
            else []
        )

        return {
            "report": report,

            "frames": records(df),

            "pose_series": records(pose_df),

            "pose_summary": pose_summary_records,

            "movement_summary": movement_records,

            "emotion_counts": (
                records(counts)
                if counts is not None and not counts.empty
                else []
            ),

            "emotion_runs": (
                records(runs)
                if runs is not None and not runs.empty
                else []
            ),
        }

    except HTTPException:
        # Preserve intentional HTTP status codes and messages.
        raise

    except Exception as exc:
        print(f"[Unexpected API Error] {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during video analysis.",
        ) from exc

    finally:
        # -------------------------------------------------------------
        # 10. Always remove temporary video and close upload
        # -------------------------------------------------------------

        if temp_video_path is not None:
            temp_video_path.unlink(missing_ok=True)

        await file.close()


# ---------------------------------------------------------------------
# AI Assistant request model
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    report: dict = Field(
        ...,
        description="Analysis report for the current video session.",
    )

    history: list[dict] = Field(
        default_factory=list,
        description="Previous conversation messages.",
    )


# ---------------------------------------------------------------------
# AI Assistant endpoint
# ---------------------------------------------------------------------

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Generate an AI response grounded in the current analysis report.

    Input:
        question: User's question.
        report: Current video analysis report.
        history: Previous chat messages.

    Output:
        answer: AI-generated response.
    """

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        from backend.assistant import answer

        evaluation_metrics = an.load_verified_evaluation_metrics()

        response = answer(
            question=question,
            report=req.report,
            evaluation_metrics=evaluation_metrics,
            history=req.history,
        )

        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                "The AI assistant returned an empty response."
            )

        return {
            "answer": response.strip(),
        }

    except ValueError as exc:
        # Invalid user input or invalid request data.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        # Provider availability, exhausted retries, or other
        # temporary AI-service failures.
        print(f"[AI Assistant Error] {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI Assistant is temporarily unavailable. "
                "Please try again shortly."
            ),
        ) from exc

    except Exception as exc:
        print(f"[Unexpected Chat Error] {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=502,
            detail="The AI Assistant could not process your request.",
        ) from exc
