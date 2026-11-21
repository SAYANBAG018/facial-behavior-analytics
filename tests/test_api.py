"""Tests for the FastAPI backend.

These use FastAPI's TestClient, so no server needs to be running.
The first import loads the FER model, so this file is slower than
the unit tests.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend import main as api_main  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)


def test_root_responds():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_reports_model_loaded():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["fer_model_loaded"] is True


def test_evaluation_metrics_are_served():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["accuracy"] > 0
    assert len(body["class_order"]) == 7


def test_analyze_rejects_non_video_file():
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"not a video", "text/plain")},
    )
    # 415 Unsupported Media Type is the intended response for wrong formats.
    assert response.status_code == 415


def test_chat_rejects_empty_question():
    response = client.post("/api/chat", json={"question": "   ", "report": {}})
    assert response.status_code == 400


def test_analyze_sample_video_end_to_end(sample_video, tmp_path, monkeypatch):
    # Write generated output to a temporary folder, not the project's outputs/.
    monkeypatch.setattr(api_main, "OUTPUT_DIR", tmp_path)

    with open(sample_video, "rb") as handle:
        response = client.post(
            "/api/analyze",
            files={"file": ("sample.mp4", handle, "video/mp4")},
            data={"interval": "1.0"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"]
    assert len(body["frames"]) > 0
    assert (tmp_path / "facial_behavior_results.csv").exists()
    assert (tmp_path / "facial_behavior_report.json").exists()
