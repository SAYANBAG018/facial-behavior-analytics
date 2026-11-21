"""Tests for the experimental engagement heuristic.

These check the code path only. The heuristic is not a validated
measure of engagement.
"""

from src.engagement import EngagementEstimator


def test_estimate_returns_score_in_range():
    estimator = EngagementEstimator()
    emotion_result = {
        "emotion": "happy",
        "scores": {
            "angry": 0.05, "disgust": 0.01, "fear": 0.02, "happy": 0.75,
            "neutral": 0.10, "sad": 0.02, "surprise": 0.05,
        },
    }
    facial_features = {
        "face_center_x": 0.5, "face_center_y": 0.5,
        "face_width": 0.3, "face_height": 0.4,
    }

    result = estimator.estimate(emotion_result, facial_features)

    assert result["score"] is not None
    assert 0 <= result["score"] <= 100
    assert result["label"] == "higher_proxy_score"


def test_estimate_handles_missing_inputs():
    result = EngagementEstimator().estimate(None, None)
    assert result["score"] is None
