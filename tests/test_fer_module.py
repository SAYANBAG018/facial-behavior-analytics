"""Tests for the FER (facial expression recognition) model wrapper."""

import numpy as np
import pytest

from src.fer import EMOTION_LABELS, MODEL_PATH, FERModel


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return FERModel()


def test_predict_returns_seven_finite_scores(model):
    image = np.full((300, 300, 3), 255, dtype=np.uint8)

    result = model.predict(image)

    assert result["emotion"] in EMOTION_LABELS
    assert set(result["scores"]) == set(EMOTION_LABELS)
    assert all(np.isfinite(v) for v in result["scores"].values())
