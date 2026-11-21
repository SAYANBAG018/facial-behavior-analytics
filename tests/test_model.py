"""Checks that the saved Keras model loads and has the expected shape.

Failures raise, so pytest reports them.
"""

import numpy as np
import pytest

from src.fer import EMOTION_LABELS, MODEL_PATH

tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def keras_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def test_model_output_matches_emotion_classes(keras_model):
    assert keras_model.output_shape[-1] == len(EMOTION_LABELS)


def test_model_prediction_is_finite(keras_model):
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)

    prediction = keras_model.predict(dummy, verbose=0)

    assert prediction.shape == (1, len(EMOTION_LABELS))
    assert np.isfinite(prediction).all()
