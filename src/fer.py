
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# Project root: parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "balanced_best_phase3.keras"

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class FERModel:
    """Load the FER model once and predict emotions from face images."""

    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"FER model not found: {model_path}"
            )

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        print(f"FER model loaded: {model_path.name}")

    def predict(self, face_image):
        """
        Predict emotion scores from a face image.

        Args:
            face_image: BGR or RGB image as a NumPy array.
                        The image should contain a cropped face.

        Returns:
            Dictionary containing predicted emotion and scores.
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face image is empty.")

        # OpenCV images are usually BGR.
        if face_image.ndim == 2:
            face_image = cv2.cvtColor(
                face_image, cv2.COLOR_GRAY2RGB
            )
        elif face_image.ndim == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(
                face_image, cv2.COLOR_BGR2RGB
            )
        else:
            raise ValueError(
                "Expected a grayscale or 3-channel image."
            )

        # Match the model's expected input dimensions.
        face_image = cv2.resize(face_image, (224, 224))

        # Match the preprocessing used during training.
        face_image = face_image.astype(np.float32)
        face_image = (face_image / 127.5) - 1.0

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        batch = np.expand_dims(face_image, axis=0)

        scores = self.model.predict(batch, verbose=0)[0]

        if len(scores) != len(EMOTION_LABELS):
            raise ValueError(
                f"Expected 7 emotion scores, got {len(scores)}."
            )

        emotion_scores = {
            label: float(score)
            for label, score in zip(EMOTION_LABELS, scores)
        }

        predicted_emotion = max(
            emotion_scores,
            key=emotion_scores.get,
        )

        return {
            "emotion": predicted_emotion,
            "scores": emotion_scores,
        }
