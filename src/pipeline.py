
import csv
from pathlib import Path

from src.fer import FERModel
from src.features import FacialFeatureExtractor
from src.video_processor import VideoProcessor


class FacialBehaviorPipeline:
    """Extract facial-expression and head-orientation signals from video."""

    EMOTION_LABELS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    def __init__(self, frame_interval_seconds=1.0):
        self.fer = FERModel()
        self.features = FacialFeatureExtractor()
        self.video = VideoProcessor(frame_interval_seconds)

    @staticmethod
    def _empty_result(timestamp):
        result = {
            "timestamp_seconds": timestamp,
            "emotion": "unknown",
        }

        result.update({
            f"emotion_score_{label}": None
            for label in FacialBehaviorPipeline.EMOTION_LABELS
        })

        result.update({
            "face_visible": False,
            "head_yaw_degrees": None,
            "head_pitch_degrees": None,
            "head_roll_degrees": None,
            "head_pose_available": False,
        })

        return result

    def analyze_video(self, video_path):
        results = []

        try:
            for sample in self.video.process(video_path):
                frame = sample["frame"]
                timestamp = sample["timestamp_seconds"]

                facial_features = self.features.extract(frame)

                if facial_features is None:
                    results.append(self._empty_result(timestamp))
                    continue

                points = facial_features["landmarks"]
                height, width = frame.shape[:2]

                x1 = max(0, int(points[:, 0].min() * width))
                y1 = max(0, int(points[:, 1].min() * height))
                x2 = min(width, int(points[:, 0].max() * width))
                y2 = min(height, int(points[:, 1].max() * height))

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    results.append(self._empty_result(timestamp))
                    continue

                emotion_result = self.fer.predict(face)

                result = {
                    "timestamp_seconds": timestamp,
                    "emotion": emotion_result["emotion"],
                    "face_visible": facial_features["face_visible"],
                    "head_yaw_degrees": facial_features[
                        "head_yaw_degrees"
                    ],
                    "head_pitch_degrees": facial_features[
                        "head_pitch_degrees"
                    ],
                    "head_roll_degrees": facial_features[
                        "head_roll_degrees"
                    ],
                    "head_pose_available": facial_features[
                        "head_pose_available"
                    ],
                }

                for label in self.EMOTION_LABELS:
                    result[f"emotion_score_{label}"] = (
                        emotion_result["scores"][label]
                    )

                results.append(result)

        finally:
            self.features.close()

        return results

    @staticmethod
    def save_csv(results, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "timestamp_seconds",
            "emotion",
            *[
                f"emotion_score_{label}"
                for label in FacialBehaviorPipeline.EMOTION_LABELS
            ],
            "face_visible",
            "head_yaw_degrees",
            "head_pitch_degrees",
            "head_roll_degrees",
            "head_pose_available",
        ]

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

        return output_path
