
from pathlib import Path

import cv2


class VideoProcessor:
    """Read a video and sample frames at a configurable interval."""

    def __init__(self, frame_interval_seconds=1.0):
        if frame_interval_seconds <= 0:
            raise ValueError("Frame interval must be positive.")

        self.frame_interval_seconds = frame_interval_seconds

    def process(self, video_path):
        """
        Yield sampled video frames.

        Each result contains:
            timestamp_seconds: approximate video timestamp
            frame: BGR image as a NumPy array
        """
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(
                "Video has invalid FPS or frame count."
            )

        frame_step = max(
            1, round(fps * self.frame_interval_seconds)
        )

        try:
            frame_index = 0

            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_index % frame_step == 0:
                    yield {
                        "timestamp_seconds": frame_index / fps,
                        "frame": frame,
                    }

                frame_index += 1

        finally:
            cap.release()
