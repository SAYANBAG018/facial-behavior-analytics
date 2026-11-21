"""Tests for frame sampling from a generated video."""

import numpy as np
import cv2

from src.video_processor import VideoProcessor


def test_samples_one_frame_per_second(tmp_path):
    video_path = tmp_path / "generated.avi"
    width, height, fps = 160, 120, 10

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened(), "Could not create test video."

    # 30 frames at 10 fps = 3 seconds of video.
    for i in range(30):
        writer.write(np.full((height, width, 3), i * 5, dtype=np.uint8))
    writer.release()

    samples = list(VideoProcessor(frame_interval_seconds=1.0).process(video_path))

    assert len(samples) == 3
    assert all(item["frame"].shape == (height, width, 3) for item in samples)
