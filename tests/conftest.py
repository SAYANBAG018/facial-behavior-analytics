"""Shared pytest configuration.

Makes the project root importable so tests can use `src` and `backend`
without installing the project as a package.

Tests generate their own short synthetic video, so no personal footage
is needed in the repository.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """Create a 5-second synthetic video (no real face) for pipeline tests."""
    path = tmp_path_factory.mktemp("video") / "synthetic.avi"

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (320, 240),
    )
    assert writer.isOpened(), "Could not create synthetic test video."

    for i in range(50):
        frame = np.full((240, 320, 3), 40 + i, dtype=np.uint8)
        cv2.circle(frame, (160, 120), 60, (200, 180, 160), -1)
        writer.write(frame)

    writer.release()
    return path
