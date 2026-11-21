"""Tests for MediaPipe facial feature extraction."""

import numpy as np

from src.features import FacialFeatureExtractor


def test_extract_runs_on_blank_frame():
    # A blank frame contains no face, so extract() should return None.
    # This checks that the pipeline runs end to end without errors.
    extractor = FacialFeatureExtractor()
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(frame)
        assert result is None or isinstance(result, dict)
    finally:
        extractor.close()
