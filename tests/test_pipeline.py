"""End-to-end test of the analysis pipeline on the sample video."""

import csv

from src.pipeline import FacialBehaviorPipeline


def test_pipeline_analyzes_sample_video(sample_video, tmp_path):
    pipeline = FacialBehaviorPipeline(frame_interval_seconds=1.0)

    results = pipeline.analyze_video(str(sample_video))

    assert results, "No frames were analyzed."
    assert {"face_visible", "head_pose_available"} <= set(results[0])

    output_path = pipeline.save_csv(results, tmp_path / "results.csv")

    with open(output_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(results)
