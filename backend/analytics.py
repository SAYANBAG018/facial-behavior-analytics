
"""
Analytics functions for Multimodal Facial Behavior Analytics.

Responsibilities:
- Temporal expression analytics.
- Consecutive predicted-expression runs.
- Head-pose statistics and movement analysis.
- Data-quality measurements.
- Verified classifier evaluation metrics.
- JSON-serializable session reports.

Important:
All video-session metrics describe sampled frames,
not necessarily every frame in the original video.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================
# PROJECT CONFIGURATION
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

POSE_COLUMNS = [
    "head_yaw_degrees",
    "head_pitch_degrees",
    "head_roll_degrees",
]


# =========================================
# GENERAL HELPERS
# =========================================

def _ensure_columns(df, columns):
    """
    Ensure required columns exist.

    Missing columns are added with appropriate empty values.
    The original DataFrame is not modified.
    """

    result = df.copy()

    for column in columns:
        if column not in result.columns:
            if column in {
                "face_visible",
                "head_pose_available",
            }:
                result[column] = False
            else:
                result[column] = np.nan

    return result


def _to_boolean(series):
    """
    Convert a pandas Series into standard NumPy-backed booleans.

    Handles:
    - True / False
    - 1 / 0
    - "true" / "false"
    - Missing values

    Missing values are treated as False.

    Explicitly avoids returning PyArrow-backed Boolean arrays.
    """

    if pd.api.types.is_bool_dtype(series):
        values = series.fillna(False).to_numpy(dtype=bool)

    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0)

        values = numeric.ne(0).to_numpy(dtype=bool)

    else:
        normalized = (
            series.astype("string")
            .str.strip()
            .str.lower()
        )

        values = normalized.isin(
            ["true", "1", "yes", "y"]
        ).to_numpy(dtype=bool)

    return pd.Series(
        values,
        index=series.index,
        dtype=bool,
    )


def _safe_float(value, digits=4):
    """
    Convert a numerical value to a JSON-safe Python float.

    Non-finite values become None.
    """

    if value is None:
        return None

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return round(number, digits)

    except (TypeError, ValueError, OverflowError):
        return None


def _safe_int(value):
    """Convert a value into a safe Python integer."""

    try:
        if pd.isna(value):
            return None

        return int(value)

    except (TypeError, ValueError, OverflowError):
        return None


def _clean_for_json(value):
    """
    Recursively convert NumPy and pandas values
    into JSON-compatible Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): _clean_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _clean_for_json(item)
            for item in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return _safe_float(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int)):
        return value

    return str(value)


def _prepare_dataframe(df):
    """
    Prepare a copy of the analysis DataFrame.

    Ensures that the columns required by analytics
    are available and timestamps are numeric.
    """

    if df is None:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Analytics input must be a pandas DataFrame."
        )

    required_columns = [
        "timestamp_seconds",
        "emotion",
        "face_visible",
        "head_pose_available",
        *POSE_COLUMNS,
    ]

    result = _ensure_columns(
        df,
        required_columns,
    )

    result["timestamp_seconds"] = pd.to_numeric(
        result["timestamp_seconds"],
        errors="coerce",
    )

    for column in POSE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # These columns are explicitly converted to ordinary
    # NumPy-backed bool Series.
    result["face_visible"] = _to_boolean(
        result["face_visible"]
    )

    result["head_pose_available"] = _to_boolean(
        result["head_pose_available"]
    )

    result["emotion"] = (
        result["emotion"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return result


# =========================================
# TEMPORAL EXPRESSION ANALYTICS
# =========================================

def build_temporal_summary(df):
    """
    Calculate:

    1. Predicted-expression frequency.
    2. Percentage of valid predicted samples.
    3. Prediction changes.
    4. Consecutive runs of identical predictions.

    Only valid expression predictions with valid
    timestamps are included in temporal calculations.

    Returns:
        counts: DataFrame
        runs: DataFrame
        prediction_changes: int
    """

    df = _prepare_dataframe(df)

    detected = df.loc[
        df["emotion"].isin(EMOTION_LABELS),
        ["timestamp_seconds", "emotion"],
    ].copy()

    # A temporal calculation requires a valid timestamp.
    detected = detected.dropna(
        subset=["timestamp_seconds"]
    )

    if detected.empty:
        return (
            pd.DataFrame(
                columns=[
                    "predicted_expression",
                    "sample_count",
                    "percentage_of_detected_samples",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "predicted_expression",
                    "start_seconds",
                    "last_sample_seconds",
                    "sample_count",
                    "sampled_span_seconds",
                ]
            ),
            0,
        )

    detected = detected.sort_values(
        "timestamp_seconds",
        kind="stable",
    ).reset_index(drop=True)

    total_detected = len(detected)

    # -------------------------------------
    # A. Expression frequency
    # -------------------------------------

    counts = (
        detected["emotion"]
        .value_counts()
        .reindex(
            EMOTION_LABELS,
            fill_value=0,
        )
        .rename_axis("predicted_expression")
        .reset_index(name="sample_count")
    )

    counts["percentage_of_detected_samples"] = (
        counts["sample_count"]
        / total_detected
        * 100
    ).round(1)

    counts["sample_count"] = counts[
        "sample_count"
    ].astype(int)

    # -------------------------------------
    # B. Prediction changes
    # -------------------------------------

    # Convert the emotion labels to a regular NumPy array.
    # This avoids PyArrow Boolean cumulative operations.
    emotions = detected["emotion"].astype(str).to_numpy()

    # The first valid prediction starts the first run.
    run_starts = np.empty(
        len(emotions),
        dtype=bool,
    )

    run_starts[0] = True

    if len(emotions) > 1:
        run_starts[1:] = (
            emotions[1:] != emotions[:-1]
        )

    # Every run after the first represents one prediction change.
    prediction_changes = int(
        np.count_nonzero(run_starts) - 1
    )

    prediction_changes = max(
        prediction_changes,
        0,
    )

    # -------------------------------------
    # C. Consecutive prediction runs
    # -------------------------------------

    # NumPy cumulative sum is used instead of Pandas cumsum
    # on a potentially PyArrow-backed Boolean Series.
    run_ids = np.cumsum(
        run_starts.astype(np.int64),
        dtype=np.int64,
    )

    detected["run_id"] = run_ids

    runs = (
        detected.groupby(
            "run_id",
            sort=True,
        )
        .agg(
            predicted_expression=("emotion", "first"),
            start_seconds=("timestamp_seconds", "first"),
            last_sample_seconds=("timestamp_seconds", "last"),
            sample_count=("emotion", "size"),
        )
        .reset_index(drop=True)
    )

    runs["sampled_span_seconds"] = (
        runs["last_sample_seconds"]
        - runs["start_seconds"]
    ).round(2)

    runs["start_seconds"] = (
        runs["start_seconds"].round(2)
    )

    runs["last_sample_seconds"] = (
        runs["last_sample_seconds"].round(2)
    )

    runs["sample_count"] = (
        runs["sample_count"].astype(int)
    )

    return counts, runs, prediction_changes


# =========================================
# HEAD-POSE ANALYTICS
# =========================================

def build_head_pose_analytics(df):
    """
    Calculate head-pose analytics.

    Measurements:
    - Mean orientation.
    - Standard deviation.
    - Minimum and maximum orientation.
    - Orientation range.
    - Absolute angle changes.
    - Approximate change rates.

    Yaw, pitch, and roll are estimated orientation
    measurements, not direct eye-gaze measurements.

    Returns:
        available: DataFrame of valid pose measurements
        summary: DataFrame of orientation statistics
        movement_summary: DataFrame of movement statistics
    """

    df = _prepare_dataframe(df)

    available = df.loc[
        df["head_pose_available"],
        ["timestamp_seconds"] + POSE_COLUMNS,
    ].copy()

    # Require all three numeric orientation measurements.
    available = available.dropna(
        subset=["timestamp_seconds"] + POSE_COLUMNS
    )

    available = available.sort_values(
        "timestamp_seconds",
        kind="stable",
    ).reset_index(drop=True)

    if available.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "Mean (degrees)",
                "Std. deviation (degrees)",
                "Minimum (degrees)",
                "Maximum (degrees)",
                "Range (degrees)",
            ]
        )

        empty_movement = pd.DataFrame(
            columns=[
                "Metric",
                "Value (degrees)",
                "Value (degrees/second)",
            ]
        )

        return (
            available,
            empty_summary,
            empty_movement,
        )

    # -------------------------------------
    # A. Orientation statistics
    # -------------------------------------

    summary = available[POSE_COLUMNS].agg(
        ["mean", "std", "min", "max"]
    ).T

    summary.columns = [
        "Mean (degrees)",
        "Std. deviation (degrees)",
        "Minimum (degrees)",
        "Maximum (degrees)",
    ]

    summary["Range (degrees)"] = (
        summary["Maximum (degrees)"]
        - summary["Minimum (degrees)"]
    )

    summary.index.name = "Orientation"

    # -------------------------------------
    # B. Absolute angle changes
    # -------------------------------------

    movement = available[
        ["timestamp_seconds"]
    ].copy()

    movement["time_gap_seconds"] = (
        available["timestamp_seconds"].diff()
    )

    for column in POSE_COLUMNS:
        change_column = f"{column}_change"

        movement[change_column] = (
            available[column].diff().abs()
        )

    change_columns = [
        f"{column}_change"
        for column in POSE_COLUMNS
    ]

    movement["mean_absolute_angle_change_degrees"] = (
        movement[change_columns].mean(axis=1)
    )

    # -------------------------------------
    # C. Approximate change rates
    # -------------------------------------

    valid_time_gap = movement[
        "time_gap_seconds"
    ].where(
        movement["time_gap_seconds"] > 0
    )

    rate_columns = []

    for column in POSE_COLUMNS:
        change_column = f"{column}_change"

        rate_column = (
            f"{column}_change_per_second"
        )

        movement[rate_column] = (
            movement[change_column]
            / valid_time_gap
        )

        rate_columns.append(rate_column)

    movement[
        "mean_absolute_angle_change_degrees_per_second"
    ] = movement[rate_columns].mean(axis=1)

    # -------------------------------------
    # D. Movement summary
    # -------------------------------------

    movement_summary = pd.DataFrame(
        {
            "Metric": [
                "Mean absolute yaw change",
                "Mean absolute pitch change",
                "Mean absolute roll change",
                "Mean combined angle change",
                "Maximum combined angle change",
                "Mean absolute yaw change rate",
                "Mean absolute pitch change rate",
                "Mean absolute roll change rate",
                "Mean combined angle change rate",
                "Maximum combined angle change rate",
            ],

            "Value (degrees)": [
                movement[
                    "head_yaw_degrees_change"
                ].mean(),

                movement[
                    "head_pitch_degrees_change"
                ].mean(),

                movement[
                    "head_roll_degrees_change"
                ].mean(),

                movement[
                    "mean_absolute_angle_change_degrees"
                ].mean(),

                movement[
                    "mean_absolute_angle_change_degrees"
                ].max(),

                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],

            "Value (degrees/second)": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,

                movement[
                    "head_yaw_degrees_change_per_second"
                ].mean(),

                movement[
                    "head_pitch_degrees_change_per_second"
                ].mean(),

                movement[
                    "head_roll_degrees_change_per_second"
                ].mean(),

                movement[
                    "mean_absolute_angle_change_degrees_per_second"
                ].mean(),

                movement[
                    "mean_absolute_angle_change_degrees_per_second"
                ].max(),
            ],
        }
    )

    return (
        available,
        summary,
        movement_summary,
    )


# =========================================
# VERIFIED MODEL EVALUATION METRICS
# =========================================

def load_verified_evaluation_metrics():
    """
    Load separately measured classifier evaluation results.

    Expected file:
        outputs/verified_evaluation_summary.json

    Returns:
        Dictionary containing evaluation results.

    An empty dictionary is returned when the file
    is missing or invalid.
    """

    evaluation_path = (
        PROJECT_ROOT
        / "outputs"
        / "verified_evaluation_summary.json"
    )

    if not evaluation_path.is_file():
        return {}

    try:
        with evaluation_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {}

        if not isinstance(data.get("metrics"), dict):
            return {}

        return _clean_for_json(data)

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return {}


# =========================================
# DATA QUALITY ANALYTICS
# =========================================

def build_quality_summary(df):
    """
    Calculate sampled-frame data-quality measurements.

    Metrics include:
    - Total sampled frames.
    - Successful face detections.
    - Face detection availability.
    - Head-pose availability.
    - Valid expression predictions.
    - Missing face detections.
    - Missing head-pose measurements.
    - Missing orientation values.

    Important:
    Percentages refer to sampled frames only.
    They do not indicate complete original-video coverage.

    Returns:
        quality: Dictionary of metrics
        quality_table: DataFrame for dashboard display
    """

    df = _prepare_dataframe(df)

    total = len(df)

    if total == 0:
        return (
            {
                "frames_analyzed": 0,
                "total_sampled_frames": 0,
                "frames_with_face_detected": 0,
                "face_detection_rate_percent": 0.0,
                "frames_with_head_pose": 0,
                "head_pose_availability_percent": 0.0,
                "frames_with_valid_expression": 0,
                "valid_expression_rate_percent": 0.0,
                "missing_face_detection_frames": 0,
                "missing_head_pose_frames": 0,
                "missing_yaw_values": 0,
                "missing_pitch_values": 0,
                "missing_roll_values": 0,
                "frames_with_complete_head_pose_values": 0,
                "frames_with_missing_timestamps": 0,
            },
            pd.DataFrame(
                columns=["Metric", "Value"]
            ),
        )

    # -------------------------------------
    # A. Face detection availability
    # -------------------------------------

    face_visible = df["face_visible"]

    face_count = int(
        face_visible.sum()
    )

    # -------------------------------------
    # B. Head-pose availability
    # -------------------------------------

    pose_available = df["head_pose_available"]

    pose_count = int(
        pose_available.sum()
    )

    # Count frames with all three numeric pose values.
    complete_pose = (
        df[POSE_COLUMNS].notna().all(axis=1)
    )

    complete_pose_count = int(
        (pose_available & complete_pose).sum()
    )

    # -------------------------------------
    # C. Expression prediction availability
    # -------------------------------------

    emotion_valid = df["emotion"].isin(
        EMOTION_LABELS
    )

    emotion_count = int(
        emotion_valid.sum()
    )

    # -------------------------------------
    # D. Missing values
    # -------------------------------------

    missing_timestamps = int(
        df["timestamp_seconds"].isna().sum()
    )

    missing_yaw = int(
        df["head_yaw_degrees"].isna().sum()
    )

    missing_pitch = int(
        df["head_pitch_degrees"].isna().sum()
    )

    missing_roll = int(
        df["head_roll_degrees"].isna().sum()
    )

    # -------------------------------------
    # E. Build quality dictionary
    # -------------------------------------

    quality = {
        "frames_analyzed": int(total),

        "total_sampled_frames": int(total),

        "frames_with_face_detected": face_count,

        "face_detection_rate_percent": round(
            face_count / total * 100,
            2,
        ),

        "frames_with_head_pose": pose_count,

        "head_pose_availability_percent": round(
            pose_count / total * 100,
            2,
        ),

        "frames_with_valid_expression": emotion_count,

        "valid_expression_rate_percent": round(
            emotion_count / total * 100,
            2,
        ),

        "missing_face_detection_frames": int(
            total - face_count
        ),

        "missing_head_pose_frames": int(
            total - pose_count
        ),

        "missing_yaw_values": missing_yaw,

        "missing_pitch_values": missing_pitch,

        "missing_roll_values": missing_roll,

        "frames_with_complete_head_pose_values": (
            complete_pose_count
        ),

        "frames_with_missing_timestamps": (
            missing_timestamps
        ),
    }

    quality_table = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": value,
            }
            for key, value in quality.items()
        ]
    )

    return quality, quality_table


# =========================================
# SESSION REPORT GENERATION
# =========================================

def build_session_report(
    df,
    filename,
    frame_interval,
    emotion_counts,
    emotion_runs,
    prediction_changes,
    pose_summary,
    movement_summary,
    quality,
):
    """
    Build the session report.

    The report is designed for:
    - Frontend dashboard display.
    - JSON download.
    - AI Assistant grounding.
    - Session analysis.

    All session metrics describe sampled frames.
    """

    df = _prepare_dataframe(df)

    # -------------------------------------
    # A. Dominant predicted expression
    # -------------------------------------

    valid_emotions = df.loc[
        df["emotion"].isin(EMOTION_LABELS),
        "emotion",
    ]

    if valid_emotions.empty:
        dominant_expression = None
    else:
        dominant_expression = str(
            valid_emotions.value_counts().idxmax()
        )

    # -------------------------------------
    # B. Last sampled timestamp
    # -------------------------------------

    valid_timestamps = df[
        "timestamp_seconds"
    ].dropna()

    if valid_timestamps.empty:
        last_timestamp = None
    else:
        last_timestamp = _safe_float(
            valid_timestamps.max()
        )

    # -------------------------------------
    # C. Orientation statistics
    # -------------------------------------

    pose_stats = {}

    if (
        isinstance(pose_summary, pd.DataFrame)
        and not pose_summary.empty
    ):
        pose_stats = {
            str(index): {
                str(column): _safe_float(value)
                for column, value in row.items()
            }
            for index, row in pose_summary.iterrows()
        }

    # -------------------------------------
    # D. Movement statistics
    # -------------------------------------

    movement_stats = {}
    movement_rate_stats = {}

    if (
        isinstance(movement_summary, pd.DataFrame)
        and not movement_summary.empty
    ):
        for _, row in movement_summary.iterrows():
            metric = str(row["Metric"])

            degree_value = _safe_float(
                row.get("Value (degrees)")
            )

            rate_value = _safe_float(
                row.get("Value (degrees/second)")
            )

            if degree_value is not None:
                movement_stats[metric] = degree_value

            if rate_value is not None:
                movement_rate_stats[metric] = rate_value

    # -------------------------------------
    # E. Expression distributions
    # -------------------------------------

    if (
        isinstance(emotion_counts, pd.DataFrame)
        and not emotion_counts.empty
    ):
        expression_distribution = (
            emotion_counts.to_dict(orient="records")
        )
    else:
        expression_distribution = []

    # -------------------------------------
    # F. Consecutive prediction runs
    # -------------------------------------

    if (
        isinstance(emotion_runs, pd.DataFrame)
        and not emotion_runs.empty
    ):
        consecutive_runs = (
            emotion_runs.to_dict(orient="records")
        )
    else:
        consecutive_runs = []

    # -------------------------------------
    # G. Quality metrics
    # -------------------------------------

    quality = (
        quality
        if isinstance(quality, dict)
        else {}
    )

    quality = _clean_for_json(quality)

    # -------------------------------------
    # H. Sampling information
    # -------------------------------------

    try:
        sampling_interval = float(frame_interval)

        if not np.isfinite(sampling_interval):
            sampling_interval = None

    except (TypeError, ValueError, OverflowError):
        sampling_interval = None

    total_sampled_frames = len(df)

    # -------------------------------------
    # I. Interpretation limitations
    # -------------------------------------

    interpretation_limits = [
        (
            "All video-session measurements refer to sampled "
            "frames, not necessarily every frame in the original video."
        ),
        (
            "Expression labels are model predictions and do not "
            "establish a person's actual emotional state."
        ),
        (
            "Model scores are raw outputs unless calibration "
            "has been independently established."
        ),
        (
            "Head orientation does not establish gaze direction, "
            "attention, or intention."
        ),
        (
            "Head-pose measurements are estimates and may contain "
            "pose-estimation noise."
        ),
        (
            "Consecutive prediction runs describe sampled labels, "
            "not verified real-world expression durations."
        ),
        (
            "Successful detection does not establish that a "
            "prediction is correct."
        ),
    ]

    # -------------------------------------
    # J. Final report
    # -------------------------------------

    report = {
        "project": "Multimodal Facial Behavior Analytics",

        "source_video": str(filename),

        "sampling_interval_seconds": (
            _safe_float(sampling_interval)
        ),

        "last_sampled_timestamp_seconds": (
            last_timestamp
        ),

        "total_sampled_frames": int(
            total_sampled_frames
        ),

        "dominant_predicted_expression": (
            dominant_expression
        ),

        "prediction_changes": int(
            prediction_changes or 0
        ),

        "expression_distribution": (
            expression_distribution
        ),

        "consecutive_prediction_runs": (
            consecutive_runs
        ),

        "head_pose_statistics": pose_stats,

        "head_pose_movement_statistics": (
            movement_stats
        ),

        "head_pose_movement_rate_statistics_degrees_per_second": (
            movement_rate_stats
        ),

        "data_quality": quality,

        "interpretation_limits": (
            interpretation_limits
        ),
    }

    return _clean_for_json(report)
