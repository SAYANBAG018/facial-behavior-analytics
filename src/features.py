
import os

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FacialFeatureExtractor:
    """Extract facial landmarks, face geometry, and approximate head pose."""

    def __init__(self):
        """Initialize MediaPipe face detection and head-pose models."""

        self.face_landmarker = None

        try:
            # Resolve the project root independently of the working directory.
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            model_path = os.path.join(
                project_root,
                "models",
                "face_landmarker.task",
            )

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"FaceLandmarker model not found: {model_path}"
                )

            # Read the model file into memory.
            # This avoids MediaPipe incorrectly resolving an absolute
            # Windows path relative to its own package directory.
            with open(model_path, "rb") as model_file:
                model_buffer = model_file.read()

            if not model_buffer:
                raise ValueError(
                    f"FaceLandmarker model file is empty: {model_path}"
                )

            base_options = python.BaseOptions(
                model_asset_buffer=model_buffer
            )

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )

            self.face_landmarker = (
                vision.FaceLandmarker.create_from_options(options)
            )

        except Exception:
            self.close()
            raise

    def _estimate_head_pose(self, result):
        """
        Estimate approximate yaw, pitch, and roll
        from the MediaPipe facial transformation matrix.

        `result` is the FaceLandmarker detection for the same frame, so
        detection runs once per frame instead of twice.
        """

        try:
            matrices = result.facial_transformation_matrixes

            if matrices is None or len(matrices) == 0:
                return None

            transform = np.asarray(
                matrices[0],
                dtype=np.float64,
            )

            # Validate transformation matrix.
            if transform.shape != (4, 4):
                print(
                    "HEAD POSE DEBUG: Unexpected matrix shape:",
                    transform.shape,
                )
                return None

            if not np.all(np.isfinite(transform)):
                print(
                    "HEAD POSE DEBUG: Matrix contains non-finite values."
                )
                return None

            rotation_matrix = transform[:3, :3]

            # Reject degenerate rotation matrices.
            if np.linalg.matrix_rank(rotation_matrix) < 3:
                print(
                    "HEAD POSE DEBUG: Degenerate rotation matrix."
                )
                return None

            # Correct small numerical deviations from orthogonality.
            u, _, vt = np.linalg.svd(rotation_matrix)

            rotation_matrix = u @ vt

            # Ensure a proper rotation matrix with determinant +1.
            if np.linalg.det(rotation_matrix) < 0:
                u[:, -1] *= -1
                rotation_matrix = u @ vt

            # Validate the corrected rotation matrix.
            if not np.allclose(
                rotation_matrix.T @ rotation_matrix,
                np.eye(3),
                atol=1e-5,
            ):
                print(
                    "HEAD POSE DEBUG: Rotation matrix validation failed."
                )
                return None

            if not np.isclose(
                np.linalg.det(rotation_matrix),
                1.0,
                atol=1e-5,
            ):
                return None

            # Extract Euler angles in degrees.
            angles, *_ = cv2.RQDecomp3x3(
                rotation_matrix
            )

            pitch, yaw, roll = (
                float(value) for value in angles
            )

            if not np.all(
                np.isfinite([yaw, pitch, roll])
            ):
                print(
                    "HEAD POSE DEBUG: Non-finite Euler angles."
                )
                return None

            return {
                "head_yaw_degrees": yaw,
                "head_pitch_degrees": pitch,
                "head_roll_degrees": roll,
                "head_pose_available": True,
                "head_pose_reprojection_error": None,
            }

        except (
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
        ) as exc:
            print(
                "HEAD POSE VALIDATION ERROR:",
                repr(exc),
            )
            return None

        except Exception as exc:
            print(
                "HEAD POSE DEBUG ERROR:",
                repr(exc),
            )
            return None

    def extract(self, frame):
        """
        Extract facial features from a BGR frame.

        Returns:
            dict: Extracted facial features if a face is detected.
            None: If no face is detected.
        """

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a NumPy array."
            )

        if frame.size == 0:
            raise ValueError(
                "Input frame is empty."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Input frame must have three color channels (BGR)."
            )

        # Convert OpenCV BGR frame to RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb_frame = np.ascontiguousarray(
            rgb_frame
        )

        # One detection per frame: landmarks and head pose share it.
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.face_landmarker.detect(image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        points = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in landmarks
            ],
            dtype=np.float32,
        )

        # Calculate normalized face bounding coordinates.
        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))

        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))

        features = {
            "landmarks": points,

            "face_center_x": float(
                np.mean(points[:, 0])
            ),

            "face_center_y": float(
                np.mean(points[:, 1])
            ),

            "face_width": x_max - x_min,
            "face_height": y_max - y_min,

            "face_visible": True,

            "head_yaw_degrees": None,
            "head_pitch_degrees": None,
            "head_roll_degrees": None,

            "head_pose_available": False,
            "head_pose_reprojection_error": None,
        }

        # Head pose from the same detection result.
        pose = self._estimate_head_pose(
            result
        )

        if pose is not None:
            features.update(pose)

        return features

    def close(self):
        """Release MediaPipe resources safely."""

        if self.face_landmarker is not None:
            try:
                self.face_landmarker.close()
            except Exception as exc:
                print(
                    "FACE LANDMARKER CLEANUP ERROR:",
                    repr(exc),
                )
            finally:
                self.face_landmarker = None
