"""
Blink Detection using MediaPipe Face Landmarker (Tasks API)
Supports: Live camera, video files, and image frame arrays
Compatible with MediaPipe >= 0.10.0
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import urllib.request
import os


@dataclass
class BlinkResult:
    """Result of blink detection for a single frame."""

    left_ear: float  # Eye Aspect Ratio for left eye
    right_ear: float  # Eye Aspect Ratio for right eye
    avg_ear: float  # Average EAR
    left_blink: bool  # Left eye blink detected
    right_blink: bool  # Right eye blink detected
    blink: bool  # Both eyes blink detected
    blink_count: int  # Total blink count so far
    frame: Optional[np.ndarray] = None  # Annotated frame (if requested)


class BlinkDetector:
    """
    Blink detection using MediaPipe Face Landmarker (Tasks API).

    Uses Eye Aspect Ratio (EAR) to detect blinks.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    When the eye is open, EAR is relatively constant.
    When the eye closes, EAR drops significantly.
    """

    # MediaPipe Face Landmarker eye landmarks indices
    # These indices correspond to the 478 face landmarks
    # Left eye landmarks (from user's perspective)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks (from user's perspective)
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    # Model URL for Face Landmarker
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_FILENAME = "face_landmarker.task"

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the blink detector.

        Args:
            ear_threshold: EAR threshold below which a blink is detected (default: 0.21)
            consecutive_frames: Number of consecutive frames below threshold for a blink (default: 2)
            min_detection_confidence: Minimum confidence for face detection (default: 0.5)
            min_tracking_confidence: Minimum confidence for face tracking (default: 0.5)
            max_num_faces: Maximum number of faces to detect (default: 1)
            model_path: Path to the face_landmarker.task model file (will download if not provided)
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        # Get or download model
        if model_path is None:
            model_path = self._get_model_path()

        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # State tracking
        self.blink_count = 0
        self.frame_counter = 0  # Frames with EAR below threshold
        self.blink_in_progress = False

    def _get_model_path(self) -> str:
        """Get the model path, downloading if necessary."""
        # Check in current directory first
        if os.path.exists(self.MODEL_FILENAME):
            return self.MODEL_FILENAME

        # Check in user's home directory
        home_dir = Path.home()
        model_dir = home_dir / ".mediapipe" / "models"
        model_path = model_dir / self.MODEL_FILENAME

        if model_path.exists():
            return str(model_path)

        # Download the model
        print(f"Downloading Face Landmarker model...")
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(self.MODEL_URL, str(model_path))
            print(f"Model downloaded to: {model_path}")
            return str(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}\n"
                f"Please download manually from:\n{self.MODEL_URL}\n"
                f"And place it in the current directory or pass the path via model_path parameter."
            )

    def _calculate_ear(
        self, landmarks: list, indices: list, frame_shape: tuple
    ) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) for one eye.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        h, w = frame_shape[:2]

        # Get the 6 eye landmarks
        points = []
        for idx in indices:
            lm = landmarks[idx]
            points.append(np.array([lm.x * w, lm.y * h]))

        p1, p2, p3, p4, p5, p6 = points

        # Calculate distances
        vertical1 = np.linalg.norm(p2 - p6)  # |p2-p6|
        vertical2 = np.linalg.norm(p3 - p5)  # |p3-p5|
        horizontal = np.linalg.norm(p1 - p4)  # |p1-p4|

        # Calculate EAR
        if horizontal == 0:
            return 0.0
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def _draw_eye_landmarks(
        self, frame: np.ndarray, landmarks: list, indices: list, color: tuple
    ) -> None:
        """Draw eye landmarks on the frame."""
        h, w = frame.shape[:2]
        points = []
        for idx in indices:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, color, -1)

        # Draw eye contour
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], True, color, 1)

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> Optional[BlinkResult]:
        """
        Process a single frame for blink detection.

        Args:
            frame: BGR image frame (numpy array)
            draw: Whether to draw annotations on the frame

        Returns:
            BlinkResult object or None if no face detected
        """
        if frame is None:
            return None

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        # Get the first face landmarks
        face_landmarks = results.face_landmarks[0]

        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(
            face_landmarks, self.LEFT_EYE_INDICES, frame.shape
        )
        right_ear = self._calculate_ear(
            face_landmarks, self.RIGHT_EYE_INDICES, frame.shape
        )
        avg_ear = (left_ear + right_ear) / 2.0

        # Detect blinks
        left_blink = left_ear < self.ear_threshold
        right_blink = right_ear < self.ear_threshold

        # Track consecutive frames for blink counting
        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consecutive_frames:
                self.blink_count += 1
                self.blink_in_progress = False
            self.frame_counter = 0

        # Current blink state
        blink = left_blink and right_blink

        # Draw annotations if requested
        annotated_frame = None
        if draw:
            annotated_frame = frame.copy()

            # Draw eye landmarks
            eye_color = (
                (0, 0, 255) if blink else (0, 255, 0)
            )  # Red if blinking, green otherwise
            self._draw_eye_landmarks(
                annotated_frame, face_landmarks, self.LEFT_EYE_INDICES, eye_color
            )
            self._draw_eye_landmarks(
                annotated_frame, face_landmarks, self.RIGHT_EYE_INDICES, eye_color
            )

            # Draw info text
            cv2.putText(
                annotated_frame,
                f"EAR: {avg_ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Blinks: {self.blink_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Left EAR: {left_ear:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                annotated_frame,
                f"Right EAR: {right_ear:.3f}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            if blink:
                cv2.putText(
                    annotated_frame,
                    "BLINK!",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

        return BlinkResult(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            left_blink=left_blink,
            right_blink=right_blink,
            blink=blink,
            blink_count=self.blink_count,
            frame=annotated_frame,
        )

    def process_frames(
        self, frames: list[np.ndarray], draw: bool = False
    ) -> list[BlinkResult]:
        """
        Process multiple image frames for blink detection.

        Args:
            frames: List of BGR image frames (numpy arrays)
            draw: Whether to draw annotations on frames

        Returns:
            List of BlinkResult objects (None entries for frames with no face detected)
        """
        results = []
        for frame in frames:
            result = self.process_frame(frame, draw=draw)
            results.append(result)
        return results

    def run_on_camera(
        self, camera_id: int = 0, window_name: str = "Blink Detection"
    ) -> None:
        """
        Run blink detection on live camera feed.

        Args:
            camera_id: Camera device ID (default: 0)
            window_name: Name of the display window
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Press 'q' to quit, 'r' to reset blink count")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Mirror the frame for more intuitive interaction
                frame = cv2.flip(frame, 1)

                result = self.process_frame(frame, draw=True)

                if result and result.frame is not None:
                    cv2.imshow(window_name, result.frame)
                else:
                    # Show original frame with "No face detected" message
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.reset()
                    print("Blink count reset")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_on_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        window_name: str = "Blink Detection",
        show_preview: bool = True,
    ) -> list[BlinkResult]:
        """
        Run blink detection on a video file.

        Args:
            video_path: Path to the input video file
            output_path: Path to save the annotated video (optional)
            window_name: Name of the display window
            show_preview: Whether to show live preview while processing

        Returns:
            List of BlinkResult objects for each frame
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_num = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                result = self.process_frame(frame, draw=True)
                results.append(result)

                if result and result.frame is not None:
                    display_frame = result.frame

                    # Add progress info
                    progress = f"Frame: {frame_num}/{total_frames}"
                    cv2.putText(
                        display_frame,
                        progress,
                        (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    if writer:
                        writer.write(display_frame)

                    if show_preview:
                        cv2.imshow(window_name, display_frame)
                else:
                    if writer:
                        writer.write(frame)
                    if show_preview:
                        cv2.imshow(window_name, frame)

                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Processing interrupted by user")
                        break

                # Print progress
                if frame_num % 100 == 0:
                    print(
                        f"Processed {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.1f}%)"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        print(f"Processing complete. Total blinks detected: {self.blink_count}")
        if output_path:
            print(f"Output saved to: {output_path}")

        return results

    def reset(self) -> None:
        """Reset the blink counter and state."""
        self.blink_count = 0
        self.frame_counter = 0
        self.blink_in_progress = False

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Main function demonstrating blink detection usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Blink Detection using MediaPipe")
    parser.add_argument(
        "--mode",
        type=str,
        default="camera",
        choices=["camera", "video"],
        help="Mode: camera (live) or video (file)",
    )
    parser.add_argument("--video", type=str, help="Path to video file (for video mode)")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera ID (for camera mode)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.21,
        help="EAR threshold for blink detection",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Consecutive frames below threshold for blink",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to face_landmarker.task model file",
    )

    args = parser.parse_args()

    # Create detector
    detector = BlinkDetector(
        ear_threshold=args.threshold,
        consecutive_frames=args.consecutive,
        model_path=args.model,
    )

    try:
        if args.mode == "camera":
            print("Starting live camera blink detection...")
            print(f"EAR threshold: {args.threshold}")
            detector.run_on_camera(camera_id=args.camera)

        elif args.mode == "video":
            if not args.video:
                print("Error: Please provide --video path for video mode")
                return
            print(f"Processing video: {args.video}")
            results = detector.run_on_video(
                video_path=args.video, output_path=args.output, show_preview=True
            )
            print(f"Processed {len(results)} frames")

    finally:
        detector.close()


if __name__ == "__main__":
    main()
