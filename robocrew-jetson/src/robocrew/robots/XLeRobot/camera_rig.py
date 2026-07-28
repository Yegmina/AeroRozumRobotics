from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


DEPTH_MIN_MM = 200
DEPTH_MAX_MM = 4000


class CameraUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraObservation:
    view: str
    label: str
    jpeg_bytes: bytes
    details: str = ""


class AuxiliaryCameraRig:
    """On-demand side RGB and Orbbec depth snapshots for the robot agent."""

    LABELS = {
        "left": "Left camera view",
        "right": "Right camera view",
        "depth": "Orbbec depth view",
    }

    def __init__(
        self,
        main_camera,
        left_port: str,
        right_port: str,
        depth_port: str,
    ) -> None:
        self.main_camera = main_camera
        self.ports = {
            "left": left_port,
            "right": right_port,
            "depth": depth_port,
        }
        self._capture_lock = threading.RLock()

    def capture(self, view: str) -> CameraObservation:
        if view not in self.ports:
            raise ValueError(f"Unknown camera view: {view}")

        with self._capture_lock:
            if view == "depth":
                image, details = self._capture_depth_with_rgb_paused()
            else:
                image = self._capture_rgb(self.ports[view])
                details = ""

        return CameraObservation(view, self.LABELS[view], image, details)

    def _capture_rgb(self, path: str) -> bytes:
        capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
        try:
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(3):
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        return self._encode_jpeg(frame)
        finally:
            capture.release()

        return self._capture_rgb_ffmpeg(path)

    def _capture_rgb_ffmpeg(self, path: str) -> bytes:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
                    "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
                ],
                capture_output=True,
                timeout=8,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise CameraUnavailable(f"Unable to capture {path}: {error}") from error
        if not result.stdout:
            raise CameraUnavailable(f"Camera {path} returned an empty frame")
        return result.stdout

    def _capture_depth_with_rgb_paused(self) -> tuple[bytes, str]:
        capture_rgbd = getattr(type(self.main_camera), "capture_rgbd_frame", None)
        if capture_rgbd is not None:
            return self.colorize_depth(capture_rgbd(self.main_camera).depth_mm)

        capture_error: Exception | None = None
        result: tuple[bytes, str] | None = None
        self.main_camera.release()
        time.sleep(0.25)
        try:
            frame = self._read_depth_frame(self.ports["depth"])
            result = self.colorize_depth(frame)
        except Exception as error:
            capture_error = error
        finally:
            try:
                self.main_camera.reopen()
            except Exception as reopen_error:
                raise CameraUnavailable(f"Center RGB camera could not be reopened: {reopen_error}") from reopen_error

        if capture_error is not None:
            if isinstance(capture_error, CameraUnavailable):
                raise capture_error
            raise CameraUnavailable(f"Depth capture failed: {capture_error}") from capture_error
        assert result is not None
        return result

    def _read_depth_frame(self, path: str) -> np.ndarray:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
                    "-input_format", "gray16le", "-video_size", "640x480", "-i", path,
                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
                ],
                capture_output=True,
                timeout=8,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise CameraUnavailable(f"Unable to capture depth camera {path}: {error}") from error

        frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if frame is None or frame.ndim != 2:
            raise CameraUnavailable("Depth camera returned an invalid frame")
        return frame

    @staticmethod
    def colorize_depth(frame: np.ndarray) -> tuple[bytes, str]:
        valid = (frame > 0) & (frame < 65535)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise CameraUnavailable("Depth frame contains no valid distance pixels")

        clipped = np.clip(frame.astype(np.float32), DEPTH_MIN_MM, DEPTH_MAX_MM)
        near_high = ((DEPTH_MAX_MM - clipped) * (255.0 / (DEPTH_MAX_MM - DEPTH_MIN_MM))).astype(np.uint8)
        color = cv2.applyColorMap(near_high, cv2.COLORMAP_TURBO)
        color[~valid] = 0

        height, width = frame.shape
        y0, y1 = int(height * 0.4), int(height * 0.6)
        x0, x1 = int(width * 0.4), int(width * 0.6)
        center_values = frame[y0:y1, x0:x1]
        center_valid = center_values[(center_values > 0) & (center_values < 65535)]
        center_mm = int(np.median(center_valid)) if center_valid.size else None
        valid_percent = valid_count * 100.0 / frame.size

        cv2.putText(color, "NEAR red | FAR blue | 0.2-4.0 m", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        center_text = f"Center median: {center_mm / 1000:.2f} m" if center_mm is not None else "Center median: unavailable"
        cv2.putText(color, center_text, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        details = f"Depth scale: near=red, far=blue, clipped to 0.2-4.0 m. Valid pixels: {valid_percent:.1f}%. {center_text}."
        return AuxiliaryCameraRig._encode_jpeg(color), details

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise CameraUnavailable("Camera frame could not be encoded as JPEG")
        return buffer.tobytes()
