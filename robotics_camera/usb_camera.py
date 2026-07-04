from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import threading
import time
from typing import Iterator

import cv2
import numpy as np


DEFAULT_DEVICE_NAME = "USB2.0_CAM1"


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or stops delivering frames."""


@dataclass(frozen=True)
class CameraConfig:
    device_name: str = DEFAULT_DEVICE_NAME
    width: int = 1280
    height: int = 720
    fps: int = 30
    ffmpeg_path: str = "ffmpeg"
    input_codec: str = "mjpeg"

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3


class DirectShowCamera:
    """Named DirectShow camera reader backed by FFmpeg.

    This avoids fragile OpenCV camera indexes on Windows. Frames are returned as
    OpenCV-compatible BGR numpy arrays.
    """

    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self._proc: subprocess.Popen | None = None
        self._latest: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._reader_error: BaseException | None = None
        self._stop_event = threading.Event()

    def __enter__(self) -> "DirectShowCamera":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> "DirectShowCamera":
        if self.is_running:
            return self

        if shutil.which(self.config.ffmpeg_path) is None:
            raise CameraError(f"FFmpeg not found: {self.config.ffmpeg_path}")

        cmd = [
            self.config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dshow",
            "-video_size",
            f"{self.config.width}x{self.config.height}",
            "-framerate",
            str(self.config.fps),
            "-vcodec",
            self.config.input_codec,
            "-i",
            f"video={self.config.device_name}",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]

        self._stop_event.clear()
        self._reader_error = None
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        self._reader_thread = None

        proc = self._proc
        self._proc = None
        if proc is None:
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    def read(self, timeout_s: float = 5.0) -> np.ndarray:
        """Read one frame. The returned BGR frame is writable."""
        proc = self._require_process()
        deadline = time.monotonic() + timeout_s
        chunks: list[bytes] = []
        remaining = self.config.frame_bytes

        while remaining:
            if proc.poll() is not None:
                raise self._process_error("Camera process exited")

            if time.monotonic() > deadline:
                raise CameraError(f"Timed out waiting for frame from {self.config.device_name}")

            chunk = proc.stdout.read(remaining)
            if not chunk:
                time.sleep(0.01)
                continue
            chunks.append(chunk)
            remaining -= len(chunk)

        frame = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(self.config.frame_shape)
        return frame.copy()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the camera is stopped or an error occurs."""
        while self.is_running:
            yield self.read()

    def start_background(self) -> "DirectShowCamera":
        """Continuously read frames so robotics loops can poll the newest one."""
        self.start()
        if self._reader_thread and self._reader_thread.is_alive():
            return self

        self._reader_thread = threading.Thread(target=self._read_loop, name="DirectShowCamera", daemon=True)
        self._reader_thread.start()
        return self

    def latest_frame(self, copy: bool = True) -> np.ndarray | None:
        """Return the newest background frame, or None before the first frame arrives."""
        if self._reader_error is not None:
            raise CameraError(f"Background camera reader failed: {self._reader_error}") from self._reader_error

        with self._latest_lock:
            if self._latest is None:
                return None
            return self._latest.copy() if copy else self._latest

    def _read_loop(self) -> None:
        try:
            while not self._stop_event.is_set() and self.is_running:
                frame = self.read(timeout_s=2.0)
                with self._latest_lock:
                    self._latest = frame
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._reader_error = exc

    def _require_process(self) -> subprocess.Popen:
        if self._proc is None or self._proc.stdout is None:
            raise CameraError("Camera is not started. Use camera.start() or a with-block first.")
        return self._proc

    def _process_error(self, prefix: str) -> CameraError:
        stderr = ""
        if self._proc and self._proc.stderr:
            try:
                stderr = self._proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                stderr = ""
        detail = f": {stderr}" if stderr else ""
        return CameraError(f"{prefix} for {self.config.device_name}{detail}")


def draw_status(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a compact green status label on a BGR frame."""
    cv2.putText(
        frame,
        text,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
