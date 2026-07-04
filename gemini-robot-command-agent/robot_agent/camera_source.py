from __future__ import annotations

from pathlib import Path
import os
import sys
import time

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEFT_CAMERA_DIR = REPO_ROOT / "left-robo-arm-camera-test"
if str(LEFT_CAMERA_DIR) not in sys.path:
    sys.path.insert(0, str(LEFT_CAMERA_DIR))

from robotics_camera import CameraConfig, DirectShowCamera  # noqa: E402


class LeftArmCamera:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, device_name: str | None = None):
        # For this one-camera app, the friendly DirectShow name is more robust
        # across USB replugging. Set LEFT_ARM_CAMERA_DEVICE to pin an exact path.
        selected_device = device_name or os.environ.get("LEFT_ARM_CAMERA_DEVICE") or "USB2.0_CAM1"
        self.camera = DirectShowCamera(CameraConfig(device_name=selected_device, width=width, height=height, fps=fps))
        self.device_name = selected_device

    def __enter__(self) -> "LeftArmCamera":
        self.camera.start_background()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.camera.stop()

    def latest(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.camera.latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.03)
        raise TimeoutError("No frame from left arm camera")

    def jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Could not encode camera frame")
        return encoded.tobytes()


def error_frame(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "left arm camera unavailable", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    y = 125
    for chunk in [message[i : i + 90] for i in range(0, len(message), 90)][:8]:
        cv2.putText(frame, chunk, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        y += 32
    return frame
