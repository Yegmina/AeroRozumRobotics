"""Client and camera adapter for the local Orbbec RGB-D service."""

from __future__ import annotations

import io
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robocrew.core.utils import basic_augmentation
from robocrew.robots.XLeRobot.rgbd_protocol import receive_packet, send_packet


DEFAULT_SOCKET = "/tmp/robocrew-orbbec.sock"
DEFAULT_SERVICE_PYTHON = "/home/jetsonl4/aerorozumdatacollectiondepth/.venv/bin/python"


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: dict[str, float]
    timestamp_ns: int


class RGBDServiceClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, timeout: float = 4.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def _request(self, command: str) -> tuple[dict, bytes]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            send_packet(connection, {"command": command})
            metadata, payload = receive_packet(connection)
        if not metadata.get("ok"):
            raise RuntimeError(metadata.get("error", "RGB-D service request failed"))
        return metadata, payload

    def health(self) -> bool:
        try:
            self._request("health")
            return True
        except (OSError, RuntimeError):
            return False

    def capture(self) -> RGBDFrame:
        metadata, payload = self._request("capture")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            color = arrays["color_bgr"].copy()
            depth = arrays["depth_mm"].copy()
        return RGBDFrame(color, depth, metadata["intrinsics"], int(metadata["timestamp_ns"]))


class RGBDServiceManager:
    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET,
        python_executable: str = DEFAULT_SERVICE_PYTHON,
    ) -> None:
        self.client = RGBDServiceClient(socket_path)
        self.python_executable = python_executable
        self.process: subprocess.Popen | None = None

    def ensure_running(self, timeout: float = 12.0) -> RGBDServiceClient:
        if self.client.health():
            return self.client
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [
                self.python_executable,
                "-m",
                "robocrew.robots.XLeRobot.rgbd_service",
                "--socket",
                self.client.socket_path,
            ],
            env=env,
            stdout=open("/tmp/robocrew-rgbd.log", "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client.health():
                return self.client
            if self.process.poll() is not None:
                break
            time.sleep(0.25)
        raise RuntimeError("Orbbec RGB-D service failed to start; see /tmp/robocrew-rgbd.log")


class AlignedRGBDCamera:
    """Drop-in main camera that also exposes aligned raw depth."""

    def __init__(self, client: RGBDServiceClient):
        self.client = client
        self._latest: RGBDFrame | None = None

    def capture_rgbd_frame(self) -> RGBDFrame:
        self._latest = self.client.capture()
        return self._latest

    def capture_image(self, camera_fov=120, center_angle=0, navigation_mode="normal") -> bytes:
        frame = self.capture_rgbd_frame().color_bgr
        augmented = basic_augmentation(
            frame,
            h_fov=camera_fov,
            center_angle=center_angle,
            navigation_mode=navigation_mode,
        )
        ok, encoded = cv2.imencode(".jpg", augmented)
        if not ok:
            raise RuntimeError("Orbbec RGB frame could not be encoded")
        return encoded.tobytes()

    def release(self) -> None:
        pass

    def reopen(self) -> None:
        if not self.client.health():
            raise RuntimeError("Orbbec RGB-D service is offline")
