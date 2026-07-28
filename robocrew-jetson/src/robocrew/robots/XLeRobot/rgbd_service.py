"""Python 3.10 Orbbec SDK service for synchronized, color-aligned depth."""

from __future__ import annotations

import argparse
import io
import os
import socketserver
import time
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (  # type: ignore[import-not-found]
    AlignFilter,
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from robocrew.robots.XLeRobot.rgbd_protocol import receive_packet, send_packet


class OrbbecPipeline:
    def __init__(self) -> None:
        self.pipeline = Pipeline()
        config = Config()
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(color)
        config.enable_stream(depth)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        try:
            self.pipeline.enable_frame_sync()
        except Exception:
            pass
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.pipeline.start(config)
        self._intrinsics: dict | None = None

    def close(self) -> None:
        self.pipeline.stop()

    def capture(self) -> tuple[dict, bytes]:
        for _ in range(5):
            frames = self.pipeline.wait_for_frames(1200)
            if not frames:
                continue
            frames = self.align.process(frames)
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            height, width = color_frame.get_height(), color_frame.get_width()
            color_rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(height, width, 3)
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            depth_mm = np.rint(depth.astype(np.float32) * depth_frame.get_depth_scale()).astype(np.uint16)
            if depth_mm.shape != (height, width):
                depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)

            if self._intrinsics is None:
                camera = self.pipeline.get_camera_param().rgb_intrinsic
                self._intrinsics = {
                    "width": int(camera.width),
                    "height": int(camera.height),
                    "fx": float(camera.fx),
                    "fy": float(camera.fy),
                    "cx": float(camera.cx),
                    "cy": float(camera.cy),
                }

            encoded = io.BytesIO()
            np.savez_compressed(encoded, color_bgr=color_bgr, depth_mm=depth_mm)
            metadata = {
                "ok": True,
                "timestamp_ns": time.time_ns(),
                "intrinsics": self._intrinsics,
                "alignment": "depth_to_color",
            }
            return metadata, encoded.getvalue()
        raise TimeoutError("Orbbec did not return an aligned RGB-D frame")


class RGBDRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            request, _ = receive_packet(self.request)
            command = request.get("command")
            if command == "health":
                send_packet(self.request, {"ok": True, "service": "orbbec-rgbd"})
            elif command == "capture":
                metadata, payload = self.server.pipeline.capture()  # type: ignore[attr-defined]
                send_packet(self.request, metadata, payload)
            else:
                send_packet(self.request, {"ok": False, "error": f"Unknown command: {command}"})
        except Exception as error:
            send_packet(self.request, {"ok": False, "error": str(error)})


class RGBDUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, pipeline: OrbbecPipeline):
        self.pipeline = pipeline
        super().__init__(socket_path, RGBDRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/robocrew-orbbec.sock")
    args = parser.parse_args()
    socket_path = Path(args.socket)
    if socket_path.exists():
        socket_path.unlink()
    pipeline = OrbbecPipeline()
    server = RGBDUnixServer(str(socket_path), pipeline)
    os.chmod(socket_path, 0o660)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        pipeline.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
