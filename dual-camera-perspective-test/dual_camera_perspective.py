from __future__ import annotations

import argparse
from dataclasses import dataclass
import shutil
import subprocess
import threading
import time
from typing import Iterator

import cv2
import numpy as np


LEFT_DEVICE = (
    r"@device_pnp_\\?\usb#vid_05a3&pid_9230&mi_00#6&37e57649&0&0000"
    r"#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
)
RIGHT_DEVICE = (
    r"@device_pnp_\\?\usb#vid_05a3&pid_9230&mi_00#6&1eb6a216&0&0000"
    r"#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    label: str
    device_name: str
    width: int = 640
    height: int = 480
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
    def __init__(self, config: CameraConfig):
        self.config = config
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
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
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
        proc = self._require_process()
        deadline = time.monotonic() + timeout_s
        chunks: list[bytes] = []
        remaining = self.config.frame_bytes
        while remaining:
            if proc.poll() is not None:
                raise self._process_error("Camera process exited")
            if time.monotonic() > deadline:
                raise CameraError(f"Timed out waiting for {self.config.label} frame")
            chunk = proc.stdout.read(remaining)
            if not chunk:
                time.sleep(0.005)
                continue
            chunks.append(chunk)
            remaining -= len(chunk)
        frame = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(self.config.frame_shape)
        return frame.copy()

    def frames(self) -> Iterator[np.ndarray]:
        while self.is_running:
            yield self.read()

    def start_background(self) -> "DirectShowCamera":
        self.start()
        if self._reader_thread and self._reader_thread.is_alive():
            return self
        self._reader_thread = threading.Thread(target=self._read_loop, name=f"{self.config.label}Camera", daemon=True)
        self._reader_thread.start()
        return self

    def latest_frame(self) -> np.ndarray | None:
        if self._reader_error is not None:
            raise CameraError(f"{self.config.label} reader failed: {self._reader_error}") from self._reader_error
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

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
            raise CameraError(f"{self.config.label} camera is not started")
        return self._proc

    def _process_error(self, prefix: str) -> CameraError:
        stderr = ""
        if self._proc and self._proc.stderr:
            try:
                stderr = self._proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                stderr = ""
        detail = f": {stderr}" if stderr else ""
        return CameraError(f"{prefix} for {self.config.label}{detail}")


@dataclass
class MatchResult:
    canvas: np.ndarray
    homography: np.ndarray | None
    keypoints_left: int
    keypoints_right: int
    good_matches: int
    inliers: int


class PerspectiveMatcher:
    def __init__(self, max_features: int, ratio: float, min_matches: int, ransac_px: float):
        self.orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.min_matches = min_matches
        self.ransac_px = ransac_px

    def match(self, left: np.ndarray, right: np.ndarray) -> MatchResult:
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        kp_left, des_left = self.orb.detectAndCompute(gray_left, None)
        kp_right, des_right = self.orb.detectAndCompute(gray_right, None)

        if des_left is None or des_right is None or len(kp_left) < 2 or len(kp_right) < 2:
            canvas = side_by_side(left, right)
            return MatchResult(canvas, None, len(kp_left), len(kp_right), 0, 0)

        pairs = self.matcher.knnMatch(des_left, des_right, k=2)
        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.ratio * second.distance:
                good.append(first)
        good = sorted(good, key=lambda m: m.distance)

        homography = None
        inlier_mask = None
        inliers = 0
        if len(good) >= self.min_matches:
            src = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
            if inlier_mask is not None:
                inliers = int(inlier_mask.sum())
            if inliers < self.min_matches:
                homography = None

        draw_matches = good[:100]
        matches_mask = None
        if inlier_mask is not None:
            matches_mask = inlier_mask.ravel().astype(np.uint8).tolist()[: len(draw_matches)]
        canvas = cv2.drawMatches(
            left,
            kp_left,
            right,
            kp_right,
            draw_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(80, 80, 80),
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return MatchResult(canvas, homography, len(kp_left), len(kp_right), len(good), inliers)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.hstack([left, right])


def label(frame: np.ndarray, text: str, origin: tuple[int, int] = (12, 28)) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)


def draw_warp_panel(left: np.ndarray, right: np.ndarray, homography: np.ndarray | None) -> np.ndarray:
    if homography is None:
        blank = np.full_like(right, 24)
        label(blank, "No stable homography yet")
        return np.hstack([right, blank])

    warped = cv2.warpPerspective(left, homography, (right.shape[1], right.shape[0]))
    mask = np.any(warped > 0, axis=2).astype(np.uint8) * 255
    blended = right.copy()
    blended[mask > 0] = cv2.addWeighted(right, 0.50, warped, 0.50, 0)[mask > 0]
    label(blended, "Right + warped left")
    return np.hstack([right, blended])


def run(args) -> None:
    left_cfg = CameraConfig("LEFT", args.left_device, args.width, args.height, args.fps)
    right_cfg = CameraConfig("RIGHT", args.right_device, args.width, args.height, args.fps)
    matcher = PerspectiveMatcher(args.max_features, args.ratio, args.min_matches, args.ransac_px)

    window = "Dual robo arm perspective matcher"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last = time.perf_counter()
    shown_fps = 0.0
    with DirectShowCamera(left_cfg).start_background() as left_camera, DirectShowCamera(right_cfg).start_background() as right_camera:
        print("Dual camera perspective matcher opened. Press q or Esc to quit.")
        while True:
            left = left_camera.latest_frame()
            right = right_camera.latest_frame()
            if left is None or right is None:
                time.sleep(0.01)
                continue

            result = matcher.match(left, right)
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                instant_fps = 1.0 / dt
                shown_fps = instant_fps if shown_fps == 0 else shown_fps * 0.85 + instant_fps * 0.15

            status = (
                f"kp L/R={result.keypoints_left}/{result.keypoints_right} "
                f"matches={result.good_matches} inliers={result.inliers} "
                f"H={'OK' if result.homography is not None else 'WAIT'} fps={shown_fps:.1f}"
            )
            label(result.canvas, "LEFT", (12, 28))
            label(result.canvas, "RIGHT", (left.shape[1] + 12, 28))
            label(result.canvas, status, (12, result.canvas.shape[0] - 14))

            warp_panel = draw_warp_panel(left, right, result.homography)
            label(warp_panel, "RIGHT", (12, 28))
            combined = np.vstack([result.canvas, warp_panel])
            cv2.imshow(window, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time automatic perspective matching between two robo arm cameras.")
    parser.add_argument("--left-device", default=LEFT_DEVICE)
    parser.add_argument("--right-device", default=RIGHT_DEVICE)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=1800)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--ransac-px", type=float, default=4.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
