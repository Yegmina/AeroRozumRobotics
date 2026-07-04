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
    raw_homography: np.ndarray | None
    keypoints_left: int
    keypoints_right: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    state: str


class PerspectiveMatcher:
    def __init__(
        self,
        max_features: int,
        ratio: float,
        min_matches: int,
        min_inlier_ratio: float,
        ransac_px: float,
        smoothing: float,
        hold_frames: int,
        max_corner_shift: float,
    ):
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=5,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_px = ransac_px
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.max_corner_shift = max_corner_shift
        self._stable_homography: np.ndarray | None = None
        self._frames_since_good = hold_frames + 1

    def match(self, left: np.ndarray, right: np.ndarray) -> MatchResult:
        gray_left = self._prepare_gray(left)
        gray_right = self._prepare_gray(right)
        kp_left, des_left = self.orb.detectAndCompute(gray_left, None)
        kp_right, des_right = self.orb.detectAndCompute(gray_right, None)

        if des_left is None or des_right is None or len(kp_left) < 2 or len(kp_right) < 2:
            canvas = side_by_side(left, right)
            homography, state = self._held_homography()
            return MatchResult(canvas, homography, None, len(kp_left), len(kp_right), 0, 0, 0.0, state)

        good = self._symmetric_ratio_matches(des_left, des_right)

        homography = None
        raw_homography = None
        inlier_mask = None
        inliers = 0
        inlier_ratio = 0.0
        state = "WAIT"
        if len(good) >= self.min_matches:
            src = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            raw_homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
            if inlier_mask is not None:
                inliers = int(inlier_mask.sum())
                inlier_ratio = inliers / max(len(good), 1)
            if (
                raw_homography is not None
                and inliers >= self.min_matches
                and inlier_ratio >= self.min_inlier_ratio
                and self._homography_is_reasonable(raw_homography, left.shape, right.shape)
            ):
                homography = self._update_stable_homography(raw_homography, left.shape)
                state = "LOCK"

        if homography is None:
            homography, state = self._held_homography()

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
        return MatchResult(
            canvas,
            homography,
            raw_homography,
            len(kp_left),
            len(kp_right),
            len(good),
            inliers,
            inlier_ratio,
            state,
        )

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _symmetric_ratio_matches(self, des_left: np.ndarray, des_right: np.ndarray) -> list[cv2.DMatch]:
        forward = self._ratio_matches(des_left, des_right)
        backward = self._ratio_matches(des_right, des_left)
        reverse_pairs = {(m.trainIdx, m.queryIdx) for m in backward}
        symmetric = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_pairs]
        return sorted(symmetric, key=lambda m: m.distance)

    def _ratio_matches(self, source: np.ndarray, target: np.ndarray) -> list[cv2.DMatch]:
        pairs = self.matcher.knnMatch(source, target, k=2)
        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.ratio * second.distance:
                good.append(first)
        return good

    def _update_stable_homography(self, raw_homography: np.ndarray, left_shape: tuple[int, ...]) -> np.ndarray:
        raw = raw_homography / raw_homography[2, 2]
        if self._stable_homography is None:
            self._stable_homography = raw
        else:
            previous = self._stable_homography / self._stable_homography[2, 2]
            if self._corner_shift(previous, raw, left_shape) > self.max_corner_shift:
                self._frames_since_good += 1
                return previous
            alpha = float(np.clip(self.smoothing, 0.0, 1.0))
            smoothed = previous * alpha + raw * (1.0 - alpha)
            self._stable_homography = smoothed / smoothed[2, 2]
        self._frames_since_good = 0
        return self._stable_homography

    def _held_homography(self) -> tuple[np.ndarray | None, str]:
        if self._stable_homography is None:
            return None, "WAIT"
        self._frames_since_good += 1
        if self._frames_since_good <= self.hold_frames:
            return self._stable_homography, "HOLD"
        return None, "WAIT"

    def _homography_is_reasonable(
        self,
        homography: np.ndarray,
        left_shape: tuple[int, ...],
        right_shape: tuple[int, ...],
    ) -> bool:
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            return False
        h_left, w_left = left_shape[:2]
        h_right, w_right = right_shape[:2]
        corners = np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            return False
        area = abs(cv2.contourArea(projected.astype(np.float32)))
        source_area = float(w_left * h_left)
        if area < source_area * 0.05 or area > source_area * 8.0:
            return False
        bounds = np.array([[-w_right * 1.5, -h_right * 1.5], [w_right * 2.5, h_right * 2.5]])
        return bool(
            np.all(projected[:, 0] >= bounds[0, 0])
            and np.all(projected[:, 0] <= bounds[1, 0])
            and np.all(projected[:, 1] >= bounds[0, 1])
            and np.all(projected[:, 1] <= bounds[1, 1])
        )

    def _corner_shift(self, previous: np.ndarray, current: np.ndarray, left_shape: tuple[int, ...]) -> float:
        h, w = left_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        prev_projected = cv2.perspectiveTransform(corners, previous).reshape(-1, 2)
        curr_projected = cv2.perspectiveTransform(corners, current).reshape(-1, 2)
        return float(np.max(np.linalg.norm(prev_projected - curr_projected, axis=1)))


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
    matcher = PerspectiveMatcher(
        args.max_features,
        args.ratio,
        args.min_matches,
        args.min_inlier_ratio,
        args.ransac_px,
        args.smoothing,
        args.hold_frames,
        args.max_corner_shift,
    )

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
                f"ratio={result.inlier_ratio:.2f} H={result.state} fps={shown_fps:.1f}"
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
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--ransac-px", type=float, default=3.0)
    parser.add_argument("--smoothing", type=float, default=0.82, help="Higher keeps the perspective estimate steadier.")
    parser.add_argument("--hold-frames", type=int, default=20, help="Keep the last good homography briefly when matches drop.")
    parser.add_argument("--max-corner-shift", type=float, default=120.0, help="Reject sudden homography jumps in pixels.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
