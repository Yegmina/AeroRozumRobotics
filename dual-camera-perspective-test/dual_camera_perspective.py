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
    stable_frames: int
    anchors: int


def create_feature_stack(detector: str, max_features: int):
    choice = detector.lower()
    if choice == "auto":
        if hasattr(cv2, "SIFT_create"):
            choice = "sift"
        elif hasattr(cv2, "AKAZE_create"):
            choice = "akaze"
        else:
            choice = "orb"

    if choice == "sift":
        feature_detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.025, edgeThreshold=12)
        return "SIFT", feature_detector, cv2.BFMatcher(cv2.NORM_L2)

    if choice == "akaze":
        feature_detector = cv2.AKAZE_create(threshold=0.0008)
        return "AKAZE", feature_detector, cv2.BFMatcher(cv2.NORM_HAMMING)

    if choice == "orb":
        feature_detector = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=5,
        )
        return "ORB", feature_detector, cv2.BFMatcher(cv2.NORM_HAMMING)

    raise ValueError(f"Unknown detector '{detector}'. Use auto, sift, akaze, or orb.")


class PerspectiveMatcher:
    def __init__(
        self,
        detector: str,
        max_features: int,
        ratio: float,
        min_matches: int,
        min_inlier_ratio: float,
        ransac_px: float,
        smoothing: float,
        hold_frames: int,
        max_corner_shift: float,
        lock_after: int,
        max_anchors: int,
        anchor_min_spacing: float,
        mutual_check: bool,
    ):
        self.detector_name, self.detector, self.matcher = create_feature_stack(detector, max_features)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.ratio = ratio
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_px = ransac_px
        self.ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.max_corner_shift = max_corner_shift
        self.lock_after = lock_after
        self.max_anchors = max_anchors
        self.anchor_min_spacing = anchor_min_spacing
        self.mutual_check = mutual_check
        self._stable_homography: np.ndarray | None = None
        self._locked_homography: np.ndarray | None = None
        self._locked_anchors: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._locked_mode: str | None = None
        self._tracked_left: np.ndarray | None = None
        self._tracked_right: np.ndarray | None = None
        self._previous_left_gray: np.ndarray | None = None
        self._previous_right_gray: np.ndarray | None = None
        self._stable_frames = 0
        self._fundamental_frames = 0
        self._frames_since_good = hold_frames + 1

    def reset(self) -> None:
        self._stable_homography = None
        self._locked_homography = None
        self._locked_anchors = []
        self._locked_mode = None
        self._tracked_left = None
        self._tracked_right = None
        self._previous_left_gray = None
        self._previous_right_gray = None
        self._stable_frames = 0
        self._fundamental_frames = 0
        self._frames_since_good = self.hold_frames + 1

    def match(self, left: np.ndarray, right: np.ndarray) -> MatchResult:
        gray_left = self._prepare_gray(left)
        gray_right = self._prepare_gray(right)

        if self._locked_mode is not None and self._track_locked(gray_left, gray_right, left.shape, right.shape):
            canvas = draw_anchor_matches(left, right, self._locked_anchors)
            return MatchResult(
                canvas,
                self._locked_homography,
                None,
                0,
                0,
                0,
                0,
                0.0,
                self._locked_mode,
                self._stable_frames,
                len(self._locked_anchors),
            )

        kp_left, des_left = self.detector.detectAndCompute(gray_left, None)
        kp_right, des_right = self.detector.detectAndCompute(gray_right, None)

        if des_left is None or des_right is None or len(kp_left) < 2 or len(kp_right) < 2:
            canvas = side_by_side(left, right)
            homography, state = self._held_homography()
            return MatchResult(canvas, homography, None, len(kp_left), len(kp_right), 0, 0, 0.0, state, self._stable_frames, 0)

        good = self._matches(des_left, des_right)

        homography = None
        raw_homography = None
        inlier_mask = None
        inliers = 0
        inlier_ratio = 0.0
        state = "WAIT"
        if len(good) >= self.min_matches:
            src = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            raw_homography, inlier_mask = cv2.findHomography(src, dst, self.ransac_method, self.ransac_px)
            if inlier_mask is not None:
                inliers = int(inlier_mask.sum())
                inlier_ratio = inliers / max(len(good), 1)
            homography_accepted = (
                raw_homography is not None
                and inliers >= self.min_matches
                and inlier_ratio >= self.min_inlier_ratio
                and self._homography_is_reasonable(raw_homography, left.shape, right.shape)
            )
            if homography_accepted:
                homography = self._update_stable_homography(raw_homography, left.shape)
                state = "ACQUIRE"
                anchors = self._build_anchors(kp_left, kp_right, good, inlier_mask)
                if self._stable_frames >= self.lock_after and len(anchors) >= min(self.min_matches, self.max_anchors):
                    self._start_tracking(anchors, gray_left, gray_right, "H_TRACK", homography)
                    state = self._locked_mode
            else:
                f_anchors, f_inliers, f_ratio = self._fundamental_anchors(kp_left, kp_right, good)
                if f_inliers >= self.min_matches and f_ratio >= self.min_inlier_ratio and len(f_anchors) >= min(self.min_matches, self.max_anchors):
                    self._fundamental_frames += 1
                    inliers = f_inliers
                    inlier_ratio = f_ratio
                    state = "F_ACQUIRE"
                    if self._fundamental_frames >= self.lock_after:
                        self._start_tracking(f_anchors, gray_left, gray_right, "F_TRACK", None)
                        state = self._locked_mode
                else:
                    self._fundamental_frames = 0

        if homography is None and state != "F_ACQUIRE":
            homography, state = self._held_homography()
            if self._locked_mode is not None:
                state = self._locked_mode
            elif state == "WAIT":
                self._stable_frames = 0

        if self._locked_mode is not None:
            canvas = draw_anchor_matches(left, right, self._locked_anchors)
        else:
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
            self._stable_frames,
            len(self._locked_anchors),
        )

    def _build_anchors(
        self,
        kp_left,
        kp_right,
        matches: list[cv2.DMatch],
        inlier_mask: np.ndarray | None,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        if inlier_mask is None:
            return []
        anchors: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used_left: list[np.ndarray] = []
        used_right: list[np.ndarray] = []
        mask = inlier_mask.ravel().astype(bool)
        for match, is_inlier in zip(matches, mask):
            if not is_inlier:
                continue
            left_pt = np.array(kp_left[match.queryIdx].pt, dtype=np.float32)
            right_pt = np.array(kp_right[match.trainIdx].pt, dtype=np.float32)
            if self._too_close(left_pt, used_left) or self._too_close(right_pt, used_right):
                continue
            used_left.append(left_pt)
            used_right.append(right_pt)
            anchors.append(
                (
                    (int(round(left_pt[0])), int(round(left_pt[1]))),
                    (int(round(right_pt[0])), int(round(right_pt[1]))),
                )
            )
            if len(anchors) >= self.max_anchors:
                break
        return anchors

    def _start_tracking(
        self,
        anchors: list[tuple[tuple[int, int], tuple[int, int]]],
        gray_left: np.ndarray,
        gray_right: np.ndarray,
        mode: str,
        homography: np.ndarray | None,
    ) -> None:
        self._locked_anchors = anchors
        self._locked_mode = mode
        self._locked_homography = homography
        self._tracked_left = np.float32([left for left, _ in anchors]).reshape(-1, 1, 2)
        self._tracked_right = np.float32([right for _, right in anchors]).reshape(-1, 1, 2)
        self._previous_left_gray = gray_left.copy()
        self._previous_right_gray = gray_right.copy()

    def _track_locked(
        self,
        gray_left: np.ndarray,
        gray_right: np.ndarray,
        left_shape: tuple[int, ...],
        right_shape: tuple[int, ...],
    ) -> bool:
        if (
            self._tracked_left is None
            or self._tracked_right is None
            or self._previous_left_gray is None
            or self._previous_right_gray is None
            or self._locked_mode is None
        ):
            self.reset()
            return False

        lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        next_left, status_left, err_left = cv2.calcOpticalFlowPyrLK(
            self._previous_left_gray,
            gray_left,
            self._tracked_left,
            None,
            **lk_params,
        )
        next_right, status_right, err_right = cv2.calcOpticalFlowPyrLK(
            self._previous_right_gray,
            gray_right,
            self._tracked_right,
            None,
            **lk_params,
        )
        if next_left is None or next_right is None or status_left is None or status_right is None:
            self.reset()
            return False

        keep = (status_left.reshape(-1) == 1) & (status_right.reshape(-1) == 1)
        if err_left is not None:
            keep &= err_left.reshape(-1) < 80.0
        if err_right is not None:
            keep &= err_right.reshape(-1) < 80.0

        left_points = next_left.reshape(-1, 2)[keep]
        right_points = next_right.reshape(-1, 2)[keep]
        if len(left_points) < self.min_matches:
            self.reset()
            return False

        self._tracked_left = left_points.reshape(-1, 1, 2).astype(np.float32)
        self._tracked_right = right_points.reshape(-1, 1, 2).astype(np.float32)
        self._previous_left_gray = gray_left.copy()
        self._previous_right_gray = gray_right.copy()
        self._locked_anchors = [
            ((int(round(left_pt[0])), int(round(left_pt[1]))), (int(round(right_pt[0])), int(round(right_pt[1]))))
            for left_pt, right_pt in zip(left_points, right_points)
        ]

        if self._locked_mode == "H_TRACK" and len(left_points) >= self.min_matches:
            homography, mask = cv2.findHomography(
                left_points.reshape(-1, 1, 2),
                right_points.reshape(-1, 1, 2),
                self.ransac_method,
                self.ransac_px,
            )
            if (
                homography is not None
                and mask is not None
                and int(mask.sum()) >= self.min_matches
                and self._homography_is_reasonable(homography, left_shape, right_shape)
            ):
                self._locked_homography = self._update_stable_homography(homography, left_shape)
            elif self._locked_homography is None:
                self._locked_mode = "F_TRACK"
        return True

    def _fundamental_anchors(self, kp_left, kp_right, matches: list[cv2.DMatch]):
        src = np.float32([kp_left[m.queryIdx].pt for m in matches])
        dst = np.float32([kp_right[m.trainIdx].pt for m in matches])
        try:
            _, mask = cv2.findFundamentalMat(src, dst, self.ransac_method, self.ransac_px, 0.99)
        except cv2.error:
            _, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, self.ransac_px, 0.99)
        if mask is None:
            return [], 0, 0.0
        inliers = int(mask.sum())
        ratio = inliers / max(len(matches), 1)
        return self._build_anchors(kp_left, kp_right, matches, mask), inliers, ratio

    def _too_close(self, point: np.ndarray, existing: list[np.ndarray]) -> bool:
        return any(float(np.linalg.norm(point - other)) < self.anchor_min_spacing for other in existing)

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _symmetric_ratio_matches(self, des_left: np.ndarray, des_right: np.ndarray) -> list[cv2.DMatch]:
        forward = self._ratio_matches(des_left, des_right)
        if not self.mutual_check:
            return sorted(forward, key=lambda m: m.distance)
        backward = self._ratio_matches(des_right, des_left)
        reverse_pairs = {(m.trainIdx, m.queryIdx) for m in backward}
        symmetric = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_pairs]
        return sorted(symmetric, key=lambda m: m.distance)

    def _matches(self, des_left: np.ndarray, des_right: np.ndarray) -> list[cv2.DMatch]:
        return self._symmetric_ratio_matches(des_left, des_right)

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
            self._stable_frames = 1
        else:
            previous = self._stable_homography / self._stable_homography[2, 2]
            if self._corner_shift(previous, raw, left_shape) > self.max_corner_shift:
                self._frames_since_good += 1
                self._stable_frames = 0
                return previous
            alpha = float(np.clip(self.smoothing, 0.0, 1.0))
            smoothed = previous * alpha + raw * (1.0 - alpha)
            self._stable_homography = smoothed / smoothed[2, 2]
            self._stable_frames += 1
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

    def locked(self) -> bool:
        return self._locked_mode is not None


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.hstack([left, right])


def draw_anchor_matches(
    left: np.ndarray,
    right: np.ndarray,
    anchors: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    canvas = side_by_side(left, right)
    offset = left.shape[1]
    colors = [
        (0, 255, 0),
        (0, 220, 255),
        (255, 170, 0),
        (255, 80, 255),
        (80, 180, 255),
        (170, 255, 80),
    ]
    for index, (left_pt, right_pt) in enumerate(anchors):
        color = colors[index % len(colors)]
        right_canvas = (right_pt[0] + offset, right_pt[1])
        cv2.line(canvas, left_pt, right_canvas, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, left_pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, right_canvas, 5, color, -1, cv2.LINE_AA)
    return canvas


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
        args.detector,
        args.max_features,
        args.ratio,
        args.min_matches,
        args.min_inlier_ratio,
        args.ransac_px,
        args.smoothing,
        args.hold_frames,
        args.max_corner_shift,
        args.lock_after,
        args.max_anchors,
        args.anchor_min_spacing,
        args.mutual_check,
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
                f"ratio={result.inlier_ratio:.2f} H={result.state} detector={matcher.detector_name} "
                f"stable={result.stable_frames}/{args.lock_after} anchors={result.anchors} fps={shown_fps:.1f}"
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
            if key == ord("r"):
                matcher.reset()

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time automatic perspective matching between two robo arm cameras.")
    parser.add_argument("--left-device", default=LEFT_DEVICE)
    parser.add_argument("--right-device", default=RIGHT_DEVICE)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detector", default="auto", choices=["auto", "sift", "akaze", "orb"])
    parser.add_argument("--max-features", type=int, default=2600)
    parser.add_argument("--ratio", type=float, default=0.82)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.18)
    parser.add_argument("--ransac-px", type=float, default=6.0)
    parser.add_argument("--smoothing", type=float, default=0.78, help="Higher keeps the perspective estimate steadier.")
    parser.add_argument("--hold-frames", type=int, default=45, help="Keep the last good homography briefly when matches drop.")
    parser.add_argument("--max-corner-shift", type=float, default=260.0, help="Reject sudden homography jumps in pixels.")
    parser.add_argument("--lock-after", type=int, default=3, help="Freeze anchor lines after this many stable homography frames.")
    parser.add_argument("--max-anchors", type=int, default=24, help="Maximum fixed correspondence lines after lock.")
    parser.add_argument("--anchor-min-spacing", type=float, default=38.0, help="Minimum pixel spacing between locked anchors.")
    parser.add_argument("--mutual-check", action="store_true", help="Require matches to agree in both directions. More stable, but harder to lock.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
