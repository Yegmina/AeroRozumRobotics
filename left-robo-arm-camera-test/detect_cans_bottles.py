import argparse
from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

from robotics_camera import CameraConfig, DirectShowCamera


DEFAULT_TARGETS = {"bottle", "cup", "wine glass"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


class FasterRcnnDetector:
    def __init__(self, targets: set[str], threshold: float, interval_s: float):
        self.targets = {target.lower() for target in targets}
        self.threshold = threshold
        self.interval_s = interval_s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=threshold).to(self.device)
        self.model.eval()

        self._frame_lock = threading.Lock()
        self._detections_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._detections: list[Detection] = []
        self._last_error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="FasterRcnnDetector", daemon=True)

    def start(self) -> "FasterRcnnDetector":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def submit(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def detections(self) -> tuple[list[Detection], str]:
        with self._detections_lock:
            return list(self._detections), self._last_error

    def _loop(self) -> None:
        last_run = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now - last_run < self.interval_s:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            last_run = now
            try:
                detections = self._predict(frame)
                with self._detections_lock:
                    self._detections = detections
                    self._last_error = ""
            except Exception as exc:
                with self._detections_lock:
                    self._last_error = str(exc)

    def _predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(frame_rgb).to(self.device)
        with torch.inference_mode():
            output = self.model([image])[0]

        detections: list[Detection] = []
        for score, label_id, box in zip(output["scores"], output["labels"], output["boxes"]):
            confidence = float(score)
            if confidence < self.threshold:
                continue
            class_id = int(label_id)
            if class_id >= len(self.categories):
                continue
            label = str(self.categories[class_id]).lower()
            if label not in self.targets:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            detections.append(Detection(label=label, score=confidence, box=(x1, y1, x2, y2)))
        return detections


def parse_targets(value: str) -> set[str]:
    targets = {item.strip().lower() for item in value.split(",") if item.strip()}
    return targets or set(DEFAULT_TARGETS)


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.box
    color = (0, 255, 255) if detection.label == "bottle" else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{detection.label} {detection.score:.2f}"
    cv2.putText(frame, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect bottles/cups/can-like objects from the left robo arm camera.")
    parser.add_argument("--targets", default="bottle,cup,wine glass", help="Comma-separated COCO class names.")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--detect-every", type=float, default=0.7, help="Seconds between Faster R-CNN inference runs.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    config = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    detector = FasterRcnnDetector(targets, args.threshold, args.detect_every).start()

    window = "Left robo arm - Faster R-CNN cans and bottles detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    shown_fps = 0.0
    last = time.perf_counter()
    try:
        with DirectShowCamera(config).start_background() as camera:
            print("Left camera Faster R-CNN detector opened. Press q or Esc to quit.")
            while True:
                frame = camera.latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                detector.submit(frame)
                detections, error = detector.detections()
                for detection in detections:
                    draw_detection(frame, detection)

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    shown_fps = instant_fps if shown_fps == 0 else shown_fps * 0.85 + instant_fps * 0.15

                status = (
                    f"left arm Faster R-CNN | bottle/cup/can-like detections={len(detections)} | "
                    f"{shown_fps:.1f} FPS"
                )
                cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
                if error:
                    cv2.putText(frame, error[:110], (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        detector.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
