import argparse
import time

import cv2
from ultralytics import YOLO

from robotics_camera import CameraConfig, DirectShowCamera


DEFAULT_TARGETS = {"bottle", "cup", "wine glass"}


def parse_targets(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def draw_detection(frame, box, label: str, confidence: float) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    color = (0, 255, 255) if label == "bottle" else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(frame, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect bottles/cups/can-like objects from the left robo arm camera.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument("--targets", default="bottle,cup,wine glass", help="Comma-separated class names to display.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    targets = parse_targets(args.targets) or DEFAULT_TARGETS
    model = YOLO(args.model)
    names = model.names
    target_ids = [class_id for class_id, name in names.items() if str(name).lower() in targets]
    if not target_ids:
        available = ", ".join(str(name) for name in names.values())
        raise RuntimeError(f"No target classes found in {args.model}. Available classes: {available}")

    config = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    window = "Left robo arm - cans and bottles detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    shown_fps = 0.0
    last = time.perf_counter()
    with DirectShowCamera(config).start_background() as camera:
        print("Left camera detector opened. Press q or Esc to quit.")
        while True:
            frame = camera.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=target_ids, verbose=False)
            detections = 0
            for result in results:
                for item in result.boxes:
                    class_id = int(item.cls[0])
                    label = str(names[class_id]).lower()
                    confidence = float(item.conf[0])
                    draw_detection(frame, item.xyxy[0].tolist(), label, confidence)
                    detections += 1

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                instant_fps = 1.0 / dt
                shown_fps = instant_fps if shown_fps == 0 else shown_fps * 0.85 + instant_fps * 0.15

            status = f"left arm camera | bottle/cup/can-like detections={detections} | {shown_fps:.1f} FPS"
            cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
