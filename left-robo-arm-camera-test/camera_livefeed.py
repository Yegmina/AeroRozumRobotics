import argparse
import time

import cv2

from robotics_camera import CameraConfig, DirectShowCamera, draw_status


def run_livefeed(config: CameraConfig) -> None:
    window_name = f"{config.device_name} livefeed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last = time.perf_counter()
    shown_fps = 0.0

    print(f"Livefeed opened for '{config.device_name}'. Press q or Esc to quit.")
    with DirectShowCamera(config) as camera:
        for frame in camera.frames():
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                instant_fps = 1.0 / dt
                shown_fps = instant_fps if shown_fps == 0 else (shown_fps * 0.85 + instant_fps * 0.15)

            draw_status(
                frame,
                f"{config.device_name}  {config.width}x{config.height}  {shown_fps:0.1f} FPS",
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the robotics USB camera livefeed.")
    parser.add_argument("--device-name", default="USB2.0_CAM1", help="DirectShow camera device name.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = CameraConfig(
        device_name=args.device_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    run_livefeed(config)


if __name__ == "__main__":
    main()
