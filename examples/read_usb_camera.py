import time

from robotics_camera import CameraConfig, DirectShowCamera


def main():
    config = CameraConfig(device_name="USB2.0_CAM1", width=1280, height=720, fps=30)

    with DirectShowCamera(config).start_background() as camera:
        while True:
            frame = camera.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Put robotics perception/control code here.
            height, width = frame.shape[:2]
            print(f"latest frame: {width}x{height}")
            time.sleep(1)


if __name__ == "__main__":
    main()
