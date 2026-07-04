# AeroRozum Robotics Camera Feed

This project accesses the camera feed from the left robo arm through the Windows
DirectShow device named `USB2.0_CAM1`.

We avoid OpenCV camera indexes because index `0` opened the laptop webcam on
this machine. The reusable wrapper opens the named USB device through FFmpeg and
returns normal OpenCV BGR `numpy` frames for future AI agent development.

## Live Feed

```powershell
py -3 camera_livefeed.py
```

## Use In Agent Code

```python
from robotics_camera import CameraConfig, DirectShowCamera

config = CameraConfig(device_name="USB2.0_CAM1", width=1280, height=720, fps=30)

with DirectShowCamera(config).start_background() as camera:
    frame = camera.latest_frame()
```

`frame` is a writable OpenCV BGR image. Use `camera.read()` instead when blocking
one-frame-at-a-time access is preferred.

