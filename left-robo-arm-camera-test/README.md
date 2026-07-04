# AeroRozum Robotics Camera Feed

This project accesses the camera feed from the left robo arm through its unique
Windows DirectShow device path.

We avoid OpenCV camera indexes because index `0` opened the laptop webcam on
this machine. Both arm cameras report the same friendly name, `USB2.0_CAM1`, so
the reusable wrapper opens the left USB camera by its DirectShow device path
through FFmpeg and returns normal OpenCV BGR `numpy` frames for future AI agent
development.

## Live Feed

```powershell
py -3 camera_livefeed.py
```

## Use In Agent Code

```python
from robotics_camera import CameraConfig, DirectShowCamera

config = CameraConfig(width=1280, height=720, fps=30)

with DirectShowCamera(config).start_background() as camera:
    frame = camera.latest_frame()
```

`frame` is a writable OpenCV BGR image. Use `camera.read()` instead when blocking
one-frame-at-a-time access is preferred.
